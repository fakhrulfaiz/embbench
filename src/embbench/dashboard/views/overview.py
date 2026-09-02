"""Overview: the one page that answers 'which model should we ship'."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from embbench.dashboard import frames, loader, ui
from embbench.dashboard.hardware import detect_gpu
from embbench.dashboard.state import selected_results


def render() -> None:
    ui.page_header(
        "Overview",
        "Which embedding model wins, by how much, and what did not finish.",
    )

    results = selected_results()
    if not results:
        ui.empty_state(
            "No results found under `results/`. Run `uv run embbench run --all` first."
        )
        return

    jobs = frames.jobs_frame(results)
    _run_header(jobs)
    _status_section(jobs)
    _manifest_disagreement(jobs)
    _headline(results)
    _failures(jobs)


def _run_header(jobs: pd.DataFrame) -> None:
    manifests = loader.load_run_manifests()

    cols = st.columns(3)
    run_ids = sorted({m.run_id for m in manifests})
    cols[0].metric("Run", run_ids[-1] if run_ids else "—")
    cols[1].metric("Models evaluated", int((jobs["usable"]).sum()))
    gpu = detect_gpu()
    cols[2].metric("GPU", gpu.label if gpu else "not detected")
    st.caption(
        "Scoring is exact in-memory search. Qdrant is the serving path only and is never "
        "mixed into nDCG or Recall."
    )


def _status_section(jobs: pd.DataFrame) -> None:
    st.subheader("Jobs")
    display = jobs.assign(
        status=[f"{ui.STATUS_ICONS.get(s, '•')} {s}" for s in jobs["status"]],
        role=[ui.ROLE_LABELS.get(r, r) for r in jobs["role"]],
        peak_vram_gb=[ui.fmt(v, 2) for v in jobs["peak_vram_gb"]],
        compute=[ui.fmt_duration(v) for v in jobs["task_seconds"]],
    )
    st.dataframe(
        display[
            ["model", "role", "status", "tasks", "failed_tasks", "peak_vram_gb", "compute", "error"]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "role": st.column_config.TextColumn("Role"),
            "status": st.column_config.TextColumn("Status"),
            "tasks": st.column_config.NumberColumn("Tasks scored"),
            "failed_tasks": st.column_config.NumberColumn("Tasks errored"),
            "peak_vram_gb": st.column_config.TextColumn("Peak VRAM (GiB)"),
            "compute": st.column_config.TextColumn("Encode + score time"),
            "error": st.column_config.TextColumn("Error", width="large"),
        },
    )


def _manifest_disagreement(jobs: pd.DataFrame) -> None:
    """The orchestrator log predates the exit-code fix and can claim success wrongly."""
    manifests = loader.load_run_manifests()
    if manifests.__len__() == 0 or jobs.empty:
        return
    truth = dict(zip(jobs["job_id"], jobs["status"], strict=True))
    mismatches = []
    for manifest in manifests:
        for entry in manifest.entries:
            actual = truth.get(entry.job_id)
            if actual and actual != entry.status:
                mismatches.append(
                    f"`{entry.model_id}`: run manifest says **{entry.status}**, "
                    f"`result.json` says **{actual}**"
                )
    if not mismatches:
        return
    st.warning(
        "The orchestrator manifest disagrees with the job results below. "
        "`result.json` is the source of truth; the manifest recorded these before the "
        "runner learned to propagate a non-zero exit code.\n\n" + "\n\n".join(f"- {m}" for m in mismatches),
        icon=":material/warning:",
    )


def _headline(results: list) -> None:
    retrieval = frames.retrieval_frame(results)
    if not retrieval.empty:
        ks = sorted({int(v) for v in retrieval["k"].dropna().unique()})
        k = int(
            st.radio(
                "Cut-off k",
                ks,
                horizontal=True,
                key="overview_k",
                help="Top-k passages considered for both nDCG and Recall.",
            )
        )
        _overall_retrieval(retrieval, k)
        _retrieval_winner(retrieval, k)
    _sts_headline(results)


def _overall_retrieval(retrieval: pd.DataFrame, k: int) -> None:
    st.subheader(f"All languages · nDCG@{k} and Recall@{k}")
    baseline = frames.baseline_model_id()
    ui.metric_help(
        f"Each score is the unweighted mean across languages, so English volume "
        f"does not swamp Chinese. nDCG@{k} rewards rank; Recall@{k} only asks whether "
        f"the gold passage appeared. The small number is how far the winner is ahead of "
        f"`{baseline}`, the model in production today."
        if baseline
        else f"Each score is the unweighted mean across languages. nDCG@{k} rewards "
        f"rank; Recall@{k} only asks whether the gold passage appeared."
    )

    overall = frames.overall_means(retrieval, k=k)
    if overall.empty:
        ui.empty_state(f"No nDCG@{k} or Recall@{k} scores in these results.")
        return

    cards = st.columns(2)
    for col, metric in zip(cards, ("ndcg", "recall"), strict=True):
        if metric not in overall.columns or overall[metric].isna().all():
            col.info(f"No {ui.metric_title(metric, k)} scores.")
            continue
        ranked = overall.dropna(subset=[metric]).sort_values(metric, ascending=False)
        top = ranked.iloc[0]
        gap = None
        if baseline and baseline in set(ranked["model"]):
            base = float(ranked.loc[ranked["model"] == baseline, metric].iloc[0])
            gap = float(top[metric]) - base
        col.metric(
            f"All languages {ui.metric_title(metric, k)}",
            f"{top[metric]:.4f}",
            delta=None if gap is None else f"{gap:+.4f} vs baseline",
            help=f"Winner: {top['model']}",
        )
        col.caption(f"**{top['model']}**")

    long = overall.melt(
        id_vars=["model"],
        value_vars=[c for c in ("ndcg", "recall") if c in overall.columns],
        var_name="Metric",
        value_name="score",
    ).dropna(subset=["score"])
    long["Metric"] = [ui.metric_title(m, k) for m in long["Metric"]]
    st.altair_chart(
        ui.combined_metric_bar(
            long,
            value_col="score",
            series_col="Metric",
            value_title=f"Mean across languages @{k}",
        ),
        use_container_width=True,
    )


def _retrieval_winner(retrieval: pd.DataFrame, k: int) -> None:
    st.subheader("Retrieval winner by language")
    metric_label = st.radio(
        "Metric",
        list(ui.METRIC_LABELS.values()),
        horizontal=True,
        key="overview_metric",
        help="nDCG rewards ranking the right passage higher. Recall only asks if it appeared.",
    )
    metric = next(m for m, text in ui.METRIC_LABELS.items() if text == metric_label)
    baseline = frames.baseline_model_id()
    comparison = (
        f"The small number under each score is how far that winner is ahead of "
        f"`{baseline}`, the model in production today."
        if baseline
        else "No model is marked `role: baseline`, so there is nothing to compare against."
    )
    ui.metric_help(f"{ui.metric_explainer(metric, k)} {comparison}")

    means = frames.language_means(retrieval, metric, k=k)
    if means.empty:
        ui.empty_state(f"No {ui.metric_title(metric, k)} scores in these results.")
        return
    means = frames.add_baseline_delta(means, metric, ["language"])

    langs = frames.ordered_languages(means["language"])
    cols = st.columns(len(langs) or 1)
    for col, lang in zip(cols, langs, strict=False):
        block = means[means["language"] == lang].sort_values(metric, ascending=False)
        top = block.iloc[0]
        base_row = block[block["model"] == baseline]
        gap = None
        if not base_row.empty:
            gap = float(top[metric]) - float(base_row.iloc[0][metric])
        col.metric(
            f"{frames.language_label(lang)} {ui.metric_title(metric, k)}",
            f"{top[metric]:.4f}",
            delta=None if gap is None else f"{gap:+.4f} vs baseline",
            help=f"Winner: {top['model']}",
        )
        col.caption(f"**{top['model']}**")

    chart_data = means.rename(columns={"language_name": "Language"})
    st.altair_chart(
        ui.grouped_bar(
            chart_data,
            value_col=metric,
            group_col="Language",
            group_title="Language",
            value_title=f"Mean {ui.metric_title(metric, k)}",
        ),
        use_container_width=False,
    )


def _sts_headline(results: list) -> None:
    sts = frames.sts_frame(results)
    if sts.empty:
        return
    st.subheader("Semantic similarity (STS)")
    agg = (
        sts.dropna(subset=["score"])
        .groupby(["model", "language_name"], observed=True)["score"]
        .mean()
        .reset_index()
        .rename(columns={"language_name": "Language"})
    )
    st.altair_chart(
        ui.grouped_bar(
            agg,
            value_col="score",
            group_col="Language",
            group_title="Language",
            value_title="Mean STS score",
            height=260,
        ),
        use_container_width=False,
    )
    st.caption(
        "Malay STS is missing from MTEB entirely. That slot stays empty until a "
        "folder is dropped into `data/sts/`."
    )


def _failures(jobs: pd.DataFrame) -> None:
    failed = jobs[jobs["status"] == "failed"]
    if failed.empty:
        return
    st.subheader("What did not finish")
    for _, row in failed.iterrows():
        error = row["error"] or "unknown"
        kind = "ran out of GPU memory" if "out of memory" in error.lower() else "hit a code error"
        st.error(f"**{row['model']}** {kind}.\n\n```\n{error}\n```", icon=":material/error:")
