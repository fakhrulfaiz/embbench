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
    started = jobs["started_at"].dropna()
    finished = jobs["finished_at"].dropna()

    cols = st.columns(4)
    run_ids = sorted({m.run_id for m in manifests})
    cols[0].metric("Run", run_ids[-1] if run_ids else "—")
    cols[1].metric("Models evaluated", int((jobs["usable"]).sum()))
    cols[2].metric(
        "Span",
        ui.fmt_duration(
            (finished.max() - started.min()).total_seconds()
            if not started.empty and not finished.empty
            else None
        ),
    )
    gpu = detect_gpu()
    cols[3].metric("GPU", gpu.label if gpu else "not detected")
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
    if retrieval.empty:
        return

    st.subheader("Retrieval winner by language")
    ui.metric_help(
        "nDCG@10 answers: when a user asks a question, do the right passages land in the "
        "top 10? The delta is the gap against the current production model."
    )

    means = frames.language_means(retrieval, "ndcg", k=10)
    means = frames.add_baseline_delta(means, "ndcg", ["language"])
    baseline = frames.baseline_model_id()

    langs = frames.ordered_languages(means["language"])
    cols = st.columns(len(langs) or 1)
    for col, lang in zip(cols, langs, strict=False):
        block = means[means["language"] == lang].sort_values("ndcg", ascending=False)
        top = block.iloc[0]
        base_row = block[block["model"] == baseline]
        gap = None
        if not base_row.empty:
            gap = float(top["ndcg"]) - float(base_row.iloc[0]["ndcg"])
        col.metric(
            f"{frames.language_label(lang)} nDCG@10",
            f"{top['ndcg']:.4f}",
            delta=None if gap is None else f"{gap:+.4f} vs baseline",
            help=f"Winner: {top['model']}",
        )
        col.caption(f"**{top['model']}**")

    chart_data = means.rename(columns={"language_name": "Language"})
    st.altair_chart(
        ui.grouped_bar(
            chart_data,
            value_col="ndcg",
            group_col="Language",
            group_title="Language",
            value_title="Mean nDCG@10",
        ),
        use_container_width=False,
    )

    sts = frames.sts_frame(results)
    if not sts.empty:
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
