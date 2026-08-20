"""Retrieval scores: nDCG and Recall at k, per language and per task."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from embbench.dashboard import frames, ui
from embbench.dashboard.state import selected_results


def render() -> None:
    ui.page_header(
        "Retrieval",
        "nDCG and Recall at k. This is the metric that decides a RAG pipeline without a reranker.",
    )

    frame = frames.retrieval_frame(selected_results())
    if frame.empty:
        ui.empty_state("No completed retrieval tasks yet.")
        return

    metric, k, view, frame = _controls(frame)
    if frame.empty:
        ui.empty_state("No rows match those filters.")
        return

    _by_language(frame, metric, k, view)
    _by_task(frame, metric, k, view)
    _table(frame, metric, k)


def _controls(frame: pd.DataFrame) -> tuple[str, int, str, pd.DataFrame]:
    cols = st.columns([1, 1, 1.4])
    metric_label = cols[0].radio(
        "Metric",
        ["nDCG", "Recall"],
        horizontal=True,
        help="nDCG rewards ranking the right passage higher. Recall only asks if it appeared.",
    )
    metric = "ndcg" if metric_label == "nDCG" else "recall"

    ks = sorted(int(v) for v in frame["k"].unique())
    k = cols[1].radio("Cut-off k", ks, horizontal=True, help="Top-k passages considered.")

    view = cols[2].radio(
        "Show",
        ["Absolute score", "Gap vs baseline"],
        horizontal=True,
        help=f"Baseline is {frames.baseline_model_id()}, the current production model.",
    )

    with st.expander("Filter languages, tasks, and models"):
        langs = frames.ordered_languages(frame["language"])
        chosen_langs = st.multiselect(
            "Languages",
            langs,
            default=langs,
            format_func=frames.language_label,
        )
        tasks = sorted(frame["task"].unique())
        chosen_tasks = st.multiselect("Tasks", tasks, default=tasks)
        models = ui.model_order(frame)
        chosen_models = st.multiselect("Models", models, default=models)

    filtered = frame[
        frame["language"].isin(chosen_langs)
        & frame["task"].isin(chosen_tasks)
        & frame["model"].isin(chosen_models)
    ]
    return metric, k, view, filtered


def _by_language(frame: pd.DataFrame, metric: str, k: int, view: str) -> None:
    st.subheader(f"{metric.upper()}@{k} by language")
    means = frames.language_means(frame, metric, k=k)
    if means.empty:
        ui.empty_state("Nothing to average for this metric.")
        return
    means = frames.add_baseline_delta(means, metric, ["language"])
    data = means.rename(columns={"language_name": "Language"})

    if view == "Absolute score":
        chart = ui.grouped_bar(
            data,
            value_col=metric,
            group_col="Language",
            group_title="Language",
            value_title=f"Mean {metric.upper()}@{k}",
        )
    else:
        chart = ui.delta_bar(
            data,
            group_col="Language",
            group_title="Language",
            value_title=f"{metric.upper()}@{k} minus baseline",
        )
    st.altair_chart(chart, use_container_width=False)


def _by_task(frame: pd.DataFrame, metric: str, k: int, view: str) -> None:
    st.subheader(f"{metric.upper()}@{k} by task")
    ui.metric_help(
        "Language averages hide per-dataset weakness. A model can win overall and still "
        "lose badly on one domain."
    )
    subset = frame[frame["k"] == k].dropna(subset=[metric])
    if subset.empty:
        ui.empty_state("No task-level scores for this metric.")
        return

    pivot = subset[["model", "task", "language", metric]].copy()
    pivot["task"] = [
        f"{task} ({frames.language_label(lang)})"
        for task, lang in zip(pivot["task"], pivot["language"], strict=True)
    ]

    if view == "Absolute score":
        chart = ui.heatmap(
            pivot,
            value_col=metric,
            x_col="task",
            x_title="Task",
            value_title=f"{metric.upper()}@{k}",
        )
    else:
        delta = frames.add_baseline_delta(pivot, metric, ["task"])
        delta = delta[delta["model"] != frames.baseline_model_id()].dropna(subset=["delta"])
        if delta.empty:
            ui.empty_state("The baseline model has no scores to compare against here.")
            return
        chart = ui.heatmap(
            delta,
            value_col="delta",
            x_col="task",
            x_title="Task",
            value_title="Gap vs baseline",
            diverging=True,
        )
    st.altair_chart(chart, use_container_width=True)


def _table(frame: pd.DataFrame, metric: str, k: int) -> None:
    st.subheader("Every score")
    subset = frame[frame["k"] == k].copy()
    subset = frames.add_baseline_delta(subset, metric, ["task"])
    subset["language"] = [frames.language_label(lang) for lang in subset["language"]]
    display = subset[
        ["model", "role", "language", "task", "ndcg", "recall", "delta", "elapsed_s"]
    ].sort_values(["language", "task", metric], ascending=[True, True, False])
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "role": st.column_config.TextColumn("Role"),
            "language": st.column_config.TextColumn("Language"),
            "task": st.column_config.TextColumn("Task"),
            "ndcg": st.column_config.NumberColumn(f"nDCG@{k}", format="%.4f"),
            "recall": st.column_config.NumberColumn(f"Recall@{k}", format="%.4f"),
            "delta": st.column_config.NumberColumn(
                f"{metric.upper()}@{k} vs baseline", format="%+.4f"
            ),
            "elapsed_s": st.column_config.NumberColumn("Seconds", format="%.0f"),
        },
    )
    st.download_button(
        "Download as CSV",
        display.to_csv(index=False).encode(),
        file_name=f"retrieval_{metric}_at_{k}.csv",
        mime="text/csv",
    )
