"""Semantic textual similarity scores."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from embbench.dashboard import frames, ui
from embbench.dashboard.state import selected_results


def render() -> None:
    ui.page_header(
        "Semantic similarity (STS)",
        "Do sentences humans called similar get similar vectors? A sanity check, not a "
        "substitute for retrieval.",
    )

    frame = frames.sts_frame(selected_results())
    if frame.empty:
        ui.empty_state("No completed STS tasks yet.")
        return

    view, frame = _controls(frame)
    if frame.empty:
        ui.empty_state("No rows match those filters.")
        return

    _by_language(frame, view)
    _by_task(frame, view)
    _table(frame)
    _missing_languages(frame)


def _controls(frame: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    view = st.radio(
        "Show",
        ["Absolute score", "Gap vs baseline"],
        horizontal=True,
        help=f"Baseline is {frames.baseline_model_id()}, the current production model.",
    )
    with st.expander("Filter languages, tasks, and models"):
        langs = frames.ordered_languages(frame["language"])
        chosen_langs = st.multiselect(
            "Languages", langs, default=langs, format_func=frames.language_label
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
    return view, filtered


def _by_language(frame: pd.DataFrame, view: str) -> None:
    st.subheader("Average by language")
    means = (
        frame.dropna(subset=["score"])
        .groupby(["model", "language", "language_name"], observed=True)["score"]
        .mean()
        .reset_index()
    )
    if means.empty:
        ui.empty_state("No STS scores to average.")
        return
    means = frames.add_baseline_delta(means, "score", ["language"])
    data = means.rename(columns={"language_name": "Language"})

    if view == "Absolute score":
        chart = ui.grouped_bar(
            data,
            value_col="score",
            group_col="Language",
            group_title="Language",
            value_title="Mean STS score",
        )
    else:
        chart = ui.delta_bar(
            data,
            group_col="Language",
            group_title="Language",
            value_title="STS score minus baseline",
        )
    st.altair_chart(chart, use_container_width=False)


def _by_task(frame: pd.DataFrame, view: str) -> None:
    st.subheader("By task")
    subset = frame.dropna(subset=["score"])[["model", "task", "language", "score"]].copy()
    if subset.empty:
        ui.empty_state("No task-level STS scores.")
        return
    subset["task"] = [
        f"{task} ({frames.language_label(lang)})"
        for task, lang in zip(subset["task"], subset["language"], strict=True)
    ]
    if view == "Absolute score":
        chart = ui.heatmap(
            subset, value_col="score", x_col="task", x_title="Task", value_title="STS score"
        )
    else:
        delta = frames.add_baseline_delta(subset, "score", ["task"])
        delta = delta[delta["model"] != frames.baseline_model_id()].dropna(subset=["delta"])
        if delta.empty:
            ui.empty_state("The baseline model has no STS scores to compare against.")
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


def _table(frame: pd.DataFrame) -> None:
    st.subheader("Every score")
    display = frames.add_baseline_delta(frame, "score", ["task"]).copy()
    display["language"] = [frames.language_label(lang) for lang in display["language"]]
    display = display[["model", "role", "language", "task", "metric", "score", "delta"]]
    st.dataframe(
        display.sort_values(["language", "task", "score"], ascending=[True, True, False]),
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "role": st.column_config.TextColumn("Role"),
            "language": st.column_config.TextColumn("Language"),
            "task": st.column_config.TextColumn("Task"),
            "metric": st.column_config.TextColumn("Metric"),
            "score": st.column_config.NumberColumn("Score", format="%.4f"),
            "delta": st.column_config.NumberColumn("vs baseline", format="%+.4f"),
        },
    )


def _missing_languages(frame: pd.DataFrame) -> None:
    """Any language benchmarked for retrieval but not for STS is a real coverage gap."""
    retrieval = frames.retrieval_frame(selected_results())
    expected = set(frames.LANGUAGE_ORDER)
    if not retrieval.empty:
        expected |= {str(code) for code in retrieval["language"].dropna().unique()}

    covered = {str(code) for code in frame["language"]}
    missing = frames.ordered_languages(expected - covered)
    if not missing:
        return

    names = ", ".join(f"**{frames.language_label(lang)}**" for lang in missing)
    st.info(
        f"Benchmarked for retrieval but not for similarity: {names}. MTEB ships no Malay STS "
        "dataset at all. Any such slot stays empty until a folder is dropped into "
        "`data/sts/<name>/`.",
        icon=":material/info:",
    )
