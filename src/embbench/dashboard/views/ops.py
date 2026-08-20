"""Serving-path cost: VRAM, encode throughput, and query latency."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from embbench.dashboard import frames, ui
from embbench.dashboard.hardware import detect_gpu
from embbench.dashboard.state import selected_results


def render() -> None:
    ui.page_header(
        "Ops and cost",
        "What each model costs to serve: GPU memory, encode throughput, query latency.",
    )

    results = selected_results()
    if not results:
        ui.empty_state("No results found under `results/`.")
        return

    _vram(results)
    st.divider()
    _throughput(results)
    st.divider()
    _profiles(results)


def _vram(results: list) -> None:
    st.subheader("Peak GPU memory")
    gpu = detect_gpu()
    budget = gpu.total_memory_gb if gpu else None
    ui.metric_help(
        "Measured across the benchmark tasks themselves, so it reflects a real encode workload."
        + (f" This box has {gpu.label}." if gpu else "")
    )

    frame = frames.vram_frame(results)
    if frame.empty:
        ui.empty_state("No VRAM measurements recorded.")
        return

    order = ui.model_order(frame)
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("peak_vram_gb:Q", title="Peak VRAM (GiB)"),
            y=alt.Y("model:N", sort=order, title=None),
            color=alt.Color("model:N", sort=order, legend=None),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("peak_vram_gb:Q", title="Peak GiB", format=".2f"),
                alt.Tooltip("mean_vram_gb:Q", title="Mean GiB", format=".2f"),
                alt.Tooltip("peak_task:N", title="Heaviest task"),
            ],
        )
        .properties(height=40 * len(frame) + 40)
    )
    if budget is not None:
        limit = alt.Chart(pd.DataFrame({"x": [budget]})).mark_rule(strokeDash=[6, 4]).encode(x="x:Q")
        st.altair_chart(bars + limit, use_container_width=True)
    else:
        st.altair_chart(bars, use_container_width=True)

    columns = ["model", "role", "peak_vram_gb", "mean_vram_gb", "peak_task"]
    display = frame
    if budget is not None:
        display = frame.assign(headroom=budget - frame["peak_vram_gb"])
        columns.insert(4, "headroom")
    st.dataframe(
        display[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "role": st.column_config.TextColumn("Role"),
            "peak_vram_gb": st.column_config.NumberColumn("Peak GiB", format="%.2f"),
            "mean_vram_gb": st.column_config.NumberColumn("Mean GiB", format="%.2f"),
            "headroom": st.column_config.NumberColumn("Headroom GiB", format="%.2f"),
            "peak_task": st.column_config.TextColumn("Heaviest task"),
        },
    )

    missing = [
        res.spec.model_id
        for res in results
        if not any(t.peak_vram_gb is not None for t in res.tasks)
    ]
    if missing:
        st.caption(
            f"No VRAM figure for {', '.join(f'`{m}`' for m in missing)} because the job never "
            "completed a task."
        )


def _throughput(results: list) -> None:
    st.subheader("Encode time on the benchmark itself")
    ui.metric_help(
        "Total seconds each model spent encoding and scoring, summed over its tasks. This is "
        "wall time on this GPU, not a throughput benchmark, but it ranks the models the same way."
    )

    jobs = frames.jobs_frame(results)
    frame = jobs.dropna(subset=["task_seconds"])
    if frame.empty:
        ui.empty_state("No per-task timings recorded.")
        return

    order = ui.model_order(frame)
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("task_seconds:Q", title="Seconds across all tasks"),
            y=alt.Y("model:N", sort=order, title=None),
            color=alt.Color("model:N", sort=order, legend=None),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("task_seconds:Q", title="Seconds", format=".0f"),
                alt.Tooltip("tasks:Q", title="Tasks"),
            ],
        )
        .properties(height=40 * len(frame) + 40)
    )
    st.altair_chart(chart, use_container_width=True)

    display = frame.assign(compute=[ui.fmt_duration(v) for v in frame["task_seconds"]])
    st.dataframe(
        display[["model", "role", "tasks", "compute"]],
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "role": st.column_config.TextColumn("Role"),
            "tasks": st.column_config.NumberColumn("Tasks"),
            "compute": st.column_config.TextColumn("Total encode + score time"),
        },
    )


def _profiles(results: list) -> None:
    st.subheader("Serving profile")
    frame = frames.ops_frame(results)
    profiled = {res.spec.model_id for res in results if res.ops is not None}
    skipped = sorted({res.spec.model_id for res in results} - profiled)

    if frame.empty:
        st.warning(
            "No model carries a serving profile. Add `--profile-ops` to a run to measure "
            "encode throughput, index build time, and query latency.",
            icon=":material/warning:",
        )
        return

    st.dataframe(
        frame[
            [
                "model",
                "n_docs",
                "embed_dim",
                "encode_docs_per_s",
                "index_build_s",
                "p50_query_ms",
                "p95_query_ms",
                "p99_query_ms",
                "backend",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "n_docs": st.column_config.NumberColumn("Docs indexed"),
            "embed_dim": st.column_config.NumberColumn("Dim"),
            "encode_docs_per_s": st.column_config.NumberColumn("Docs/s", format="%.1f"),
            "index_build_s": st.column_config.NumberColumn("Index build (s)", format="%.2f"),
            "p50_query_ms": st.column_config.NumberColumn("p50 (ms)", format="%.3f"),
            "p95_query_ms": st.column_config.NumberColumn("p95 (ms)", format="%.3f"),
            "p99_query_ms": st.column_config.NumberColumn("p99 (ms)", format="%.3f"),
            "backend": st.column_config.TextColumn("Backend"),
        },
    )

    for _, row in frame.iterrows():
        if row["notes"]:
            st.caption(f"`{row['model']}`: {row['notes']}")

    if (frame["backend"] == "exact").any():
        st.info(
            "These latencies come from exact in-memory search, not Qdrant, so treat them as a "
            "floor rather than a production number. The profile also indexes only a few hundred "
            "documents, which is far smaller than a real corpus.",
            icon=":material/info:",
        )
    if skipped:
        st.caption(
            f"No serving profile for {', '.join(f'`{m}`' for m in skipped)}. "
            "Only runs launched with `--profile-ops` that reached the profiling step have one."
        )
