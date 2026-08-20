"""Shared chart builders and formatting for the dashboard pages."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from embbench.dashboard import frames

ROLE_LABELS = {
    "baseline": "baseline (current production)",
    "predicted-winner": "predicted winner",
    "backup": "backup",
    "reference": "popular reference",
    "hybrid-check": "hybrid check",
}

STATUS_ICONS = {"completed": "✅", "failed": "❌", "running": "⏳", "skipped": "⏭️"}


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def fmt_delta(value: float | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.{digits}f}"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s"


def model_order(frame: pd.DataFrame) -> list[str]:
    """Baseline first so every chart reads as 'incumbent vs challengers'."""
    baseline = frames.baseline_model_id()
    models = sorted(frame["model"].dropna().unique().tolist())
    if baseline in models:
        models.remove(baseline)
        models.insert(0, baseline)
    return models


def grouped_bar(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
    group_title: str,
    value_title: str,
    zero_baseline: bool = True,
    height: int = 320,
) -> alt.Chart:
    """Score by model, faceted along `group_col` (usually language or task)."""
    order = model_order(frame)
    scale = alt.Scale(zero=zero_baseline) if zero_baseline else alt.Scale(zero=False)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y(f"{value_col}:Q", title=value_title, scale=scale),
            color=alt.Color("model:N", sort=order, title="Model"),
            column=alt.Column(f"{group_col}:N", title=group_title),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip(f"{group_col}:N", title=group_title),
                alt.Tooltip(f"{value_col}:Q", title=value_title, format=".4f"),
            ],
        )
        .properties(height=height, width=110)
    )


def delta_bar(
    frame: pd.DataFrame,
    *,
    group_col: str,
    group_title: str,
    value_title: str,
    height: int = 320,
) -> alt.Chart:
    """Signed gap against the baseline. Positive bars beat current production."""
    order = [m for m in model_order(frame) if m != frames.baseline_model_id()]
    data = frame[frame["model"].isin(order)].dropna(subset=["delta"])
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("delta:Q", title=value_title),
            color=alt.Color("model:N", sort=order, title="Model"),
            column=alt.Column(f"{group_col}:N", title=group_title),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip(f"{group_col}:N", title=group_title),
                alt.Tooltip("delta:Q", title="vs baseline", format="+.4f"),
            ],
        )
        .properties(height=height, width=110)
    )


def heatmap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    x_col: str,
    x_title: str,
    value_title: str,
    diverging: bool = False,
    height: int = 260,
) -> alt.Chart:
    order = model_order(frame)
    scheme = "redblue" if diverging else "blues"
    color = alt.Color(
        f"{value_col}:Q",
        title=value_title,
        scale=alt.Scale(scheme=scheme, domainMid=0) if diverging else alt.Scale(scheme=scheme),
    )
    base = alt.Chart(frame).encode(
        x=alt.X(f"{x_col}:N", title=x_title, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("model:N", sort=order, title=None),
    )
    cells = base.mark_rect().encode(
        color=color,
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip(f"{x_col}:N", title=x_title),
            alt.Tooltip(f"{value_col}:Q", title=value_title, format=".4f"),
        ],
    )
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text(f"{value_col}:Q", format=".3f"),
        color=alt.value("#1f2933"),
    )
    return (cells + labels).properties(height=height)


def empty_state(message: str) -> None:
    st.info(message, icon=":material/inbox:")


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def metric_help(text: str) -> None:
    st.caption(text)
