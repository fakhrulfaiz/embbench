"""Metric explorer over the raw MTEB cache.

`report.md` keeps four numbers per task. MTEB actually wrote roughly sixty:
nDCG, MAP, Recall, Precision and MRR at k = 1 through 1000. The k-sweep here is
what tells you whether a weak nDCG@10 is a ranking problem a reranker could fix,
or a genuine recall miss that no reranker can recover.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from embbench.dashboard import frames, loader, ui

SWEEP_FAMILIES = {
    "ndcg": "nDCG",
    "recall": "Recall",
    "map": "MAP",
    "precision": "Precision",
    "mrr": "MRR",
}


def render() -> None:
    ui.page_header(
        "Metric explorer",
        "Every metric MTEB computed, at every cut-off. The source is results/mteb-cache.",
    )

    records = loader.load_mteb_records()
    if not records:
        ui.empty_state(
            "No MTEB cache found under `results/mteb-cache/`. It is written during a run."
        )
        return

    frame = frames.mteb_detail_frame(records)
    if frame.empty:
        ui.empty_state("The MTEB cache holds no numeric scores.")
        return

    frame = frame[~frame["metric"].str.startswith("nauc_")]
    _sweep(frame)
    st.divider()
    _table(frame)


def _sweep(frame: pd.DataFrame) -> None:
    st.subheader("Cut-off sweep")

    sweepable = frame.dropna(subset=["at_k"])
    sweepable = sweepable[sweepable["family"].isin(SWEEP_FAMILIES)]
    if sweepable.empty:
        ui.empty_state("This cache has no metrics measured at multiple cut-offs.")
        return

    cols = st.columns([1.6, 1, 1])
    tasks = sorted(sweepable["task"].unique())
    task = cols[0].selectbox("Task", tasks)

    task_rows = sweepable[sweepable["task"] == task]
    families = [f for f in SWEEP_FAMILIES if f in set(task_rows["family"])]
    family = cols[1].selectbox(
        "Metric family", families, format_func=lambda f: SWEEP_FAMILIES.get(f, f)
    )

    splits = sorted(task_rows["split"].unique())
    split = cols[2].selectbox("Split", splits)

    data = task_rows[(task_rows["family"] == family) & (task_rows["split"] == split)]
    subsets = sorted(data["subset"].unique())
    if len(subsets) > 1:
        subset = st.selectbox("Subset", subsets)
        data = data[data["subset"] == subset]

    data = data.dropna(subset=["value"]).copy()
    if data.empty:
        ui.empty_state("No values for that combination.")
        return
    data["at_k"] = data["at_k"].astype(int)

    order = ui.model_order(data)
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "at_k:Q",
                title="Cut-off k (log scale)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format="d"),
            ),
            y=alt.Y("value:Q", title=SWEEP_FAMILIES.get(family, family)),
            color=alt.Color("model:N", sort=order, title="Model"),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("at_k:Q", title="k"),
                alt.Tooltip("value:Q", title=SWEEP_FAMILIES.get(family, family), format=".4f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

    if family == "recall":
        st.caption(
            "Read this as the ceiling. If Recall is already high at k=100 but nDCG@10 is low, "
            "the right passage is being fetched and merely ranked too low, which a reranker "
            "can fix. If Recall stays low even at k=1000, the passage is never retrieved and "
            "no reranker will help."
        )

    wide = (
        data.pivot_table(index="model", columns="at_k", values="value", aggfunc="mean")
        .sort_index(axis=1)
    )
    wide.columns = [f"@{int(c)}" for c in wide.columns]
    st.dataframe(
        wide.reset_index(),
        hide_index=True,
        width="stretch",
        column_config={"model": st.column_config.TextColumn("Model")},
    )


def _table(frame: pd.DataFrame) -> None:
    st.subheader("Full metric table")
    ui.metric_help(
        "Filter by substring, for example `mrr`, `precision_at_5`, or `map`. "
        "Correlation diagnostics (`nauc_*`) are hidden."
    )

    cols = st.columns([1.4, 1, 1])
    tasks = sorted(frame["task"].unique())
    chosen_tasks = cols[0].multiselect("Tasks", tasks, default=tasks[:1] or tasks)
    models = ui.model_order(frame)
    chosen_models = cols[1].multiselect("Models", models, default=models)
    query = cols[2].text_input("Metric contains", "")

    subset = frame[frame["task"].isin(chosen_tasks) & frame["model"].isin(chosen_models)]
    if query:
        subset = subset[subset["metric"].str.contains(query, case=False, regex=False)]
    if subset.empty:
        ui.empty_state("No metrics match those filters.")
        return

    pivot = subset.pivot_table(
        index=["task", "split", "subset", "metric"],
        columns="model",
        values="value",
        aggfunc="mean",
    ).reset_index()
    st.dataframe(pivot, hide_index=True, width="stretch")
    st.download_button(
        "Download as CSV",
        pivot.to_csv(index=False).encode(),
        file_name="mteb_metrics.csv",
        mime="text/csv",
    )
