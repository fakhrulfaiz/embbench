"""Sidebar controls shared by every page."""

from __future__ import annotations

import streamlit as st

from embbench.core.schemas import JobResult
from embbench.dashboard import loader


def sidebar_controls() -> None:
    with st.sidebar:
        st.markdown("### Data")
        st.caption(f"Reading `{loader.results_dir()}`")
        st.toggle(
            "Include smoke tests",
            key="include_smoke",
            value=st.session_state.get("include_smoke", False),
            help="Smoke jobs are one-task sanity checks, not part of the benchmark.",
        )
        if st.button("Reload from disk", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption(
            "Read-only view. Nothing here writes to `results/` or touches the GPU."
        )


def selected_results() -> list[JobResult]:
    return loader.load_results(include_smoke=st.session_state.get("include_smoke", False))
