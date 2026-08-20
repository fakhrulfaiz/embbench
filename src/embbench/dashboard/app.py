"""Streamlit entry point. Launch with `uv run embbench dashboard`.

Read-only: it renders what the benchmark already wrote to `results/` and never
writes, re-scores, or touches the GPU.
"""

from __future__ import annotations

# The /mnt/c HuggingFace cache guard must run before anything imports settings.
from embbench.core.config import bootstrap_env

bootstrap_env()

import streamlit as st  # noqa: E402

from embbench.dashboard.state import sidebar_controls  # noqa: E402
from embbench.dashboard.views import (  # noqa: E402
    artifacts,
    explorer,
    ops,
    overview,
    retrieval,
    sts,
)

# Every page callable is named `render`, so Streamlit cannot infer distinct URLs.
PAGES = [
    (overview.render, "Overview", ":material/dashboard:", "overview"),
    (retrieval.render, "Retrieval", ":material/search:", "retrieval"),
    (sts.render, "Similarity", ":material/compare_arrows:", "similarity"),
    (explorer.render, "Metric explorer", ":material/query_stats:", "metrics"),
    (ops.render, "Ops and cost", ":material/memory:", "ops"),
    (artifacts.render, "Artifacts", ":material/folder_open:", "artifacts"),
]


def main() -> None:
    st.set_page_config(
        page_title="Embedding benchmark",
        page_icon=":material/insights:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    sidebar_controls()
    pages = [
        st.Page(fn, title=title, icon=icon, url_path=path, default=index == 0)
        for index, (fn, title, icon, path) in enumerate(PAGES)
    ]
    st.navigation(pages).run()


main()
