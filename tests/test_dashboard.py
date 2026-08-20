"""Smoke tests: every dashboard page must render without raising.

These run headlessly through Streamlit's AppTest, so they catch chart and
DataFrame errors that a plain import check would miss. They pass whether or not
`results/` has data, since an empty results directory is a supported state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from embbench.core.config import bootstrap_env

bootstrap_env()

from streamlit.testing.v1 import AppTest  # noqa: E402

import embbench.dashboard.app as dashboard_app  # noqa: E402

PAGES = ["overview", "retrieval", "sts", "explorer", "ops", "artifacts"]

APP_PATH = Path(dashboard_app.__file__)


def _page_script(name: str) -> str:
    return (
        "from embbench.core.config import bootstrap_env\n"
        "bootstrap_env()\n"
        f"from embbench.dashboard.views import {name}\n"
        f"{name}.render()\n"
    )


@pytest.mark.parametrize("name", PAGES)
def test_page_renders(name: str) -> None:
    app = AppTest.from_string(_page_script(name), default_timeout=120).run()
    assert not app.exception, f"{name} raised: {[e.value for e in app.exception]}"


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_with_smoke_jobs(name: str) -> None:
    app = AppTest.from_string(_page_script(name), default_timeout=120)
    app.session_state["include_smoke"] = True
    app.run()
    assert not app.exception, f"{name} raised: {[e.value for e in app.exception]}"


def test_app_entrypoint_runs() -> None:
    """The real entry point, so navigation and the sidebar are covered too."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not app.exception, f"app.py raised: {[e.value for e in app.exception]}"


def test_new_models_and_languages_are_picked_up(tmp_path, monkeypatch) -> None:
    """A future run must appear with no code change.

    Covers three things at once: a model that previously failed and now succeeds,
    a model id absent from `configs/models.yaml`, and a language the project has
    never benchmarked.
    """
    from embbench.core.schemas import JobResult, JobSpec, TaskScore
    from embbench.dashboard import frames, loader

    def write(job_id: str, model_id: str, language: str) -> None:
        result = JobResult(
            spec=JobSpec(job_id=job_id, model_id=model_id),
            status="completed",
            tasks=[
                TaskScore(
                    name=f"{language.title()}Retrieval",
                    task_type="Retrieval",
                    language=language,
                    source="mteb",
                    ndcg={"10": 0.5, "30": 0.6},
                    recall={"10": 0.7, "30": 0.8},
                    peak_vram_gb=1.0,
                    elapsed_s=1.0,
                )
            ],
        )
        target = tmp_path / job_id
        target.mkdir(parents=True)
        (target / "result.json").write_text(result.model_dump_json())

    write("20270101T000000Z-bce-embedding-base_v1", "bce-embedding-base_v1", "eng")
    write("20270101T000000Z-bge-m3", "bge-m3", "eng")
    write("20270101T000000Z-future", "some-future-model-v9", "jpn")

    monkeypatch.setenv("EMBBENCH_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb-cache"))
    import embbench.core.config as config

    monkeypatch.setattr(config, "_settings", None)
    st_cache_cleared = getattr(loader._collect, "clear", None)
    if st_cache_cleared:
        st_cache_cleared()

    results = loader.load_results()
    models = {res.spec.model_id for res in results}
    assert {"bge-m3", "some-future-model-v9"} <= models

    retrieval = frames.retrieval_frame(results)
    assert "jpn" in {str(code) for code in retrieval["language"]}
    # The unregistered model must not be dropped just because models.yaml lacks it.
    assert "some-future-model-v9" in set(retrieval["model"])
    # A brand new language must survive into the filter options, not just the frame.
    assert "jpn" in frames.ordered_languages(retrieval["language"])

    monkeypatch.setattr(config, "_settings", None)
    if st_cache_cleared:
        st_cache_cleared()


def test_ordered_languages_keeps_project_order_then_appends() -> None:
    from embbench.dashboard import frames

    assert frames.ordered_languages(["jpn", "zsm", "eng"]) == ["eng", "zsm", "jpn"]
    assert frames.ordered_languages([]) == []


@pytest.mark.parametrize("name", ["retrieval", "sts"])
def test_baseline_delta_view(name: str) -> None:
    """The delta view builds a different chart than the default absolute view."""
    app = AppTest.from_string(_page_script(name), default_timeout=120).run()
    toggles = [r for r in app.radio if "Gap vs baseline" in r.options]
    if not toggles:
        pytest.skip("no scores loaded, so the toggle is not rendered")
    toggles[0].set_value("Gap vs baseline").run()
    assert not app.exception, f"{name} delta view raised: {[e.value for e in app.exception]}"


def test_retrieval_recall_metric() -> None:
    app = AppTest.from_string(_page_script("retrieval"), default_timeout=120).run()
    metric = [r for r in app.radio if "Recall" in r.options]
    if not metric:
        pytest.skip("no retrieval scores loaded")
    metric[0].set_value("Recall").run()
    assert not app.exception, f"recall view raised: {[e.value for e in app.exception]}"


def test_predictions_drilldown_loads_one_file() -> None:
    """Clicking through to a prediction file must parse and rank without error."""
    app = AppTest.from_string(_page_script("artifacts"), default_timeout=300).run()
    tabs = app.get("tab")
    if not tabs:
        pytest.skip("artifacts page rendered no tabs")
    buttons = [b for b in app.button if b.label.startswith("Load ")]
    if not buttons:
        pytest.skip("no prediction files on disk")
    buttons[0].set_value(True).run()
    assert not app.exception, f"predictions raised: {[e.value for e in app.exception]}"
