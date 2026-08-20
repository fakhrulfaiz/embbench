"""`run --all` must reuse finished work and redo it only when asked.

These tests never load a model. They exercise the lookup that decides whether a
subprocess gets spawned at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from embbench.core.config import bootstrap_env

bootstrap_env()

import embbench.core.config as config  # noqa: E402
from embbench.cli import _find_completed, _scope  # noqa: E402
from embbench.core.schemas import JobResult, JobSpec, TaskScore  # noqa: E402

MODEL = "some-model"


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBBENCH_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("MTEB_CACHE", str(tmp_path / "mteb-cache"))
    monkeypatch.setattr(config, "_settings", None)
    yield tmp_path
    monkeypatch.setattr(config, "_settings", None)


def write_job(
    root,
    job_id: str,
    *,
    status: str = "completed",
    task_error: str | None = None,
    finished: datetime | None = None,
    **spec_kwargs,
) -> JobSpec:
    spec = JobSpec(job_id=job_id, model_id=MODEL, **spec_kwargs)
    result = JobResult(
        spec=spec,
        status=status,
        finished_at=finished or datetime.now(timezone.utc),
        tasks=[
            TaskScore(
                name="SciFact",
                task_type="Retrieval",
                language="eng",
                source="mteb",
                ndcg={"10": 0.5},
                error=task_error,
            )
        ],
    )
    folder = root / job_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(result.model_dump_json())
    return spec


def test_completed_job_is_reused_across_run_ids(results_dir):
    """The whole point: a new run id must still find last week's finished work."""
    write_job(results_dir, "20260101T000000Z-old")
    found = _find_completed(MODEL, JobSpec(job_id="20270101T000000Z-new", model_id=MODEL))
    assert found is not None
    assert found.spec.job_id == "20260101T000000Z-old"


def test_no_results_means_run(results_dir):
    assert _find_completed(MODEL, JobSpec(job_id="new", model_id=MODEL)) is None


def test_failed_job_is_retried(results_dir):
    write_job(results_dir, "20260101T000000Z-old", status="failed")
    assert _find_completed(MODEL, JobSpec(job_id="new", model_id=MODEL)) is None


def test_job_whose_tasks_all_errored_is_retried(results_dir):
    write_job(results_dir, "20260101T000000Z-old", task_error="CUDA out of memory")
    assert _find_completed(MODEL, JobSpec(job_id="new", model_id=MODEL)) is None


def test_other_model_is_not_reused(results_dir):
    write_job(results_dir, "20260101T000000Z-old")
    assert _find_completed("different-model", JobSpec(job_id="new", model_id="different-model")) is None


@pytest.mark.parametrize(
    "narrower",
    [
        {"languages": ["eng"]},
        {"task_types": ["Retrieval"]},
        {"task_names": ["SciFact"]},
        {"include_mteb": False},
        {"include_local": False},
        {"include_heavy": True},
    ],
)
def test_different_scope_is_not_reused(results_dir, narrower):
    """Narrowing or widening a run must never silently reuse a mismatched job."""
    write_job(results_dir, "20260101T000000Z-old")
    wanted = JobSpec(job_id="new", model_id=MODEL, **narrower)
    assert _find_completed(MODEL, wanted) is None


def test_newest_matching_job_wins(results_dir):
    now = datetime.now(timezone.utc)
    write_job(results_dir, "20260101T000000Z-older", finished=now - timedelta(days=10))
    write_job(results_dir, "20260601T000000Z-newer", finished=now)
    found = _find_completed(MODEL, JobSpec(job_id="new", model_id=MODEL))
    assert found is not None
    assert found.spec.job_id == "20260601T000000Z-newer"


def test_scope_ignores_ordering_and_job_identity():
    a = JobSpec(job_id="a", model_id=MODEL, languages=["zsm", "eng"], task_types=["STS", "Retrieval"])
    b = JobSpec(job_id="b", model_id=MODEL, languages=["eng", "zsm"], task_types=["Retrieval", "STS"])
    assert _scope(a) == _scope(b)


def test_scope_excludes_profile_ops_and_overwrite():
    """Neither changes which tasks are scored, so neither should force a re-run."""
    plain = JobSpec(job_id="a", model_id=MODEL)
    flagged = JobSpec(job_id="b", model_id=MODEL, profile_ops=True, overwrite=True)
    assert _scope(plain) == _scope(flagged)
