"""Cached, read-only readers for everything under `results/`.

Every reader is keyed on a (path, mtime) signature so a benchmark that finishes
while the server is up shows up on the next rerun without a restart. Nothing
here writes to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from embbench.core.config import get_settings
from embbench.core.schemas import JobResult, JobSpec, RunManifest
from embbench.evaluation.report import collect_results

# Above this, a prediction file is big enough that we make the user opt in.
PREDICTION_WARN_MB = 8.0

Signature = tuple[tuple[str, float], ...]


def results_dir() -> Path:
    return get_settings().results_dir


def _signature(paths: list[Path]) -> Signature:
    out: list[tuple[str, float]] = []
    for path in sorted(paths):
        try:
            out.append((str(path), path.stat().st_mtime))
        except OSError:
            continue
    return tuple(out)


# --------------------------------------------------------------------------
# Job results
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _collect(_sig: Signature, include_smoke: bool) -> list[JobResult]:
    return collect_results(include_smoke=include_smoke)


def load_results(include_smoke: bool = False) -> list[JobResult]:
    """All `result.json` files, parsed with the same code path as `report.md`."""
    root = results_dir()
    if not root.exists():
        return []
    return _collect(_signature(list(root.glob("*/result.json"))), include_smoke)


@st.cache_data(show_spinner=False)
def _read_spec(path: str, _mtime: float) -> JobSpec | None:
    try:
        return JobSpec.model_validate_json(Path(path).read_text())
    except Exception:
        return None


def load_spec(job_id: str) -> JobSpec | None:
    """The JobSpec that produced a job, from `<job>/manifest.json`."""
    path = results_dir() / job_id / "manifest.json"
    if not path.exists():
        return None
    return _read_spec(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _read_run_manifest(path: str, _mtime: float) -> RunManifest | None:
    try:
        return RunManifest.model_validate_json(Path(path).read_text())
    except Exception:
        return None


def load_run_manifests() -> list[RunManifest]:
    """Orchestrator logs: which subprocess ran, in what order, and its exit status."""
    root = results_dir()
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*/run-manifest.json")):
        manifest = _read_run_manifest(str(path), path.stat().st_mtime)
        if manifest is not None:
            out.append(manifest)
    return out


# --------------------------------------------------------------------------
# MTEB cache: the full metric family, far richer than result.json
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _read_mteb_file(path: str, _mtime: float) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def mteb_cache_files() -> list[Path]:
    root = results_dir() / "mteb-cache" / "results"
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/*/*.json") if p.name != "model_meta.json")


def load_mteb_records() -> list[dict[str, Any]]:
    """One record per (model, revision, task, split, subset) with every MTEB metric."""
    records: list[dict[str, Any]] = []
    for path in mteb_cache_files():
        raw = _read_mteb_file(str(path), path.stat().st_mtime)
        if not raw:
            continue
        model_slug = path.parent.parent.name.replace("__", "/")
        revision = path.parent.name
        task_name = raw.get("task_name", path.stem)
        scores = raw.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        for split, entries in scores.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                records.append(
                    {
                        "model": model_slug,
                        "revision": revision,
                        "task": task_name,
                        "split": split,
                        "subset": entry.get("hf_subset", "default"),
                        "languages": ", ".join(entry.get("languages", []) or []),
                        "metrics": {
                            k: v
                            for k, v in entry.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        },
                        "mteb_version": raw.get("mteb_version"),
                        "dataset_revision": raw.get("dataset_revision"),
                        "path": str(path),
                    }
                )
    return records


# --------------------------------------------------------------------------
# Markdown reports
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _read_text(path: str, _mtime: float) -> str:
    return Path(path).read_text()


def load_markdown(name: str) -> str | None:
    path = results_dir() / name
    if not path.exists():
        return None
    return _read_text(str(path), path.stat().st_mtime)


def markdown_reports() -> list[Path]:
    root = results_dir()
    if not root.exists():
        return []
    return sorted(root.glob("*.md"))


# --------------------------------------------------------------------------
# Raw JSON viewer
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_json(path: str, _mtime: float) -> Any:
    return json.loads(Path(path).read_text())


def read_json_file(path: Path) -> Any:
    return load_json(str(path), path.stat().st_mtime)


# --------------------------------------------------------------------------
# Predictions: hundreds of MB per job, so listing is cheap and loading is opt-in
# --------------------------------------------------------------------------


def list_predictions(job_id: str) -> list[tuple[str, Path, int]]:
    """(task name, path, size in bytes) without opening a single file."""
    folder = results_dir() / job_id / "predictions"
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.glob("*_predictions.json")):
        task = path.name.removesuffix("_predictions.json")
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        out.append((task, path, size))
    return out


@st.cache_data(show_spinner=False, max_entries=2)
def load_predictions(path: str, _mtime: float) -> dict[str, Any]:
    """Parse one prediction file. Bounded to 2 entries so memory stays sane."""
    return json.loads(Path(path).read_text())


def prediction_runs(payload: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Flatten a prediction payload into (subset, split, {query_id: {doc_id: score}}).

    Layout is `{"mteb_model_meta": {...}, "<subset>": {"<split>": {...}}}`. STS
    files carry no per-query rankings, so they flatten to nothing.
    """
    out = []
    for subset, splits in payload.items():
        if subset == "mteb_model_meta" or not isinstance(splits, dict):
            continue
        for split, queries in splits.items():
            if not isinstance(queries, dict) or not queries:
                continue
            first = next(iter(queries.values()))
            if isinstance(first, dict):
                out.append((subset, split, queries))
    return out


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def directory_listing() -> list[dict[str, Any]]:
    """Every file under `results/`, with a plain-language note on what it is."""
    root = results_dir()
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name == ".gitkeep":
            continue
        rel = path.relative_to(root)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        rows.append(
            {
                "file": str(rel),
                "kind": describe_artifact(rel),
                "size": human_size(size),
                "bytes": size,
            }
        )
    return rows


def describe_artifact(rel: Path) -> str:
    name = rel.name
    parts = rel.parts
    if name == "result.json":
        return "Job result: per-task scores, VRAM, errors"
    if name == "manifest.json":
        return "Job spec: what was requested for this run"
    if name == "run-manifest.json":
        return "Orchestrator log: subprocess order and status"
    if name.endswith("_predictions.json"):
        return "Per-query ranked doc scores (large)"
    if "mteb-cache" in parts:
        if name == "model_meta.json":
            return "MTEB model metadata"
        return "MTEB raw scores: full metric family at every k"
    if name.endswith(".md"):
        return "Markdown report"
    return "Other artifact"
