"""Tidy DataFrames built from JobResult objects and the MTEB cache.

Every number here is read from what the benchmark already wrote. Nothing is
recomputed, so the dashboard and `report.md` cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import streamlit as st

from embbench.core.registry import ModelConfig, load_models
from embbench.core.schemas import JobResult

LANGUAGE_NAMES = {"eng": "English", "cmn": "Chinese", "zsm": "Malay", "msa": "Malay"}
LANGUAGE_ORDER = ["eng", "cmn", "zsm"]


def language_label(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def ordered_languages(codes: Iterable[str]) -> list[str]:
    """Project languages first, then anything new, so a future language still shows up."""
    present = {str(code) for code in codes if code is not None and str(code) != "nan"}
    known = [code for code in LANGUAGE_ORDER if code in present]
    return known + sorted(present - set(known))


@st.cache_data(show_spinner=False)
def _models_by_id() -> dict[str, ModelConfig]:
    try:
        return {m.id: m for m in load_models()}
    except Exception:
        return {}


def model_meta(model_id: str) -> ModelConfig | None:
    return _models_by_id().get(model_id)


def model_id_for_hf(hf_name: str) -> str:
    """MTEB writes its cache under HF names; show the friendly id instead."""
    for model_id, cfg in _models_by_id().items():
        if cfg.hf_name == hf_name:
            return model_id
    return hf_name


def baseline_model_id() -> str | None:
    """The incumbent every other model is measured against."""
    for model_id, cfg in _models_by_id().items():
        if cfg.role == "baseline":
            return model_id
    return None


def _role(model_id: str) -> str:
    cfg = model_meta(model_id)
    return cfg.role if cfg else "unknown"


def is_usable(result: JobResult) -> bool:
    """A job only counts if it produced at least one task that did not error."""
    return (
        result.status == "completed"
        and bool(result.tasks)
        and not all(t.error for t in result.tasks)
    )


def jobs_frame(results: list[JobResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        peaks = [t.peak_vram_gb for t in res.tasks if t.peak_vram_gb is not None]
        if not peaks and res.ops and res.ops.peak_vram_gb is not None:
            peaks = [res.ops.peak_vram_gb]
        elapsed = [t.elapsed_s for t in res.tasks if t.elapsed_s is not None]
        duration = None
        if res.started_at and res.finished_at:
            duration = (res.finished_at - res.started_at).total_seconds()
        rows.append(
            {
                "model": res.spec.model_id,
                "role": _role(res.spec.model_id),
                "hf_name": res.model_hf_name or "",
                "job_id": res.spec.job_id,
                "status": res.status,
                "usable": is_usable(res),
                "tasks": len(res.tasks),
                "failed_tasks": sum(1 for t in res.tasks if t.error),
                "peak_vram_gb": max(peaks) if peaks else None,
                "task_seconds": sum(elapsed) if elapsed else None,
                "wall_seconds": duration,
                "started_at": res.started_at,
                "finished_at": res.finished_at,
                "has_ops": res.ops is not None,
                "error": (res.error or "").split("\n")[0],
            }
        )
    return pd.DataFrame(rows)


def retrieval_frame(results: list[JobResult]) -> pd.DataFrame:
    """One row per (model, task, k) with both nDCG and Recall at that k."""
    rows = []
    for res in results:
        if not is_usable(res):
            continue
        for task in res.tasks:
            if task.task_type != "Retrieval" or task.error:
                continue
            for k in sorted({*task.ndcg, *task.recall}, key=_as_int):
                rows.append(
                    {
                        "model": res.spec.model_id,
                        "role": _role(res.spec.model_id),
                        "task": task.name,
                        "language": task.language,
                        "language_name": language_label(task.language),
                        "source": task.source,
                        "k": _as_int(k),
                        "ndcg": task.ndcg.get(k),
                        "recall": task.recall.get(k),
                        "peak_vram_gb": task.peak_vram_gb,
                        "elapsed_s": task.elapsed_s,
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["language"] = pd.Categorical(
            frame["language"],
            categories=LANGUAGE_ORDER + sorted(set(frame["language"]) - set(LANGUAGE_ORDER)),
            ordered=True,
        )
    return frame


def sts_frame(results: list[JobResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        if not is_usable(res):
            continue
        for task in res.tasks:
            if task.task_type != "STS" or task.error:
                continue
            rows.append(
                {
                    "model": res.spec.model_id,
                    "role": _role(res.spec.model_id),
                    "task": task.name,
                    "language": task.language,
                    "language_name": language_label(task.language),
                    "source": task.source,
                    "score": task.main_score,
                    "metric": task.main_score_name or "cosine spearman",
                    "elapsed_s": task.elapsed_s,
                }
            )
    return pd.DataFrame(rows)


def ops_frame(results: list[JobResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        ops = res.ops
        if ops is None:
            continue
        rows.append(
            {
                "model": res.spec.model_id,
                "role": _role(res.spec.model_id),
                "n_docs": ops.n_docs,
                "embed_dim": ops.embed_dim,
                "peak_vram_gb": ops.peak_vram_gb,
                "encode_docs_per_s": ops.encode_docs_per_s,
                "encode_wall_s": ops.encode_wall_s,
                "index_build_s": ops.index_build_s,
                "index_size_bytes": ops.index_size_bytes,
                "p50_query_ms": ops.p50_query_ms,
                "p95_query_ms": ops.p95_query_ms,
                "p99_query_ms": ops.p99_query_ms,
                "exact_recall_at_30": ops.exact_recall_at_30,
                "qdrant_recall_at_30": ops.qdrant_recall_at_30,
                "ann_recall_delta": ops.ann_recall_delta,
                "backend": ops.backend,
                "notes": "; ".join(ops.notes),
            }
        )
    return pd.DataFrame(rows)


def vram_frame(results: list[JobResult]) -> pd.DataFrame:
    """Peak VRAM per model taken from the task-level measurements."""
    rows = []
    for res in results:
        peaks = [(t.name, t.peak_vram_gb) for t in res.tasks if t.peak_vram_gb is not None]
        if not peaks:
            continue
        task, peak = max(peaks, key=lambda pair: pair[1])
        rows.append(
            {
                "model": res.spec.model_id,
                "role": _role(res.spec.model_id),
                "peak_vram_gb": peak,
                "peak_task": task,
                "mean_vram_gb": sum(p for _, p in peaks) / len(peaks),
            }
        )
    return pd.DataFrame(rows)


def mteb_detail_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Long format: one row per (model, task, split, subset, metric)."""
    rows = []
    for rec in records:
        for metric, value in rec["metrics"].items():
            rows.append(
                {
                    "model": model_id_for_hf(rec["model"]),
                    "hf_name": rec["model"],
                    "task": rec["task"],
                    "split": rec["split"],
                    "subset": rec["subset"],
                    "languages": rec["languages"],
                    "metric": metric,
                    "value": value,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    family, at_k = zip(*(_split_metric(m) for m in frame["metric"]), strict=True)
    frame["family"] = family
    frame["at_k"] = at_k
    return frame


def _split_metric(metric: str) -> tuple[str, int | None]:
    """`ndcg_at_10` -> ("ndcg", 10). Metrics without a k keep `None`."""
    if "_at_" not in metric:
        return metric, None
    head, _, tail = metric.rpartition("_at_")
    try:
        return head, int(tail)
    except ValueError:
        return metric, None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def language_means(
    frame: pd.DataFrame,
    value_col: str,
    *,
    k: int | None = None,
) -> pd.DataFrame:
    """Mean score per (model, language), matching how `report.md` averages."""
    if frame.empty:
        return pd.DataFrame()
    subset = frame if k is None else frame[frame["k"] == k]
    subset = subset.dropna(subset=[value_col])
    if subset.empty:
        return pd.DataFrame()
    grouped = (
        subset.groupby(["model", "language", "language_name"], observed=True)[value_col]
        .mean()
        .reset_index()
    )
    return grouped


def overall_means(frame: pd.DataFrame, *, k: int | None = None) -> pd.DataFrame:
    """One row per model: nDCG and Recall, each the unweighted mean of language means.

    Languages count equally even when one has fewer tasks, so a small Chinese set
    does not get drowned by English.
    """
    ndcg = language_means(frame, "ndcg", k=k)
    recall = language_means(frame, "recall", k=k)
    if ndcg.empty and recall.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    if not ndcg.empty:
        parts.append(
            ndcg.groupby("model", observed=True)["ndcg"].mean().rename("ndcg").to_frame()
        )
    if not recall.empty:
        parts.append(
            recall.groupby("model", observed=True)["recall"].mean().rename("recall").to_frame()
        )
    return pd.concat(parts, axis=1).reset_index()


def add_baseline_delta(
    frame: pd.DataFrame,
    value_col: str,
    key_cols: list[str],
    *,
    baseline: str | None = None,
) -> pd.DataFrame:
    """Append a `delta` column: score minus the baseline model's score on the same key."""
    baseline = baseline or baseline_model_id()
    out = frame.copy()
    out["delta"] = pd.NA
    if not baseline or frame.empty or baseline not in set(frame["model"]):
        return out
    ref = (
        frame[frame["model"] == baseline]
        .set_index(key_cols)[value_col]
        .groupby(level=list(range(len(key_cols))), observed=True)
        .first()
    )
    keys = pd.MultiIndex.from_frame(out[key_cols]) if len(key_cols) > 1 else pd.Index(out[key_cols[0]])
    out["delta"] = out[value_col].to_numpy() - ref.reindex(keys).to_numpy()
    out.loc[out["model"] == baseline, "delta"] = pd.NA
    return out
