"""nDCG@k and Recall@k from MTEB prediction dumps (and raw run dicts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytrec_eval


K_DEFAULT = (10, 30)


def metrics_from_prediction_file(
    path: Path,
    qrels: dict[str, dict[str, int]] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    payload = json.loads(path.read_text())
    run, embedded_qrels = _extract_run(payload)
    qrels = qrels or embedded_qrels
    if not qrels:
        return {}
    return evaluate_run(qrels, run, k_values or list(K_DEFAULT))


def evaluate_run(
    qrels: dict[str, dict[str, int | float]],
    run: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    if not qrels or not run:
        return {}
    measures = []
    for k in k_values:
        measures.append(f"ndcg_cut.{k}")
        measures.append(f"recall.{k}")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(measures))
    per_query = evaluator.evaluate(run)
    return _mean(per_query, k_values)


def depth_ok(run: dict[str, dict[str, float]], k: int) -> bool:
    if not run:
        return False
    depths = [len(docs) for docs in run.values()]
    return bool(depths) and min(depths) >= k


def _extract_run(payload: Any) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]] | None]:
    """Accept MTEB dump shapes and always return {qid: {docid: score}}."""
    qrels = None
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
        if isinstance(payload, dict) and "qrels" in payload:
            qrels = payload.get("qrels")
    if isinstance(payload, dict) and "run" in payload:
        qrels = payload.get("qrels") or qrels
        payload = payload["run"]

    if isinstance(payload, dict) and "mteb_model_meta" in payload:
        payload = {k: v for k, v in payload.items() if k != "mteb_model_meta"}

    payload = _unwrap_subset_split(payload)
    run: dict[str, dict[str, float]] = {}
    if isinstance(payload, dict):
        sample = next(iter(payload.values()), None)
        if isinstance(sample, dict):
            for qid, docs in payload.items():
                if not isinstance(docs, dict):
                    continue
                if docs and all(isinstance(v, (int, float)) for v in docs.values()):
                    run[str(qid)] = {str(doc): float(score) for doc, score in docs.items()}
        elif isinstance(sample, list):
            for qid, rows in payload.items():
                scored: dict[str, float] = {}
                for i, row in enumerate(rows):
                    if isinstance(row, dict):
                        doc_id = str(row.get("id") or row.get("doc_id") or row.get("corpus_id"))
                        score = float(row.get("score", 1.0 / (i + 1)))
                    else:
                        doc_id = str(row)
                        score = 1.0 / (i + 1)
                    scored[doc_id] = score
                run[str(qid)] = scored
    return run, qrels


def _unwrap_subset_split(payload: Any) -> Any:
    """MTEB writes {subset: {split: {qid: {docid: score}}}}."""
    if not isinstance(payload, dict) or not payload:
        return payload
    sample = next(iter(payload.values()))
    if not isinstance(sample, dict) or not sample:
        return payload
    inner = next(iter(sample.values()))
    if isinstance(inner, dict) and inner and all(isinstance(v, (int, float)) for v in inner.values()):
        # {qid: {docid: score}} already, but nested one level as {split: run}
        merged: dict[str, dict[str, float]] = {}
        for split_run in payload.values():
            if isinstance(split_run, dict) and split_run and all(
                isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values())
                for v in split_run.values()
            ):
                for qid, docs in split_run.items():
                    merged[str(qid)] = {str(d): float(s) for d, s in docs.items()}
        return merged or payload
    if isinstance(inner, dict):
        # {subset: {split: run}}
        merged = {}
        for subset in payload.values():
            if not isinstance(subset, dict):
                continue
            for split_run in subset.values():
                if isinstance(split_run, dict):
                    for qid, docs in split_run.items():
                        if isinstance(docs, dict) and docs and all(
                            isinstance(v, (int, float)) for v in docs.values()
                        ):
                            merged[str(qid)] = {str(d): float(s) for d, s in docs.items()}
        if merged:
            return merged
    return payload


def _mean(per_query: dict[str, dict[str, float]], k_values: list[int]) -> dict[str, float]:
    if not per_query:
        return {}
    buckets: dict[str, list[float]] = {}
    for metrics in per_query.values():
        for key, value in metrics.items():
            buckets.setdefault(key, []).append(float(value))
    out: dict[str, float] = {}
    for k in k_values:
        ndcg_key = f"ndcg_cut_{k}"
        recall_key = f"recall_{k}"
        if ndcg_key in buckets:
            out[f"ndcg@{k}"] = sum(buckets[ndcg_key]) / len(buckets[ndcg_key])
        if recall_key in buckets:
            out[f"recall@{k}"] = sum(buckets[recall_key]) / len(buckets[recall_key])
    return out


def split_ndcg_recall(scores: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    ndcg = {k.replace("ndcg@", ""): v for k, v in scores.items() if k.startswith("ndcg@")}
    recall = {k.replace("recall@", ""): v for k, v in scores.items() if k.startswith("recall@")}
    return ndcg, recall
