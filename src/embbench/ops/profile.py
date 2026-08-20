"""VRAM / throughput / Qdrant serving-path profile. Not used for nDCG/Recall."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from embbench.core.encoders import Encoder
from embbench.core.schemas import OpsProfile
from embbench.evaluation.metrics import evaluate_run
from embbench.vector.memory import ExactMemoryBackend


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        return None


def _reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def encode_matrix(model: Encoder, texts: list[str], batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = model.encode(batch, batch_size=batch_size, show_progress_bar=False)
        chunks.append(np.asarray(encoded, dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def profile_ops(
    model: Encoder,
    corpus_texts: list[str],
    query_texts: list[str] | None = None,
    batch_size: int = 16,
    k: int = 30,
    n_latency_queries: int = 200,
    use_qdrant: bool = True,
) -> OpsProfile:
    if not corpus_texts:
        return OpsProfile(n_docs=0, embed_dim=0, notes=["empty corpus"])

    queries = query_texts or corpus_texts[: min(len(corpus_texts), max(n_latency_queries, 32))]
    ids = [str(i) for i in range(len(corpus_texts))]

    _reset_peak()
    t0 = time.perf_counter()
    doc_vectors = encode_matrix(model, corpus_texts, batch_size)
    encode_wall = time.perf_counter() - t0
    peak = _peak_vram_gb()
    dim = int(doc_vectors.shape[1]) if doc_vectors.size else 0
    docs_per_s = len(corpus_texts) / encode_wall if encode_wall > 0 else None

    query_vectors = encode_matrix(model, queries, batch_size)

    exact = ExactMemoryBackend()
    t_index = time.perf_counter()
    exact.upsert(ids, doc_vectors)
    exact_build = time.perf_counter() - t_index
    exact_stats = exact.stats()

    exact_run: dict[str, dict[str, float]] = {}
    qrels: dict[str, dict[str, int]] = {}
    latencies_ms: list[float] = []

    n_lat = min(n_latency_queries, len(queries), len(doc_vectors))
    for i in range(n_lat):
        qrels[str(i)] = {ids[i % len(ids)]: 1}
        tq = time.perf_counter()
        hits = exact.search(query_vectors[i], k=k)
        latencies_ms.append((time.perf_counter() - tq) * 1000)
        exact_run[str(i)] = {doc: score for doc, score in hits}

    exact_metrics = evaluate_run(qrels, exact_run, [k])
    exact_recall = exact_metrics.get(f"recall@{k}")

    notes: list[str] = []
    qdrant_recall = None
    p50 = p95 = p99 = None
    index_build_s = exact_build
    size_bytes = exact_stats.get("size_bytes")
    backend_name = "exact"

    if use_qdrant:
        try:
            from embbench.vector.qdrant import QdrantBackend

            qdrant = QdrantBackend()
            t_index = time.perf_counter()
            step = 256
            for start in range(0, len(ids), step):
                qdrant.upsert(ids[start : start + step], doc_vectors[start : start + step])
            index_build_s = time.perf_counter() - t_index
            q_lat: list[float] = []
            q_run: dict[str, dict[str, float]] = {}
            for i in range(n_lat):
                tq = time.perf_counter()
                hits = qdrant.search(query_vectors[i], k=k)
                q_lat.append((time.perf_counter() - tq) * 1000)
                q_run[str(i)] = {doc: score for doc, score in hits}
            q_metrics = evaluate_run(qrels, q_run, [k])
            qdrant_recall = q_metrics.get(f"recall@{k}")
            latencies_ms = q_lat
            backend_name = "qdrant"
            size_bytes = qdrant.stats().get("size_bytes")
            qdrant.drop()
        except Exception as exc:
            notes.append(f"qdrant unavailable ({exc}); latency is exact in-memory")

    if latencies_ms:
        arr = np.sort(np.asarray(latencies_ms))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))

    ann_delta = None
    if exact_recall is not None and qdrant_recall is not None:
        ann_delta = float(exact_recall) - float(qdrant_recall)

    return OpsProfile(
        n_docs=len(corpus_texts),
        embed_dim=dim,
        peak_vram_gb=peak,
        encode_docs_per_s=docs_per_s,
        encode_wall_s=encode_wall,
        index_build_s=index_build_s,
        index_size_bytes=size_bytes,
        p50_query_ms=p50,
        p95_query_ms=p95,
        p99_query_ms=p99,
        exact_recall_at_30=exact_recall,
        qdrant_recall_at_30=qdrant_recall,
        ann_recall_delta=ann_delta,
        backend=backend_name,
        notes=notes,
    )
