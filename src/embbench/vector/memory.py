"""Exact brute-force cosine index. This is the scoring backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from embbench.vector.base import VectorBackend


class ExactMemoryBackend:
    name = "exact"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        del payloads
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        matrix = matrix / norms
        if self._vectors is None:
            self._ids = list(ids)
            self._vectors = matrix
            return
        self._ids.extend(ids)
        self._vectors = np.concatenate([self._vectors, matrix], axis=0)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._vectors is None or not self._ids:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        qn = np.linalg.norm(query)
        if qn > 0:
            query = query / qn
        scores = self._vectors @ query
        k = min(k, scores.shape[0])
        if k <= 0:
            return []
        # argpartition is exact for the top-k set; sort that slice.
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self._ids[i], float(scores[i])) for i in idx]

    def stats(self) -> dict[str, Any]:
        n = 0 if self._vectors is None else int(self._vectors.shape[0])
        dim = 0 if self._vectors is None else int(self._vectors.shape[1])
        nbytes = 0 if self._vectors is None else int(self._vectors.nbytes)
        return {"n_vectors": n, "dim": dim, "size_bytes": nbytes, "backend": self.name}

    def drop(self) -> None:
        self._ids = []
        self._vectors = None
