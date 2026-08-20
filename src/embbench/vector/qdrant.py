"""Qdrant backend. Serving-path metrics only; never used for nDCG/Recall scoring."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np

from embbench.core.config import get_settings


class QdrantBackend:
    name = "qdrant"

    def __init__(self, collection: str | None = None, url: str | None = None) -> None:
        from qdrant_client import QdrantClient

        settings = get_settings()
        self.collection = collection or f"embbench-{uuid4().hex[:8]}"
        self.client = QdrantClient(
            url=url or settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            check_compatibility=False,
        )
        self._dim: int | None = None
        self._n = 0

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        from qdrant_client.http import models as qm

        matrix = np.asarray(vectors, dtype=np.float32)
        dim = int(matrix.shape[1])
        if self._dim is None:
            self._ensure_collection(dim)
            self._dim = dim
        points = []
        for i, doc_id in enumerate(ids):
            payload = {"doc_id": doc_id}
            if payloads and i < len(payloads) and payloads[i]:
                payload.update(payloads[i])
            points.append(
                qm.PointStruct(
                    id=_point_id(doc_id, i + self._n),
                    vector=matrix[i].tolist(),
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)
        self._n += len(ids)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist()
        # qdrant-client 1.14+ removed Client.search; query_points is the current API.
        response = self.client.query_points(
            collection_name=self.collection,
            query=query,
            limit=k,
            with_payload=True,
        )
        points = getattr(response, "points", response)
        out: list[tuple[str, float]] = []
        for hit in points:
            payload = hit.payload or {}
            doc_id = str(payload.get("doc_id") or hit.id)
            out.append((doc_id, float(hit.score)))
        return out

    def stats(self) -> dict[str, Any]:
        info = self.client.get_collection(self.collection)
        vectors_count = getattr(info, "points_count", None) or getattr(info, "vectors_count", 0)
        disk = None
        try:
            disk = info.segments_count
        except Exception:
            disk = None
        return {
            "n_vectors": int(vectors_count or 0),
            "dim": self._dim,
            "size_bytes": None,
            "segments": disk,
            "backend": self.name,
            "collection": self.collection,
        }

    def drop(self) -> None:
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self._dim = None
        self._n = 0

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.http import models as qm

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )


def _point_id(doc_id: str, fallback: int) -> int | str:
    # Qdrant accepts unsigned int or UUID. Hash string ids into a stable unsigned 64-bit-ish int.
    try:
        return int(doc_id)
    except ValueError:
        return abs(hash(doc_id)) % (2**63 - 1) or fallback
