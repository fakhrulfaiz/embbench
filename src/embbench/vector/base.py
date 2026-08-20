"""Scoring path: exact in-memory cosine search. Never used through Qdrant."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorBackend(Protocol):
    name: str

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def stats(self) -> dict[str, Any]: ...

    def drop(self) -> None: ...
