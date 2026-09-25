"""Exact brute-force k-NN baseline for recall validation and ground truth."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.atlas_vector.index.metrics import cosine_distance, l2_distance


def matches_filter(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluates whether metadata satisfies where clause conditions."""
    if not where:
        return True
    for key, expected in where.items():
        if key not in metadata or metadata[key] != expected:
            return False
    return True


@dataclass
class FlatIndex:
    dimension: int
    metric: str = "cosine"
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported metric: '{self.metric}'. Must be 'cosine' or 'l2'.")

    def upsert(self, vector_id: str, vector: list[float] | np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dimension,):
            raise ValueError(f"Expected vector dimension {self.dimension}, got {arr.shape[0] if arr.ndim > 0 else 0}")
        self.vectors[vector_id] = arr
        self.metadata[vector_id] = dict(metadata) if metadata else {}
        self.deleted.discard(vector_id)

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self.vectors or vector_id in self.deleted:
            return False
        self.deleted.add(vector_id)
        return True

    def search(
        self,
        query: list[float] | np.ndarray,
        k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be greater than 0")
        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.shape != (self.dimension,):
            raise ValueError(f"Expected query dimension {self.dimension}, got {query_arr.shape[0] if query_arr.ndim > 0 else 0}")

        candidates = [
            vid for vid in self.vectors
            if vid not in self.deleted and matches_filter(self.metadata.get(vid, {}), where)
        ]
        if not candidates:
            return []

        matrix = np.vstack([self.vectors[vid] for vid in candidates])
        if self.metric == "cosine":
            distances = cosine_distance(query_arr, matrix)
        else:
            distances = l2_distance(query_arr, matrix)

        actual_k = min(k, len(candidates))
        order = np.argsort(distances)[:actual_k]
        return [(candidates[int(i)], float(distances[int(i)])) for i in order]

    def stats(self) -> dict[str, Any]:
        return {
            "total_vectors": len(self.vectors),
            "active_vectors": len(self.vectors) - len(self.deleted),
            "deleted_vectors": len(self.deleted),
            "dimension": self.dimension,
            "metric": self.metric,
        }
