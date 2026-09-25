"""Hierarchical Navigable Small World (HNSW) graph index built from scratch.

Implements the Malkov & Yashunin (2018) algorithm with:
- Layered navigable graphs (l = 0 .. max_level)
- Configurable M, M0 = 2*M, efConstruction, efSearch
- Vectorized L2 (Euclidean) and Cosine distance metrics
- Heuristic neighbor selection with directional diversity pruning
- Deterministic random seed reproducibility
- Soft deletes (tombstones) and metadata attribute filtering
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.atlas_vector.index.flat import matches_filter
from src.atlas_vector.index.metrics import pairwise_cosine, pairwise_l2


@dataclass
class HNSWIndex:
    dimension: int
    metric: str = "cosine"
    m: int = 16
    ef_construction: int = 100
    seed: int = 42
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    levels: dict[str, int] = field(default_factory=dict)
    # links: {vector_id: {layer_idx: set(neighbor_ids)}}
    links: dict[str, dict[int, set[str]]] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    entry_point: str | None = None
    max_level: int = -1

    def __post_init__(self) -> None:
        if self.metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported metric: '{self.metric}'. Must be 'cosine' or 'l2'.")
        if self.m < 2:
            raise ValueError("m must be at least 2")
        if self.ef_construction < self.m:
            raise ValueError("ef_construction must be >= m")

        self.m0 = 2 * self.m
        self._ml = 1.0 / math.log(float(self.m))
        self._rng = random.Random(self.seed)

    def _dist(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calculates distance between two vectors according to configured metric."""
        if self.metric == "cosine":
            return pairwise_cosine(u, v)
        return pairwise_l2(u, v)

    def _dist_id(self, query: np.ndarray, vector_id: str) -> float:
        """Calculates distance between query array and stored vector by ID."""
        return self._dist(query, self.vectors[vector_id])

    def _random_level(self) -> int:
        """Assigns a maximum graph layer level using exponential distribution."""
        u = self._rng.random()
        u = max(u, 1e-12)
        return int(-math.log(u) * self._ml)

    def _max_m(self, layer: int) -> int:
        """Returns max connections allowed at given layer (M0 at layer 0, M above)."""
        return self.m0 if layer == 0 else self.m

    def upsert(
        self,
        vector_id: str,
        vector: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Inserts or updates a vector in the HNSW index."""
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dimension,):
            raise ValueError(
                f"Expected vector dimension {self.dimension}, got {arr.shape[0] if arr.ndim > 0 else 0}"
            )

        # If vector already exists, perform clean in-place update by soft-deleting previous state
        if vector_id in self.vectors:
            self.delete(vector_id)

        level = self._random_level()
        self.vectors[vector_id] = arr
        self.metadata[vector_id] = dict(metadata) if metadata else {}
        self.levels[vector_id] = level
        self.links[vector_id] = {layer: set() for layer in range(level + 1)}
        self.deleted.discard(vector_id)

        # First node in index
        if self.entry_point is None or len(self.vectors) == 1:
            self.entry_point = vector_id
            self.max_level = level
            return

        curr_obj = self.entry_point
        curr_dist = self._dist_id(arr, curr_obj)

        # Phase 1: Top-down greedy routing from max_level down to level + 1
        for layer in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.links[curr_obj].get(layer, ()):
                    dist = self._dist_id(arr, neighbor)
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_obj = neighbor
                        changed = True

        # Phase 2: Layer-by-layer insertion from min(level, max_level) down to 0
        ep_candidates = [(curr_dist, curr_obj)]
        for layer in range(min(level, self.max_level), -1, -1):
            max_m = self._max_m(layer)
            # Find nearest candidates at this layer
            candidates = self._search_layer_candidates(arr, ep_candidates, self.ef_construction, layer)

            # Select neighbors using heuristic (diversity pruning)
            neighbors = self._select_neighbors_heuristic(arr, candidates, max_m)

            # Create bi-directional connections
            for neighbor in neighbors:
                self.links[vector_id][layer].add(neighbor)
                self.links[neighbor][layer].add(vector_id)
                # Prune neighbor connections if exceeding max_m
                if len(self.links[neighbor][layer]) > max_m:
                    self._prune_neighbors(neighbor, layer, max_m)

            ep_candidates = candidates

        if level > self.max_level:
            self.max_level = level
            self.entry_point = vector_id

    def _search_layer_candidates(
        self,
        query: np.ndarray,
        entry_points: list[tuple[float, str]],
        ef: int,
        layer: int,
    ) -> list[tuple[float, str]]:
        """Algorithm 2 (SEARCH-LAYER): Finds ef nearest neighbors at a specific layer.

        Returns list of (distance, vector_id) sorted by distance ascending.
        """
        visited = set()
        c_min_heap: list[tuple[float, str]] = []  # Min-heap of candidates to visit
        w_max_heap: list[tuple[float, str]] = []  # Max-heap (-dist, id) of best found

        for dist, ep in entry_points:
            visited.add(ep)
            heapq.heappush(c_min_heap, (dist, ep))
            heapq.heappush(w_max_heap, (-dist, ep))

        while c_min_heap:
            cand_dist, cand_id = heapq.heappop(c_min_heap)
            furthest_dist = -w_max_heap[0][0]

            if cand_dist > furthest_dist:
                break

            for neighbor in self.links[cand_id].get(layer, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                n_dist = self._dist_id(query, neighbor)
                furthest_dist = -w_max_heap[0][0]

                if n_dist < furthest_dist or len(w_max_heap) < ef:
                    heapq.heappush(c_min_heap, (n_dist, neighbor))
                    heapq.heappush(w_max_heap, (-n_dist, neighbor))

                    if len(w_max_heap) > ef:
                        heapq.heappop(w_max_heap)

        # Convert max-heap into ascending list by distance
        results = [(-neg_d, node) for neg_d, node in w_max_heap]
        results.sort(key=lambda item: item[0])
        return results

    def _select_neighbors_heuristic(
        self,
        base_vec: np.ndarray,
        candidates: list[tuple[float, str]],
        max_m: int,
    ) -> list[str]:
        """Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC): Selects diverse neighbors.

        Ensures chosen neighbors are not clustered in the same directional subspace
        by checking if candidate is closer to base_vec than to already chosen neighbors.
        """
        if len(candidates) <= max_m:
            return [node for _, node in candidates]

        chosen: list[str] = []
        discarded: list[str] = []

        # Sort candidates ascending by distance to base vector
        sorted_candidates = sorted(candidates, key=lambda item: item[0])

        for dist_to_base, cand_id in sorted_candidates:
            cand_vec = self.vectors[cand_id]
            is_diverse = True
            for selected_id in chosen:
                dist_between = self._dist(cand_vec, self.vectors[selected_id])
                if dist_between < dist_to_base:
                    is_diverse = False
                    break

            if is_diverse:
                chosen.append(cand_id)
                if len(chosen) >= max_m:
                    break
            else:
                discarded.append(cand_id)

        # If diversity constraint left available slots, fill from discarded to maintain connectivity
        if len(chosen) < max_m and discarded:
            for cand_id in discarded:
                chosen.append(cand_id)
                if len(chosen) >= max_m:
                    break

        return chosen

    def _prune_neighbors(self, vector_id: str, layer: int, max_m: int) -> None:
        """Prunes neighbor list of a node at a given layer to respect maximum degree."""
        current_neighbors = self.links[vector_id][layer]
        if len(current_neighbors) <= max_m:
            return

        base_vec = self.vectors[vector_id]
        candidates = [(self._dist(base_vec, self.vectors[nid]), nid) for nid in current_neighbors]
        pruned = set(self._select_neighbors_heuristic(base_vec, candidates, max_m))
        removed = current_neighbors - pruned
        self.links[vector_id][layer] = pruned

        # Symmetrically prune reverse connections to maintain bi-directional link integrity
        for removed_id in removed:
            if removed_id in self.links and layer in self.links[removed_id]:
                self.links[removed_id][layer].discard(vector_id)

    def search(
        self,
        query: list[float] | np.ndarray,
        k: int = 10,
        ef_search: int = 50,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Performs approximate k-NN search on the HNSW graph with metadata filtering."""
        if k <= 0:
            raise ValueError("k must be greater than 0")

        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.shape != (self.dimension,):
            raise ValueError(
                f"Expected query dimension {self.dimension}, got {query_arr.shape[0] if query_arr.ndim > 0 else 0}"
            )

        if not self.entry_point or not self.vectors:
            return []

        curr_obj = self.entry_point
        curr_dist = self._dist_id(query_arr, curr_obj)

        # Top-down greedy search to layer 1
        for layer in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.links[curr_obj].get(layer, ()):
                    dist = self._dist_id(query_arr, neighbor)
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_obj = neighbor
                        changed = True

        # Layer 0 search with ef_search
        ef = max(ef_search, k)
        ep = [(curr_dist, curr_obj)]
        raw_candidates = self._search_layer_candidates(query_arr, ep, ef, layer=0)

        # Filter out tombstones and metadata mismatches
        filtered_results: list[tuple[str, float]] = []
        for dist, vid in raw_candidates:
            if vid in self.deleted:
                continue
            if not matches_filter(self.metadata.get(vid, {}), where):
                continue
            filtered_results.append((vid, float(dist)))
            if len(filtered_results) >= k:
                break

        return filtered_results

    def delete(self, vector_id: str) -> bool:
        """Soft-deletes (tombstones) a vector from search results."""
        if vector_id not in self.vectors or vector_id in self.deleted:
            return False
        self.deleted.add(vector_id)
        return True

    def stats(self) -> dict[str, Any]:
        """Returns structural statistics of the HNSW index."""
        active_count = len(self.vectors) - len(self.deleted)
        layer_distribution: dict[int, int] = {}
        for lvl in self.levels.values():
            layer_distribution[lvl] = layer_distribution.get(lvl, 0) + 1

        total_edges = sum(
            sum(len(neighbors) for neighbors in node_layers.values())
            for node_layers in self.links.values()
        )

        return {
            "total_vectors": len(self.vectors),
            "active_vectors": active_count,
            "deleted_vectors": len(self.deleted),
            "max_level": self.max_level,
            "dimension": self.dimension,
            "metric": self.metric,
            "m": self.m,
            "m0": self.m0,
            "ef_construction": self.ef_construction,
            "layer_distribution": layer_distribution,
            "total_directed_edges": total_edges,
        }
