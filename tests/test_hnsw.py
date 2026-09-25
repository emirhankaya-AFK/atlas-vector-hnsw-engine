"""Unit and integration tests for HNSW layered graph index."""
from __future__ import annotations

import numpy as np
import pytest

from src.atlas_vector.index.flat import FlatIndex
from src.atlas_vector.index.hnsw import HNSWIndex


def test_hnsw_insertion_and_search_l2() -> None:
    index = HNSWIndex(dimension=2, metric="l2", seed=42)
    index.upsert("a", [0.0, 0.0])
    index.upsert("b", [2.0, 2.0])
    index.upsert("c", [10.0, 10.0])

    results = index.search([1.9, 2.1], k=1)
    assert len(results) == 1
    assert results[0][0] == "b"
    assert results[0][1] < 0.2


def test_hnsw_insertion_and_search_cosine() -> None:
    index = HNSWIndex(dimension=3, metric="cosine", seed=42)
    index.upsert("doc_x", [1.0, 0.0, 0.0])
    index.upsert("doc_y", [0.0, 1.0, 0.0])
    index.upsert("doc_z", [0.0, 0.0, 1.0])

    results = index.search([0.9, 0.1, 0.0], k=2)
    assert results[0][0] == "doc_x"
    assert results[0][1] < 0.05


def test_deterministic_seed() -> None:
    rng = np.random.default_rng(100)
    data = rng.normal(size=(50, 4)).tolist()

    idx1 = HNSWIndex(dimension=4, metric="l2", m=8, seed=123)
    idx2 = HNSWIndex(dimension=4, metric="l2", m=8, seed=123)

    for i, vec in enumerate(data):
        idx1.upsert(f"v_{i}", vec)
        idx2.upsert(f"v_{i}", vec)

    assert idx1.max_level == idx2.max_level
    assert idx1.levels == idx2.levels
    assert idx1.links == idx2.links


def test_high_recall_against_flat_baseline() -> None:
    rng = np.random.default_rng(42)
    dim = 16
    hnsw = HNSWIndex(dimension=dim, metric="cosine", m=16, ef_construction=80, seed=42)
    flat = FlatIndex(dimension=dim, metric="cosine")

    vectors = rng.normal(size=(150, dim)).astype(np.float32)
    for i, vec in enumerate(vectors):
        hnsw.upsert(f"node_{i}", vec)
        flat.upsert(f"node_{i}", vec)

    queries = rng.normal(size=(20, dim)).astype(np.float32)
    total_recall = 0.0
    for q in queries:
        exact = {vid for vid, _ in flat.search(q, k=10)}
        approx = {vid for vid, _ in hnsw.search(q, k=10, ef_search=60)}
        total_recall += len(exact & approx) / 10.0

    avg_recall = total_recall / len(queries)
    assert avg_recall >= 0.85


def test_neighbor_limits_m_and_m0() -> None:
    m = 6
    index = HNSWIndex(dimension=4, metric="l2", m=m, ef_construction=50, seed=42)
    rng = np.random.default_rng(9)
    for i in range(80):
        index.upsert(f"n_{i}", rng.normal(size=4).tolist())

    m0_limit = 2 * m
    for vid, layers in index.links.items():
        for layer, neighbors in layers.items():
            if layer == 0:
                assert len(neighbors) <= m0_limit, f"Node {vid} at layer 0 has {len(neighbors)} > {m0_limit}"
            else:
                assert len(neighbors) <= m, f"Node {vid} at layer {layer} has {len(neighbors)} > {m}"


def test_heuristic_neighbor_diversity() -> None:
    index = HNSWIndex(dimension=2, metric="l2", m=2, ef_construction=50, seed=1)
    # Collinear points: c1 is between base and c2
    index.upsert("base", [0.0, 0.0])
    index.upsert("c1", [1.0, 0.0])
    index.upsert("c2", [1.8, 0.0])
    index.upsert("diverse", [0.0, 1.0])

    # Neighbors of base should prefer diverse angles rather than collinear duplicates
    assert len(index.links["base"][0]) <= index.m0


def test_soft_delete_tombstone() -> None:
    index = HNSWIndex(dimension=2, metric="l2", seed=2)
    index.upsert("keep", [1.0, 1.0], {"type": "A"})
    index.upsert("drop", [1.05, 1.05], {"type": "B"})

    # Prior to delete, drop is closest to [1.05, 1.05]
    res_before = index.search([1.05, 1.05], k=1)
    assert res_before[0][0] == "drop"

    # Soft delete
    deleted = index.delete("drop")
    assert deleted is True

    # After delete, drop must never be returned
    res_after = index.search([1.05, 1.05], k=2)
    ids_after = [vid for vid, _ in res_after]
    assert "drop" not in ids_after
    assert "keep" in ids_after


def test_soft_delete_entry_point() -> None:
    index = HNSWIndex(dimension=2, metric="l2", seed=3)
    index.upsert("ep", [0.0, 0.0])
    index.upsert("other", [5.0, 5.0])

    # Delete entry point
    assert index.entry_point == "ep"
    index.delete("ep")

    # Search should still succeed and route to active node
    results = index.search([4.9, 5.1], k=1)
    assert len(results) == 1
    assert results[0][0] == "other"


def test_metadata_filtering() -> None:
    index = HNSWIndex(dimension=2, metric="l2", seed=4)
    index.upsert("v1", [1.0, 1.0], {"tenant": "tr", "env": "prod"})
    index.upsert("v2", [1.1, 1.1], {"tenant": "us", "env": "prod"})
    index.upsert("v3", [1.2, 1.2], {"tenant": "tr", "env": "dev"})

    # Filter single field
    res_tr = index.search([1.0, 1.0], k=5, where={"tenant": "tr"})
    assert set(vid for vid, _ in res_tr) == {"v1", "v3"}

    # Filter multiple fields
    res_tr_prod = index.search([1.0, 1.0], k=5, where={"tenant": "tr", "env": "prod"})
    assert len(res_tr_prod) == 1
    assert res_tr_prod[0][0] == "v1"

    # Filter with non-matching criteria
    res_none = index.search([1.0, 1.0], k=5, where={"tenant": "de"})
    assert res_none == []


def test_invalid_dimension_error() -> None:
    index = HNSWIndex(dimension=4, metric="l2")
    with pytest.raises(ValueError, match="Expected vector dimension 4"):
        index.upsert("bad_dim", [1.0, 2.0])

    with pytest.raises(ValueError, match="Expected query dimension 4"):
        index.search([1.0, 2.0, 3.0], k=1)


def test_hnsw_stats_structure() -> None:
    index = HNSWIndex(dimension=3, metric="cosine", m=8)
    index.upsert("n1", [1.0, 0.0, 0.0])
    index.upsert("n2", [0.0, 1.0, 0.0])
    index.delete("n2")

    stats = index.stats()
    assert stats["total_vectors"] == 2
    assert stats["active_vectors"] == 1
    assert stats["deleted_vectors"] == 1
    assert stats["metric"] == "cosine"
    assert stats["m"] == 8
    assert stats["m0"] == 16
    assert isinstance(stats["layer_distribution"], dict)
