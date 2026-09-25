"""Unit tests for distance metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.atlas_vector.index.metrics import (
    cosine_distance,
    l2_distance,
    pairwise_cosine,
    pairwise_l2,
)


def test_pairwise_l2() -> None:
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = np.array([4.0, 6.0, 3.0], dtype=np.float32)
    # sqrt((4-1)^2 + (6-2)^2 + (3-3)^2) = sqrt(9 + 16) = 5.0
    assert pytest.approx(pairwise_l2(u, v), rel=1e-5) == 5.0
    assert pytest.approx(pairwise_l2(u, u), abs=1e-6) == 0.0


def test_pairwise_cosine() -> None:
    # Identical vectors -> similarity 1.0 -> distance 0.0
    u = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert pytest.approx(pairwise_cosine(u, u), abs=1e-6) == 0.0

    # Orthogonal vectors -> similarity 0.0 -> distance 1.0
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    assert pytest.approx(pairwise_cosine(v1, v2), abs=1e-6) == 1.0

    # Opposite vectors -> similarity -1.0 -> distance 2.0
    v3 = np.array([-1.0, 0.0], dtype=np.float32)
    assert pytest.approx(pairwise_cosine(v1, v3), abs=1e-6) == 2.0


def test_vectorized_l2_matches_pairwise() -> None:
    q = np.array([1.0, 1.0], dtype=np.float32)
    matrix = np.array([[1.0, 1.0], [4.0, 5.0], [1.0, 4.0]], dtype=np.float32)
    distances = l2_distance(q, matrix)
    assert len(distances) == 3
    assert pytest.approx(distances[0], abs=1e-6) == 0.0
    assert pytest.approx(distances[1], rel=1e-5) == 5.0
    assert pytest.approx(distances[2], rel=1e-5) == 3.0


def test_vectorized_cosine_matches_pairwise() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]], dtype=np.float32)
    distances = cosine_distance(q, matrix)
    assert len(distances) == 3
    assert pytest.approx(distances[0], abs=1e-6) == 0.0
    assert pytest.approx(distances[1], abs=1e-6) == 1.0
    assert pytest.approx(distances[2], abs=1e-6) == 2.0
