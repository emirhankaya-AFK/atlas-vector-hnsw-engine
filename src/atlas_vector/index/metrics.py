"""Vector distance functions used by exact and approximate indexes."""
from __future__ import annotations

import numpy as np


def pairwise_l2(u: np.ndarray, v: np.ndarray) -> float:
    """Calculates Euclidean (L2) distance between two 1D vectors."""
    diff = u - v
    return float(np.sqrt(np.dot(diff, diff)))


def pairwise_cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Calculates Cosine distance (1.0 - cosine_similarity) between two 1D vectors."""
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    denom = max(norm_u * norm_v, 1e-12)
    similarity = float(np.dot(u, v)) / denom
    # Clamp to [-1.0, 1.0] to prevent floating point imprecision giving < 0 or > 2
    similarity = max(-1.0, min(1.0, similarity))
    return float(1.0 - similarity)


def l2_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculates Euclidean (L2) distance between a 1D query and a 2D matrix of vectors."""
    diff = vectors - query
    return np.sqrt(np.sum(diff * diff, axis=1))


def cosine_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculates Cosine distance between a 1D query and a 2D matrix of vectors."""
    q_norm = float(np.linalg.norm(query))
    v_norms = np.linalg.norm(vectors, axis=1)
    denom = np.maximum(q_norm * v_norms, 1e-12)
    dots = vectors @ query
    sims = np.clip(dots / denom, -1.0, 1.0)
    return 1.0 - sims
