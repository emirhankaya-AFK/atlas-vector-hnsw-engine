"""Binary snapshot serialization and deserialization for HNSW index state with fsync durability."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

from src.atlas_vector.index.hnsw import HNSWIndex


def save_snapshot(index: HNSWIndex, path: str | Path) -> None:
    """Serializes complete HNSW index state to a binary snapshot file with fsync and atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(".tmp")
    with temp_target.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temp_target.replace(target)


def load_snapshot(path: str | Path) -> HNSWIndex:
    """Deserializes an HNSWIndex from a binary snapshot file."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Snapshot file not found: {target}")
    with target.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, HNSWIndex):
        raise TypeError(f"Loaded object is not an HNSWIndex: {type(loaded)}")
    return loaded
