"""Unit and integration tests for Write-Ahead Log (WAL) and Snapshot recovery."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.atlas_vector.engine import AtlasEngine, VectorCollection
from src.atlas_vector.index.hnsw import HNSWIndex
from src.atlas_vector.storage.snapshot import load_snapshot, save_snapshot
from src.atlas_vector.storage.wal import WriteAheadLog


def test_wal_append_and_replay(tmp_path: Path) -> None:
    wal_file = tmp_path / "test.wal"
    wal = WriteAheadLog(wal_file)

    wal.append({"op": "upsert", "id": "v1", "vector": [1.0, 2.0]})
    wal.append({"op": "upsert", "id": "v2", "vector": [3.0, 4.0]})
    wal.append({"op": "delete", "id": "v1"})

    records = list(wal.replay())
    assert len(records) == 3
    assert records[0]["op"] == "upsert"
    assert records[0]["id"] == "v1"
    assert records[2]["op"] == "delete"
    assert wal.count() == 3


def test_wal_handles_corrupted_line(tmp_path: Path) -> None:
    wal_file = tmp_path / "corrupt.wal"
    wal = WriteAheadLog(wal_file)

    wal.append({"op": "upsert", "id": "v1", "vector": [1.0, 2.0]})
    # Simulate partial write / crash corruption
    with wal_file.open("a", encoding="utf-8") as f:
        f.write('{"incomplete_json": \n')
    wal.append({"op": "upsert", "id": "v2", "vector": [3.0, 4.0]})

    records = list(wal.replay())
    assert len(records) == 2
    assert records[0]["id"] == "v1"
    assert records[1]["id"] == "v2"


def test_wal_clear(tmp_path: Path) -> None:
    wal_file = tmp_path / "clear.wal"
    wal = WriteAheadLog(wal_file)
    wal.append({"op": "upsert", "id": "v1"})
    assert wal.count() == 1
    wal.clear()
    assert wal.count() == 0
    assert not wal_file.exists()


def test_snapshot_save_and_load(tmp_path: Path) -> None:
    snap_path = tmp_path / "index.snapshot"
    index = HNSWIndex(dimension=3, metric="cosine", m=12, seed=42)
    index.upsert("doc1", [1.0, 0.0, 0.0], {"title": "Doc 1"})
    index.upsert("doc2", [0.0, 1.0, 0.0], {"title": "Doc 2"})

    save_snapshot(index, snap_path)
    assert snap_path.exists()

    restored = load_snapshot(snap_path)
    assert restored.dimension == 3
    assert restored.metric == "cosine"
    assert len(restored.vectors) == 2
    assert restored.metadata["doc1"]["title"] == "Doc 1"

    # Search on restored index
    res = restored.search([0.9, 0.1, 0.0], k=1)
    assert res[0][0] == "doc1"


def test_snapshot_non_existent_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "missing.snapshot")


def test_crash_recovery_snapshot_plus_wal(tmp_path: Path) -> None:
    collection_dir = tmp_path / "engine_data"
    col = VectorCollection("financial", dimension=2, metric="l2", data_dir=collection_dir)

    # Step 1: Initial upserts
    col.upsert("stock_a", [1.0, 1.0], {"ticker": "AAPL"})
    col.upsert("stock_b", [2.0, 2.0], {"ticker": "MSFT"})

    # Step 2: Create periodic snapshot (this checkpoints and clears WAL)
    snap_file = col.snapshot()
    assert Path(snap_file).exists()

    # Step 3: Perform operations post-snapshot
    col.upsert("stock_c", [3.0, 3.0], {"ticker": "GOOG"})
    col.delete("stock_a")

    # Step 4: Simulate total process crash and restart fresh collection instance
    crashed_col = VectorCollection("financial", dimension=2, metric="l2", data_dir=collection_dir)
    crashed_col.recover()

    # Verify state after recovery
    stats = crashed_col.stats()
    assert stats["total_vectors"] == 3
    assert stats["active_vectors"] == 2
    assert stats["deleted_vectors"] == 1

    # Search on recovered instance: stock_a must NOT be returned, stock_c must be found
    results = crashed_col.query([2.9, 3.1], k=5)
    retrieved_ids = [r["id"] for r in results]
    assert "stock_c" in retrieved_ids
    assert "stock_b" in retrieved_ids
    assert "stock_a" not in retrieved_ids
    assert results[0]["id"] == "stock_c"


def test_multi_collection_engine(tmp_path: Path) -> None:
    engine = AtlasEngine(data_dir=tmp_path)
    engine.create_collection("products", dimension=4, metric="cosine")
    engine.create_collection("users", dimension=8, metric="l2")

    assert len(engine.collections) == 2
    assert engine.get("products").dimension == 4
    assert engine.get("users").dimension == 8

    # Duplicate collection name should raise ValueError
    with pytest.raises(ValueError, match="already exists"):
        engine.create_collection("products", dimension=4)

    # Unknown collection raises KeyError
    with pytest.raises(KeyError):
        engine.get("non_existent")

    # Delete collection
    assert engine.delete_collection("products") is True
    assert len(engine.collections) == 1
