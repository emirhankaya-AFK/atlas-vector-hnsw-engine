"""Collection manager coordinating HNSW indexing, metadata, WAL persistence, and snapshots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.atlas_vector.index.hnsw import HNSWIndex
from src.atlas_vector.storage.snapshot import load_snapshot, save_snapshot
from src.atlas_vector.storage.wal import WriteAheadLog


class VectorCollection:
    """Manages an isolated vector collection with its HNSW index and WAL persistence."""

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        data_dir: str | Path = "data",
        m: int = 16,
        ef_construction: int = 100,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.index = HNSWIndex(
            dimension=dimension,
            metric=metric,
            m=m,
            ef_construction=ef_construction,
            seed=seed,
        )
        self.wal = WriteAheadLog(self.data_dir / f"{name}.wal")

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upserts a vector into the index and appends operation to the WAL."""
        self.index.upsert(vector_id, vector, metadata)
        self.wal.append({
            "op": "upsert",
            "id": vector_id,
            "vector": vector,
            "metadata": metadata or {},
        })

    def delete(self, vector_id: str) -> bool:
        """Tombstones a vector in the index and logs deletion to the WAL."""
        deleted = self.index.delete(vector_id)
        if deleted:
            self.wal.append({"op": "delete", "id": vector_id})
        return deleted

    def query(
        self,
        vector: list[float],
        k: int = 10,
        ef_search: int = 50,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Queries the HNSW index with approximate k-NN and optional metadata filtering."""
        results = self.index.search(vector, k=k, ef_search=ef_search, where=where)
        return [
            {
                "id": vid,
                "distance": dist,
                "metadata": self.index.metadata.get(vid, {}),
            }
            for vid, dist in results
        ]

    def snapshot(self) -> str:
        """Persists index state to snapshot and truncates WAL."""
        snapshot_path = self.data_dir / f"{self.name}.snapshot"
        save_snapshot(self.index, snapshot_path)
        self.wal.clear()
        return str(snapshot_path)

    def recover(self) -> None:
        """Reconstructs state by restoring latest snapshot and replaying subsequent WAL records."""
        snapshot_path = self.data_dir / f"{self.name}.snapshot"
        if snapshot_path.exists():
            self.index = load_snapshot(snapshot_path)

        for op in self.wal.replay():
            op_type = op.get("op")
            if op_type == "upsert":
                self.index.upsert(op["id"], op["vector"], op.get("metadata"))
            elif op_type == "delete":
                self.index.delete(op["id"])

    def stats(self) -> dict[str, Any]:
        """Returns collection operational and structural statistics."""
        base_stats = self.index.stats()
        base_stats.update({
            "collection_name": self.name,
            "wal_entries": self.wal.count(),
            "snapshot_exists": (self.data_dir / f"{self.name}.snapshot").exists(),
        })
        return base_stats


class AtlasEngine:
    """Multi-collection vector database engine with disk persistence coordination."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collections: dict[str, VectorCollection] = {}

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 100,
        seed: int = 42,
    ) -> VectorCollection:
        """Creates a new isolated vector collection."""
        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists")
        collection = VectorCollection(
            name=name,
            dimension=dimension,
            metric=metric,
            data_dir=self.data_dir,
            m=m,
            ef_construction=ef_construction,
            seed=seed,
        )
        self.collections[name] = collection
        return collection

    def get(self, name: str) -> VectorCollection:
        """Retrieves an existing collection by name."""
        if name not in self.collections:
            raise KeyError(f"Collection '{name}' not found")
        return self.collections[name]

    def list_collections(self) -> list[dict[str, Any]]:
        """Lists all registered collections with their dimensions and metrics."""
        return [
            {
                "name": col.name,
                "dimension": col.dimension,
                "metric": col.metric,
                "vectors": len(col.index.vectors) - len(col.index.deleted),
            }
            for col in self.collections.values()
        ]

    def delete_collection(self, name: str) -> bool:
        """Removes a collection and deletes its associated disk files."""
        if name not in self.collections:
            return False
        self.collections.pop(name)
        snapshot_file = self.data_dir / f"{name}.snapshot"
        wal_file = self.data_dir / f"{name}.wal"
        if snapshot_file.exists():
            snapshot_file.unlink()
        if wal_file.exists():
            wal_file.unlink()
        return True
