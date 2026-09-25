"""Append-only JSONL write-ahead log for durable crash recovery with strict fsync."""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WriteAheadLog:
    """Manages append-only JSONL write-ahead logging with replay capabilities and physical fsync."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, operation: dict[str, Any]) -> None:
        """Appends a single JSON operation record and flushes + fsyncs to physical disk."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(operation, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def replay(self) -> Iterable[dict[str, Any]]:
        """Replays all logged operations in chronological order, ignoring corrupted records."""
        if not self.path.exists():
            return []
        operations = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    operations.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping corrupted WAL line %d in %s: %s", line_idx, self.path, exc)
        return operations

    def clear(self) -> None:
        """Truncates the write-ahead log (used during snapshot compaction)."""
        if self.path.exists():
            self.path.unlink()

    def count(self) -> int:
        """Returns the number of non-empty entries in the WAL."""
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
