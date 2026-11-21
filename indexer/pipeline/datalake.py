"""A03 Part A2 – Data Lake (persistent document registry).

The data lake is a JSON file stored at DATA_LAKE_DIR/registry.json.
It records every document ever ingested: its hash, version, chunk IDs,
and superseded / archived status.

This is the source of truth for incremental updates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import DataLakeRecord

DATA_LAKE_DIR = os.environ.get("DATA_LAKE_DIR", "/data_lake")
REGISTRY_FILE = "registry.json"


def _registry_path() -> Path:
    p = Path(DATA_LAKE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p / REGISTRY_FILE


def load_registry() -> dict[str, DataLakeRecord]:
    """Load the registry from disk. Returns empty dict on first run."""
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return {r["doc_id"]: DataLakeRecord.from_dict(r) for r in raw}
    except Exception:
        return {}


def save_registry(registry: dict[str, DataLakeRecord]) -> None:
    """Persist the registry to disk atomically."""
    path = _registry_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps([r.to_dict() for r in registry.values()], indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def upsert_record(registry: dict[str, DataLakeRecord], record: DataLakeRecord) -> None:
    """Add or replace a registry record (in-memory only – call save_registry after)."""
    registry[record.doc_id] = record


def mark_superseded(registry: dict[str, DataLakeRecord], doc_id: str) -> None:
    """Mark an existing record as superseded."""
    if doc_id in registry:
        registry[doc_id].superseded = True


def archive_record(
    registry: dict[str, DataLakeRecord],
    doc_id: str,
    archive_path: str,
) -> None:
    """Mark a record as archived and record where it was stored."""
    if doc_id in registry:
        registry[doc_id].archived = True
        registry[doc_id].archive_path = archive_path


def find_orphan_chunk_ids(
    registry: dict[str, DataLakeRecord],
    active_doc_ids: set[str],
) -> list[str]:
    """Return chunk IDs from documents that are no longer in the active corpus."""
    orphan_ids: list[str] = []
    for rec in registry.values():
        if rec.doc_id not in active_doc_ids and not rec.archived:
            orphan_ids.extend(rec.chunk_ids)
    return orphan_ids
