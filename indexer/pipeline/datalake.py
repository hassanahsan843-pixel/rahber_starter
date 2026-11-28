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


def record_batch(rec: DataLakeRecord, legacy_batch: str) -> str:
    """The batch a record belongs to; legacy records (no batch) belong to the primary corpus."""
    return rec.batch or legacy_batch


def in_batch(rec: DataLakeRecord, batch: str, legacy_batch: str) -> bool:
    return record_batch(rec, legacy_batch) == batch


def find_orphan_chunk_ids(
    registry: dict[str, DataLakeRecord],
    active_doc_ids: set[str],
    batch: str | None = None,
    legacy_batch: str = "",
) -> list[str]:
    """Return chunk IDs of documents that were removed from THIS batch's corpus.

    Only records owned by `batch` are considered: ingesting Batch B on its own must
    never delete Batch A's chunks just because Batch A's files are not in this run.
    `batch=None` keeps the old whole-registry behaviour (single-corpus callers).
    """
    orphan_ids: list[str] = []
    for rec in registry.values():
        if batch is not None and not in_batch(rec, batch, legacy_batch):
            continue
        if rec.doc_id not in active_doc_ids and not rec.archived:
            orphan_ids.extend(rec.chunk_ids)
    return orphan_ids


# ---------------------------------------------------------------------------
# Document snapshots: the data lake keeps the text of every indexed version, so
# archival can preserve the real previous content and verification can compare
# new documents against documents indexed by earlier runs or other batches.

def _snapshot_path(doc_id: str, doc_hash: str) -> Path:
    return Path(DATA_LAKE_DIR) / "documents" / doc_id / f"{doc_hash}.txt"


def save_snapshot(doc_id: str, doc_hash: str, text: str) -> None:
    path = _snapshot_path(doc_id, doc_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def _vector_path(doc_id: str, doc_hash: str, model: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
    return Path(DATA_LAKE_DIR) / "documents" / doc_id / f"{doc_hash}.{safe}.vec.json"


def save_doc_vector(doc_id: str, doc_hash: str, model: str, vector: list[float]) -> None:
    """Cache a whole-document embedding (near-duplicate check) for this exact content."""
    path = _vector_path(doc_id, doc_hash, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vector), encoding="utf-8")


def load_doc_vector(doc_id: str, doc_hash: str, model: str) -> list[float] | None:
    try:
        return list(json.loads(_vector_path(doc_id, doc_hash, model).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def load_snapshot(doc_id: str, doc_hash: str) -> str | None:
    path = _snapshot_path(doc_id, doc_hash)
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
