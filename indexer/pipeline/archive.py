"""A03 Part A2 – Document archival.

Copies older document versions to the archive directory before replacing them.
Archived documents can be restored.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .models import DataLakeRecord

ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "/archive")


def _archive_dir() -> Path:
    p = Path(ARCHIVE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def archive_document(record: DataLakeRecord, source_text: str) -> str:
    """Write an archived copy of the document text and its metadata.
    
    Returns the archive path so it can be recorded in the registry.
    """
    adir = _archive_dir()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{record.doc_id}_{record.doc_version}_{stamp}"

    # Save the metadata
    meta_path = adir / f"{name}.meta.json"
    meta_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

    # Save the document text
    text_path = adir / f"{name}.txt"
    text_path.write_text(source_text, encoding="utf-8")

    return str(text_path)


def restore_document(archive_path: str, restore_to: str) -> None:
    """Copy an archived document back to the given path."""
    src = Path(archive_path)
    if not src.exists():
        raise FileNotFoundError(f"Archive file not found: {archive_path}")
    shutil.copy2(src, restore_to)
