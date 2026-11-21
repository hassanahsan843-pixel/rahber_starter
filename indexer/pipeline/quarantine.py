"""A03 Part A1 – Quarantine management.

Documents that fail verification are moved to quarantine so they are
never chunked, embedded or indexed. The quarantine directory holds the
original file plus a JSON sidecar with the failure reasons.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

QUARANTINE_DIR = os.environ.get("QUARANTINE_DIR", "/quarantine")


def _quarantine_dir() -> Path:
    p = Path(QUARANTINE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def quarantine_document(source_path: str, reasons: list[str], metadata: dict[str, Any]) -> str:
    """Move a document to quarantine and write a sidecar JSON with failure details.
    
    Returns the destination path.
    """
    src = Path(source_path)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest_name = f"{src.stem}_{stamp}{src.suffix}"
    qdir = _quarantine_dir()
    dest = qdir / dest_name

    # Copy the file (don't move – the caller may still need the original)
    if src.exists():
        shutil.copy2(src, dest)

    # Write sidecar
    sidecar = qdir / f"{dest_name}.quarantine.json"
    sidecar.write_text(
        json.dumps({
            "timestamp": stamp,
            "source": str(source_path),
            "reasons": reasons,
            "metadata": metadata,
        }, indent=2),
        encoding="utf-8",
    )
    return str(dest)
