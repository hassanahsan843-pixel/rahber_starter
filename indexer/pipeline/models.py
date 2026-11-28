"""The shapes passed between stages. Metadata fields are required (section 3.1).

Extended for A03 RAGOps:
  - Document now carries doc_id, doc_hash, title, date, page_count
  - VerificationResult records pass/fail per check
  - DataLakeRecord is the persistent registry entry
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    """Text of one page, as extraction produced it."""

    source: str              # file name, e.g. Academic_Rules_Handbook_2026.pdf
    page: int                # 1-based
    text: str
    ocr: bool = False        # True when this page had no text layer and was OCR'd


@dataclass
class Document:
    source: str
    doc_version: str         # e.g. "2026-06"; from corpus/sources.yaml or front matter
    superseded: bool         # True for an older version of a policy
    pages: list[Page] = field(default_factory=list)

    # A03 additions
    doc_id: str = ""         # stable ID: sha256 of source filename (not content)
    doc_hash: str = ""       # sha256 of full text content – changes when doc changes
    title: str = ""          # from front matter or first heading
    date: str = ""           # from front matter "date:" field
    page_count: int = 0      # total pages extracted

    def compute_hash(self) -> str:
        """Compute content hash from all page texts."""
        combined = "\n".join(p.text for p in self.pages)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def compute_id(self) -> str:
        """Stable ID based on source filename only."""
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "doc_hash": self.doc_hash,
            "doc_version": self.doc_version,
            "title": self.title,
            "date": self.date,
            "page_count": self.page_count,
            "superseded": self.superseded,
        }

    def populate_derived(self) -> None:
        """Fill doc_id, doc_hash and page_count after pages are set."""
        self.doc_id = self.compute_id()
        self.doc_hash = self.compute_hash()
        self.page_count = len(self.pages)
        # Try to extract title from first non-empty line of first page
        if not self.title and self.pages:
            for line in self.pages[0].text.splitlines():
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    self.title = stripped[:120]
                    break


@dataclass
class Chunk:
    id: str                  # stable: same document + same position gives the same id
    text: str
    source: str
    section: str             # the heading this chunk sits under, e.g. "5.4 Academic Warning"
    page: int
    doc_version: str
    superseded: bool
    doc_id: str = ""         # A03: links back to Document.doc_id

    def metadata(self) -> dict[str, str | int | bool]:
        """Chroma metadata values must be str, int, float or bool."""
        m: dict[str, str | int | bool] = {
            "source": self.source,
            "section": self.section,
            "page": self.page,
            "doc_version": self.doc_version,
            "superseded": self.superseded,
        }
        if self.doc_id:
            m["doc_id"] = self.doc_id
        return m


# ---------------------------------------------------------------------------
# A03: Verification

@dataclass
class CheckResult:
    name: str
    passed: bool
    reason: str = ""


@dataclass
class VerificationResult:
    """Collects the outcome of every check for one document."""
    source: str
    doc_id: str
    doc_hash: str
    checks: list[CheckResult] = field(default_factory=list)
    quarantined: bool = False
    quarantine_reason: str = ""

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and not self.quarantined

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "doc_hash": self.doc_hash,
            "passed": self.passed,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "checks": [
                {"name": c.name, "passed": c.passed, "reason": c.reason}
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# A03: Data lake registry entry

@dataclass
class DataLakeRecord:
    """One row in the data lake registry – persisted as JSON."""
    doc_id: str
    source: str
    title: str
    doc_version: str
    doc_hash: str
    date: str
    page_count: int
    chunk_ids: list[str]
    superseded: bool
    archived: bool = False
    archive_path: str = ""
    # Ingestion batch that owns this record (e.g. "corpus", "batch_b"). A run only
    # treats records of its own batch as removable; "" = legacy (primary corpus).
    batch: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "doc_version": self.doc_version,
            "doc_hash": self.doc_hash,
            "date": self.date,
            "page_count": self.page_count,
            "chunk_ids": self.chunk_ids,
            "superseded": self.superseded,
            "archived": self.archived,
            "archive_path": self.archive_path,
            "batch": self.batch,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DataLakeRecord":
        return DataLakeRecord(
            doc_id=d["doc_id"],
            source=d["source"],
            title=d.get("title", ""),
            doc_version=d.get("doc_version", "unknown"),
            doc_hash=d.get("doc_hash", ""),
            date=d.get("date", ""),
            page_count=d.get("page_count", 0),
            chunk_ids=d.get("chunk_ids", []),
            superseded=d.get("superseded", False),
            archived=d.get("archived", False),
            archive_path=d.get("archive_path", ""),
            batch=d.get("batch", ""),
        )
