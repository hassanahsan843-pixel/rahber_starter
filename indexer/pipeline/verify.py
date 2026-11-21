"""A03 Part A1 – Document Verification Stage.

Checks every extracted Document for:
  1. Quality   – empty text, garbled content, extraction failures
  2. Completeness – required metadata (source, title, date/version, page_count)
  3. Recency   – detect newer version of an existing document via data lake
  4. Uniqueness – exact duplicates (hash), near-duplicates (embedding cosine similarity)

Documents that fail must be quarantined before chunking.
Generates a per-run verification report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from .models import CheckResult, DataLakeRecord, Document, VerificationResult


# ---------------------------------------------------------------------------
# Similarity threshold for near-duplicate detection (configurable)
NEAR_DUPLICATE_THRESHOLD = float(os.environ.get("NEAR_DUPLICATE_THRESHOLD", "0.97"))

# Minimum non-whitespace characters for a document to be considered non-empty
MIN_CONTENT_CHARS = int(os.environ.get("MIN_CONTENT_CHARS", "50"))


# ---------------------------------------------------------------------------
# Helpers

def _simple_bag_of_words_vector(text: str, dim: int = 512) -> list[float]:
    """Fast offline embedding for near-duplicate detection (no model required)."""
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (n1 * n2)))


def _full_text(doc: Document) -> str:
    return "\n".join(p.text for p in doc.pages)


def _is_garbled(text: str) -> bool:
    """Heuristic: garbled text has very low ratio of alphabetic characters."""
    if not text.strip():
        return False  # empty is caught by the empty check
    alpha = sum(1 for c in text if c.isalpha())
    total = len(text.replace(" ", ""))
    if total == 0:
        return True
    ratio = alpha / total
    return ratio < 0.3


# ---------------------------------------------------------------------------
# Individual checks

def check_quality(doc: Document) -> list[CheckResult]:
    results: list[CheckResult] = []
    full = _full_text(doc)

    # Empty document
    stripped = full.strip()
    if len(stripped) < MIN_CONTENT_CHARS:
        results.append(CheckResult("quality_empty", False,
                                   f"Document text is too short ({len(stripped)} chars < {MIN_CONTENT_CHARS})"))
    else:
        results.append(CheckResult("quality_empty", True))

    # Garbled text
    if _is_garbled(full):
        results.append(CheckResult("quality_garbled", False,
                                   "Text has low alphabetic ratio – likely garbled or binary content"))
    else:
        results.append(CheckResult("quality_garbled", True))

    # Extraction failure (pages with zero content)
    empty_pages = [p.page for p in doc.pages if not p.text.strip()]
    if len(empty_pages) == len(doc.pages) and doc.pages:
        results.append(CheckResult("quality_extraction", False,
                                   "All pages have empty text – extraction may have failed"))
    else:
        results.append(CheckResult("quality_extraction", True))

    return results


def check_completeness(doc: Document) -> list[CheckResult]:
    results: list[CheckResult] = []

    # Source
    if doc.source:
        results.append(CheckResult("completeness_source", True))
    else:
        results.append(CheckResult("completeness_source", False, "Missing source filename"))

    # Title
    if doc.title:
        results.append(CheckResult("completeness_title", True))
    else:
        results.append(CheckResult("completeness_title", False, "Title could not be extracted"))

    # Date / version
    if doc.date or (doc.doc_version and doc.doc_version != "unknown"):
        results.append(CheckResult("completeness_date", True))
    else:
        results.append(CheckResult("completeness_date", False,
                                   "No date or version found (set 'date:' or 'doc_version:' in front matter)"))

    # Page count
    if doc.page_count > 0:
        results.append(CheckResult("completeness_pages", True))
    else:
        results.append(CheckResult("completeness_pages", False, "Zero pages extracted"))

    # Truncation heuristic: last page ends abruptly (< 20 chars and no terminal punctuation)
    if doc.pages:
        last_text = doc.pages[-1].text.strip()
        if last_text and len(last_text) < 20 and not last_text[-1] in ".?!\n":
            results.append(CheckResult("completeness_truncation", False,
                                       f"Last page ends abruptly: '{last_text[:40]}'"))
        else:
            results.append(CheckResult("completeness_truncation", True))

    return results


def check_recency(doc: Document, registry: dict[str, DataLakeRecord]) -> list[CheckResult]:
    """Check whether a newer version of this document already exists in the data lake."""
    results: list[CheckResult] = []
    existing = registry.get(doc.doc_id)
    if existing is None:
        results.append(CheckResult("recency_new", True, "First ingestion of this document"))
        return results

    # Compare versions lexicographically (works for ISO-date strings like "2026-06")
    existing_ver = existing.doc_version
    incoming_ver = doc.doc_version
    if incoming_ver > existing_ver:
        results.append(CheckResult("recency_newer", True,
                                   f"Incoming version {incoming_ver} supersedes {existing_ver}"))
    elif incoming_ver == existing_ver:
        results.append(CheckResult("recency_same", True,
                                   f"Same version {incoming_ver} as existing – will check hash"))
    else:
        # Incoming is OLDER than what is already indexed
        results.append(CheckResult("recency_older", False,
                                   f"Incoming version {incoming_ver} is older than indexed {existing_ver} – quarantining"))
    return results


def check_uniqueness(
    doc: Document,
    registry: dict[str, DataLakeRecord],
    all_docs: list[Document],
    near_dup_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[CheckResult]:
    """Detect exact duplicates (hash) and near-duplicates (embedding similarity)."""
    results: list[CheckResult] = []

    # Exact duplicate: same hash as already-registered document
    for rec in registry.values():
        if rec.doc_hash == doc.doc_hash and rec.doc_id != doc.doc_id:
            results.append(CheckResult("uniqueness_exact_dup", False,
                                       f"Exact duplicate of '{rec.source}' (same content hash)"))
            return results  # no need to continue

    results.append(CheckResult("uniqueness_exact_dup", True))

    # Near-duplicate: cosine similarity against other docs in this batch
    full = _full_text(doc)
    if not full.strip():
        results.append(CheckResult("uniqueness_near_dup", True, "Skipped (empty document)"))
        return results

    doc_vec = _simple_bag_of_words_vector(full)
    near_dups: list[str] = []
    for other in all_docs:
        if other.doc_id == doc.doc_id:
            continue
        other_full = _full_text(other)
        if not other_full.strip():
            continue
        other_vec = _simple_bag_of_words_vector(other_full)
        sim = _cosine_similarity(doc_vec, other_vec)
        if sim >= near_dup_threshold:
            near_dups.append(f"{other.source} (sim={sim:.3f})")

    if near_dups:
        results.append(CheckResult("uniqueness_near_dup", False,
                                   f"Near-duplicate of: {', '.join(near_dups)}"))
    else:
        results.append(CheckResult("uniqueness_near_dup", True))

    return results


# ---------------------------------------------------------------------------
# Top-level verification

def verify(
    doc: Document,
    registry: dict[str, DataLakeRecord],
    all_docs: list[Document],
) -> VerificationResult:
    """Run all four check categories and return a VerificationResult."""
    result = VerificationResult(
        source=doc.source,
        doc_id=doc.doc_id,
        doc_hash=doc.doc_hash,
    )

    # Quality
    result.checks.extend(check_quality(doc))

    # Completeness
    result.checks.extend(check_completeness(doc))

    # Recency
    result.checks.extend(check_recency(doc, registry))

    # Uniqueness
    result.checks.extend(check_uniqueness(doc, registry, all_docs))

    # Decide quarantine: fail any non-"older version" check → quarantine
    failures = [c for c in result.checks if not c.passed]
    if failures:
        # Recency_older is a definite quarantine; quality/completeness failures are too
        result.quarantined = True
        result.quarantine_reason = "; ".join(c.reason for c in failures)

    return result


def generate_report(results: list[VerificationResult], run_id: str) -> dict[str, Any]:
    """Generate a structured verification report for the ingestion run."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    quarantined = sum(1 for r in results if r.quarantined)

    return {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total": total,
            "passed": passed,
            "quarantined": quarantined,
        },
        "documents": [r.to_dict() for r in results],
    }
