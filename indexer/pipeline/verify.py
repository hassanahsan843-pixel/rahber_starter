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
import math
import os
import re
import time
from typing import Any

from .models import CheckResult, DataLakeRecord, Document, VerificationResult


# ---------------------------------------------------------------------------
# Similarity threshold for near-duplicate detection (configurable)
NEAR_DUPLICATE_THRESHOLD = float(os.environ.get("NEAR_DUPLICATE_THRESHOLD", "0.97"))
# Report-only band: at or above this (but below NEAR_DUPLICATE_THRESHOLD) a document
# passes but the report flags it as a possible near-duplicate. With nomic-embed-text the
# highest similarity between distinct, non-duplicate documents in this corpus is 0.84.
NEAR_DUPLICATE_REPORT_THRESHOLD = float(os.environ.get("NEAR_DUPLICATE_REPORT_THRESHOLD", "0.90"))

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

    # Truncation: the document's last line stops mid-sentence. Only for text-native
    # files: PDF/HTML extraction (and OCR) often ends on a table cell, list item or
    # footer without punctuation, which is not a truncated document.
    if doc.pages and doc.source.lower().endswith((".md", ".txt")):
        last_line = _last_content_line(_full_text(doc))
        if _ends_mid_sentence(last_line):
            results.append(CheckResult("completeness_truncation", False,
                                       f"Document ends mid-sentence: '...{last_line[-40:]}'"))
        else:
            results.append(CheckResult("completeness_truncation", True))
    elif doc.pages:
        results.append(CheckResult("completeness_truncation", True,
                                   "Skipped: end-of-text check applies to .md/.txt only"))

    return results


def _last_content_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _ends_mid_sentence(line: str) -> bool:
    """True when prose stops without closing punctuation ("...in writing with supporting").

    List items, table rows, headings and "Key: value" lines (e.g. "Dinner: 7:30 PM")
    legitimately end without a full stop, so they are not treated as truncation.
    """
    if not line or line[0] in "-*+|#>" or re.match(r"^\d+[.)]\s", line) or ":" in line:
        return False
    last = line[-1]
    return last.islower() or last in ",;(-–"


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


_YEAR_RE = re.compile(r"[-_ ]?(?:19|20)\d{2}(?:[-_]\d{2})?")


def family_key(source: str) -> str:
    """Versions of one policy share a family: academic-rules-2026.md -> academic-rules."""
    stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    return _YEAR_RE.sub("", stem).strip("-_ ")


def check_version_family(doc: Document, registry: dict[str, DataLakeRecord]) -> list[CheckResult]:
    """Recency across file names: academic-rules-2027.md is a newer version of
    academic-rules-2026.md even though the two files have different doc_ids."""
    fam = family_key(doc.source)
    if not fam or doc.doc_version in ("", "unknown"):
        return []
    relatives = [r for r in registry.values()
                 if r.doc_id != doc.doc_id and not r.archived
                 and r.doc_version not in ("", "unknown") and family_key(r.source) == fam]
    if not relatives:
        return []
    newest = max(relatives, key=lambda r: r.doc_version)
    if doc.doc_version > newest.doc_version:
        return [CheckResult("recency_family", True,
                            f"Newer version {doc.doc_version} of '{newest.source}' "
                            f"({newest.doc_version})")]
    if doc.doc_version < newest.doc_version and not doc.superseded:
        # Reported, not quarantined: file names cannot tell a replaced version
        # (academic-rules-2023 -> 2026) from a dated series (annual-report-2019-20,
        # annual-report-2023-24) whose older members are still valid documents.
        return [CheckResult("recency_family", True,
                            f"WARNING possible stale version {doc.doc_version}: '{newest.source}' "
                            f"({newest.doc_version}) is newer; mark superseded if it replaces this one")]
    return [CheckResult("recency_family", True,
                        "Historical version (marked superseded)" if doc.superseded else
                        f"Same version as '{newest.source}'")]


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def check_uniqueness(
    doc: Document,
    registry: dict[str, DataLakeRecord],
    all_docs: list[Document],
    near_dup_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    reference_texts: dict[str, tuple[str, str]] | None = None,
    doc_vectors: dict[str, list[float]] | None = None,
    report_threshold: float = NEAR_DUPLICATE_REPORT_THRESHOLD,
) -> list[CheckResult]:
    """Detect exact duplicates (hash) and near-duplicates (embedding cosine).

    Compared against: documents already in the data lake (other runs and other
    batches, via `reference_texts` = {doc_id: (source, text)} snapshots), and
    documents EARLIER in this run's `all_docs`. Comparing only with earlier batch
    members keeps the first copy and quarantines the later one, instead of
    quarantining both halves of a duplicate pair.

    Near-duplicates use `doc_vectors` ({doc_id: embedding of the whole document},
    from the pinned embedding model) when given, else a bag-of-words vector.
    Versions of the same document family (academic-rules-2023 / -2026 / -2027)
    are skipped: they are expected to be similar and the recency checks own them.
    """
    results: list[CheckResult] = []
    references = reference_texts or {}
    full = _full_text(doc)
    norm = _normalized(full)

    earlier: list[Document] = []
    for other in all_docs:
        if other is doc or other.doc_id == doc.doc_id:
            break
        earlier.append(other)

    # Exact duplicate: same content (raw hash, or same text ignoring case/whitespace)
    exact_of = ""
    for rec in registry.values():
        if rec.doc_hash == doc.doc_hash and rec.doc_id != doc.doc_id:
            exact_of = rec.source
            break
    if not exact_of:
        for ref_id, (ref_source, ref_text) in references.items():
            if ref_id != doc.doc_id and norm and _normalized(ref_text) == norm:
                exact_of = ref_source
                break
    if not exact_of:
        for other in earlier:
            if norm and _normalized(_full_text(other)) == norm:
                exact_of = other.source
                break
    if exact_of:
        results.append(CheckResult("uniqueness_exact_dup", False,
                                   f"Exact duplicate of '{exact_of}' (same content)"))
        return results  # no need to continue

    results.append(CheckResult("uniqueness_exact_dup", True))

    if not full.strip():
        results.append(CheckResult("uniqueness_near_dup", True, "Skipped (empty document)"))
        return results

    vectors = doc_vectors or {}

    def _vec(doc_id: str, text: str) -> list[float]:
        return vectors.get(doc_id) or _simple_bag_of_words_vector(text)

    method = "embedding" if doc.doc_id in vectors else "bag-of-words"
    doc_vec = _vec(doc.doc_id, full)
    family = family_key(doc.source)
    candidates = [(rid, src, txt) for rid, (src, txt) in references.items() if rid != doc.doc_id]
    candidates += [(o.doc_id, o.source, _full_text(o)) for o in earlier]
    near_dups: list[str] = []
    possible: list[str] = []
    for other_id, other_source, other_full in candidates:
        if not other_full.strip() or (family and family_key(other_source) == family):
            continue
        sim = _cosine_similarity(doc_vec, _vec(other_id, other_full))
        if sim >= near_dup_threshold:
            near_dups.append(f"{other_source} (sim={sim:.3f})")
        elif sim >= report_threshold:
            possible.append(f"{other_source} (sim={sim:.3f})")

    if near_dups:
        results.append(CheckResult("uniqueness_near_dup", False,
                                   f"Near-duplicate of: {', '.join(near_dups)} [{method}]"))
    elif possible:
        results.append(CheckResult(
            "uniqueness_near_dup", True,
            f"WARNING possible near-duplicate of: {', '.join(possible)} [{method}; "
            f"reported at >= {report_threshold}, quarantined at >= {near_dup_threshold}]"))
    else:
        results.append(CheckResult("uniqueness_near_dup", True))

    return results


# ---------------------------------------------------------------------------
# Top-level verification

def verify(
    doc: Document,
    registry: dict[str, DataLakeRecord],
    all_docs: list[Document],
    reference_texts: dict[str, tuple[str, str]] | None = None,
    doc_vectors: dict[str, list[float]] | None = None,
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
    result.checks.extend(check_version_family(doc, registry))

    # Uniqueness
    result.checks.extend(check_uniqueness(doc, registry, all_docs,
                                          reference_texts=reference_texts,
                                          doc_vectors=doc_vectors))

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
