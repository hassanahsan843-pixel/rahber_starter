"""Tests for Person 2 indexing pipeline stages: extract, clean, chunk, embed, and store."""

from __future__ import annotations

import math
from pipeline.chunk import chunk
from pipeline.clean import clean
from pipeline.config import get_settings
from pipeline.embed import _deterministic_stub_vector, embed
from pipeline.models import Document, Page


def test_clean_preserves_section_and_removes_headers() -> None:
    doc = Document(
        source="test.pdf",
        doc_version="2026-06",
        superseded=False,
        pages=[
            Page(
                source="test.pdf",
                page=1,
                text="Page 1 of 53\nFAST-NUCES Academic Rules\n### 1.21 Attendance Policy\nAll students must maintain 80% attendance.\nPage 1 of 53",
            )
        ],
    )
    cleaned = clean(doc)
    text = cleaned.pages[0].text
    assert "Page 1 of 53" not in text
    assert "### 1.21 Attendance Policy" in text
    assert "80% attendance" in text


def test_chunking_structure_aware_and_overlap() -> None:
    doc = Document(
        source="handbook.pdf",
        doc_version="2026-06",
        superseded=False,
        pages=[
            Page(
                source="handbook.pdf",
                page=1,
                text="### 1.21 Attendance Policy\nAttendance is mandatory for all students.\n\n"
                     "### 5.4 Academic Warning\nA student whose CGPA falls below 2.00 is placed on academic warning.\n"
                     "Additional guidelines for probationary status apply.",
            )
        ],
    )
    chunks = chunk(doc, size=15, overlap=3)
    assert len(chunks) >= 2
    # Check that metadata is preserved
    sections = [c.section for c in chunks]
    assert any("1.21 Attendance" in s for s in sections)
    assert any("5.4 Academic Warning" in s for s in sections)
    for c in chunks:
        assert c.source == "handbook.pdf"
        assert c.doc_version == "2026-06"
        assert c.superseded is False
        assert c.id is not None
        meta = c.metadata()
        assert set(meta.keys()) == {"source", "section", "page", "doc_version", "superseded"}


def test_deterministic_vectors() -> None:
    v1 = _deterministic_stub_vector("prerequisite course repeat")
    v2 = _deterministic_stub_vector("prerequisite course repeat")
    v3 = _deterministic_stub_vector("hostel accommodation curfew")
    assert v1 == v2
    assert v1 != v3
    norm = math.sqrt(sum(x * x for x in v1))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_offline_vectors_normalize_punctuation_and_inflection() -> None:
    punctuation = _deterministic_stub_vector("ATTENDANCE? 80%")
    plain = _deterministic_stub_vector("attendance 80")
    inflected = _deterministic_stub_vector("repeating courses")
    base = _deterministic_stub_vector("repeat course")
    assert punctuation == plain
    assert inflected == base


def test_embed_fallback_offline(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_URL", "offline")
    settings = get_settings()
    texts = ["Clause 1.21 attendance", "Fee structure tuition"]
    embeddings = embed(texts, settings)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
