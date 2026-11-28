"""A03 Part A: batch-safe incremental indexing and Batch B verification.

Runs the real pipeline against temporary copies of Batch A (indexer/corpus) and
Batch B (corpus/batch_b) with an in-memory fake Chroma and offline embeddings.
Nothing touches a real Chroma collection or the repository's data folders.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

import pipeline.archive as archive_mod
import pipeline.datalake as datalake_mod
import pipeline.quarantine as quarantine_mod
import pipeline.run as run_mod
import pipeline.store as store_mod
from pipeline.config import get_settings

HERE = Path(__file__).resolve().parent
BATCH_A = HERE.parent / "corpus"
BATCH_B = HERE.parent.parent / "corpus" / "batch_b"


class FakeCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name, self.metadata, self.rows = name, metadata, {}

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self.rows[i] = (d, e, m)

    add = upsert

    def delete(self, ids) -> None:
        for i in ids:
            self.rows.pop(i, None)

    def count(self) -> int:
        return len(self.rows)


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name, metadata):
        return self.collections.setdefault(name, FakeCollection(name, metadata))

    def get_collection(self, name):
        return self.collections[name]

    def delete_collection(self, name):
        self.collections.pop(name)


@pytest.fixture()
def lake(tmp_path, monkeypatch):
    for mod, attr in ((datalake_mod, "DATA_LAKE_DIR"), (quarantine_mod, "QUARANTINE_DIR"),
                      (archive_mod, "ARCHIVE_DIR")):
        monkeypatch.setattr(mod, attr, str(tmp_path / attr.lower()))
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.delenv("INGEST_BATCH", raising=False)
    client = FakeClient()
    monkeypatch.setattr(store_mod, "get_client", lambda s: client)
    a = tmp_path / "corpus"
    b = tmp_path / "batch_b"
    shutil.copytree(BATCH_A, a)
    shutil.copytree(BATCH_B, b)
    base = dataclasses.replace(get_settings(), ollama_url="offline", collection="rahbar")

    def run(corpus: Path, **kw) -> int:
        return run_mod.run_pipeline(dataclasses.replace(base, corpus_dir=str(corpus)), **kw)

    def rows() -> dict:
        return client.collections["rahbar"].rows

    def sources() -> set[str]:
        return {m["source"] for _, _, m in rows().values()}

    def registry() -> dict:
        return {r.doc_id: r for r in datalake_mod.load_registry().values()}

    def report() -> dict:
        reps = sorted((tmp_path / "data_lake_dir" / "reports").glob("*.json"),
                      key=lambda p: p.stat().st_mtime_ns)
        docs = json.loads(reps[-1].read_text())["documents"]
        return {d["source"]: d for d in docs}

    return dataclasses.make_dataclass("Lake", ["a", "b", "run", "rows", "sources",
                                               "registry", "report", "tmp"])(
        a, b, run, rows, sources, registry, report, tmp_path)


BATCH_A_DOCS = {"academic-rules-2023.md", "academic-rules-2026.md", "fees-2026.md"}
BATCH_B_VALID = {"academic-rules-2027.md", "hostel-rules-2026.md", "library-policy-2026.txt",
                 "it-policy-with-injection.md", "transport-routes-2026.md"}


def assert_registry_consistent(lake) -> None:
    """Every live registry chunk is in Chroma, and Chroma holds nothing unregistered."""
    reg_ids = {cid for r in lake.registry().values() if not r.archived for cid in r.chunk_ids}
    assert reg_ids == set(lake.rows()), (reg_ids ^ set(lake.rows()))


# --------------------------------------------------------------- Phase 2: batches
def test_batch_b_does_not_delete_batch_a(lake):
    lake.run(lake.a)
    a_ids = set(lake.rows())
    a_doc_ids = {r.doc_id for r in lake.registry().values()}
    assert lake.sources() == BATCH_A_DOCS and a_ids

    lake.run(lake.b)
    assert a_ids <= set(lake.rows()), "Batch A chunks were deleted by a Batch B run"
    reg = lake.registry()
    assert a_doc_ids <= set(reg) and not any(reg[d].archived or reg[d].superseded is None
                                             for d in a_doc_ids)
    assert BATCH_A_DOCS | BATCH_B_VALID <= lake.sources()
    assert {r.batch for r in reg.values()} == {"corpus", "batch_b"}
    assert_registry_consistent(lake)


def test_reprocessing_batch_b_creates_no_duplicates(lake):
    lake.run(lake.a)
    lake.run(lake.b)
    before = dict(lake.rows())
    assert lake.run(lake.b) == 0
    assert lake.rows().keys() == before.keys()
    assert_registry_consistent(lake)


def test_invalid_batch_b_document_leaves_valid_documents_alone(lake):
    lake.run(lake.a)
    lake.run(lake.b)
    before = set(lake.rows())
    (lake.b / "broken-policy-2026.md").write_text(
        "---\ntitle: Broken\ndoc_version: 2026\n---\n# Broken\n\nThe committee shall review all\n",
        encoding="utf-8")
    lake.run(lake.b)
    assert lake.report()["broken-policy-2026.md"]["quarantined"]
    assert set(lake.rows()) == before
    assert_registry_consistent(lake)


def test_removing_a_batch_b_doc_only_removes_that_doc(lake):
    lake.run(lake.a)
    lake.run(lake.b)
    hostel = {i for i, (_, _, m) in lake.rows().items() if m["source"] == "hostel-rules-2026.md"}
    (lake.b / "hostel-rules-2026.md").unlink()
    lake.run(lake.b)
    assert not hostel & set(lake.rows())
    assert (BATCH_A_DOCS | BATCH_B_VALID) - {"hostel-rules-2026.md"} <= lake.sources()
    rec = next(r for r in lake.registry().values() if r.source == "hostel-rules-2026.md")
    assert rec.archived and rec.chunk_ids == []
    assert "Hostel" in Path(rec.archive_path).read_text(encoding="utf-8")
    assert_registry_consistent(lake)


def test_changed_document_replaces_only_its_chunks_and_archives_real_text(lake):
    lake.run(lake.a)
    lake.run(lake.b)
    other = {i for i, (_, _, m) in lake.rows().items() if m["source"] != "fees-2026.md"}
    f = lake.a / "fees-2026.md"
    old = f.read_text(encoding="utf-8")
    f.write_text(old.replace("No refund is given after day 20.",
                             "No refund is given after day 25."), encoding="utf-8")
    lake.run(lake.a)
    assert other <= set(lake.rows())
    fees_text = " ".join(d for d, _, m in lake.rows().values() if m["source"] == "fees-2026.md")
    assert "day 25" in fees_text and "after day 20" not in fees_text, "stale chunk kept"
    archived = list((lake.tmp / "archive_dir").glob("*.txt"))
    assert any("after day 20" in p.read_text(encoding="utf-8") for p in archived)
    assert_registry_consistent(lake)


def test_quarantined_update_keeps_the_indexed_version(lake):
    lake.run(lake.a)
    before = set(lake.rows())
    f = lake.a / "fees-2026.md"
    f.write_text(f.read_text(encoding="utf-8").rstrip().rstrip(".") + " and the remaining\n",
                 encoding="utf-8")
    lake.run(lake.a)
    assert lake.report()["fees-2026.md"]["quarantined"]
    assert set(lake.rows()) == before
    assert_registry_consistent(lake)


def test_full_reindex_of_batch_b_keeps_batch_a(lake):
    lake.run(lake.a)
    lake.run(lake.b)
    a_ids = {i for i, (_, _, m) in lake.rows().items() if m["source"] in BATCH_A_DOCS}
    lake.run(lake.b, full_reindex=True)
    assert a_ids <= set(lake.rows())
    assert BATCH_B_VALID <= lake.sources()
    assert_registry_consistent(lake)


def test_legacy_registry_without_batch_is_treated_as_primary_corpus(lake):
    lake.run(lake.a)
    reg = datalake_mod.load_registry()
    for r in reg.values():
        r.batch = ""                      # registry written before batches existed
    datalake_mod.save_registry(reg)
    a_ids = set(lake.rows())
    lake.run(lake.b)
    assert a_ids <= set(lake.rows())


# ------------------------------------------------- Phase 3: Batch B verification
def test_batch_b_planted_flaws(lake):
    lake.run(lake.a)
    lake.run(lake.b)
    rep = lake.report()

    def failed(name: str) -> set[str]:
        return {c["name"] for c in rep[name]["checks"] if not c["passed"]}

    assert "completeness_truncation" in failed("exam-policy-truncated.txt")
    assert "completeness_date" in failed("scholarship-policy-no-metadata.txt")
    assert failed("academic-rules-2026-copy.md") == {"uniqueness_exact_dup"}
    assert failed("fees-2026-duplicate.md") == {"uniqueness_exact_dup"}
    for name in BATCH_B_VALID:
        assert not rep[name]["quarantined"], (name, rep[name]["quarantine_reason"])
    fam = next(c for c in rep["academic-rules-2027.md"]["checks"] if c["name"] == "recency_family")
    assert fam["passed"] and "academic-rules-2026.md" in fam["reason"]
    # Quarantined documents never reach Chroma
    assert not {"exam-policy-truncated.txt", "scholarship-policy-no-metadata.txt",
                "academic-rules-2026-copy.md", "fees-2026-duplicate.md"} & lake.sources()


def test_true_duplicates_of_indexed_documents_are_quarantined(lake):
    """The detector itself: real copies of Batch A documents arriving in Batch B."""
    lake.run(lake.a)
    for f in lake.b.iterdir():
        f.unlink()
    shutil.copy(lake.a / "fees-2026.md", lake.b / "fees-copy.md")                  # exact
    txt = (lake.a / "academic-rules-2026.md").read_text(encoding="utf-8")
    (lake.b / "rules-reformatted.md").write_text(txt.replace("\n\n", "\n\n\n"),   # whitespace
                                                encoding="utf-8")
    near = txt.replace("80 percent", "eighty percent")                             # near-dup
    (lake.b / "rules-near.md").write_text(near, encoding="utf-8")
    (lake.b / "hostel-2026.md").write_text(
        "---\ntitle: Hostel\ndoc_version: 2026\n---\n# Hostel\n\nCurfew is 10 PM on weekdays.\n",
        encoding="utf-8")
    lake.run(lake.b)
    rep = lake.report()
    assert "Exact duplicate" in rep["fees-copy.md"]["quarantine_reason"]
    assert "Exact duplicate" in rep["rules-reformatted.md"]["quarantine_reason"]
    assert "Near-duplicate" in rep["rules-near.md"]["quarantine_reason"]
    assert not rep["hostel-2026.md"]["quarantined"]


def test_in_batch_duplicate_pair_keeps_the_first_copy(lake):
    for f in lake.b.iterdir():
        f.unlink()
    body = "---\ntitle: Gym\ndoc_version: 2026\n---\n# Gym\n\nThe gym opens at 6 AM daily.\n"
    (lake.b / "a-gym-2026.md").write_text(body, encoding="utf-8")
    (lake.b / "b-gym-copy-2026.md").write_text(body, encoding="utf-8")
    lake.run(lake.b)
    rep = lake.report()
    assert not rep["a-gym-2026.md"]["quarantined"]
    assert rep["b-gym-copy-2026.md"]["quarantined"]


def test_older_family_member_is_warned_not_quarantined(lake):
    """fees-2024 vs fees-2026 could be a replaced version or a dated series
    (annual-report-2019-20 / 2023-24): report it, but keep a valid document."""
    lake.run(lake.a)
    for f in lake.b.iterdir():
        f.unlink()
    (lake.b / "fees-2024.md").write_text(
        "---\ntitle: Fees\ndoc_version: 2024\n---\n# Fees\n\nTuition was Rs. 8,000 per credit hour.\n",
        encoding="utf-8")
    lake.run(lake.b)
    doc = lake.report()["fees-2024.md"]
    fam = next(c for c in doc["checks"] if c["name"] == "recency_family")
    assert not doc["quarantined"]
    assert "WARNING possible stale version 2024" in fam["reason"] and "fees-2026.md" in fam["reason"]
    assert "fees-2024.md" in lake.sources()


def test_truncation_check_ignores_pdf_and_html_endings():
    """A PDF whose extracted text ends on a table cell must not be quarantined."""
    from pipeline.models import Document, Page
    from pipeline.verify import check_completeness

    def trunc(source: str, text: str) -> bool:
        doc = Document(source=source, doc_version="2026", superseded=False,
                       pages=[Page(source=source, page=1, text=text)])
        doc.populate_derived()
        return not next(c for c in check_completeness(doc)
                        if c.name == "completeness_truncation").passed

    ending = "Policy text.\n\nMetrics to be used for monitoring progress"
    assert not trunc("policy.pdf", ending)
    assert not trunc("page.html", ending)
    assert trunc("policy.txt", ending)
    assert not trunc("hostel.md", "Rules.\n\nDinner: 7:30 PM – 9:00 PM")


def test_near_duplicate_uses_document_vectors_and_skips_version_families():
    """Embedding near-duplicate check: report band, quarantine band, family exclusion."""
    from pipeline.models import Document, Page
    from pipeline.verify import check_uniqueness

    def doc(name: str, text: str) -> Document:
        d = Document(source=name, doc_version="2026", superseded=False,
                     pages=[Page(source=name, page=1, text=text)])
        d.populate_derived()
        return d

    orig = doc("rules-2026.md", "Students must attend eighty percent of lectures.")
    para = doc("rules-paraphrased.md", "Learners have to be present in most classes.")
    newer = doc("rules-2027.md", "Students must attend seventy five percent of lectures.")
    vecs = {orig.doc_id: [1.0, 0.0], para.doc_id: [0.93, 0.37], newer.doc_id: [0.99, 0.14]}

    def near(d: Document, threshold: float = 0.97) -> dict:
        res = check_uniqueness(d, {}, [orig, d], doc_vectors=vecs, near_dup_threshold=threshold)
        return next(c for c in res if c.name == "uniqueness_near_dup").__dict__

    assert near(para)["passed"] and "WARNING possible near-duplicate of: rules-2026.md" in near(para)["reason"]
    assert "[embedding" in near(para)["reason"]
    assert not near(para, threshold=0.92)["passed"]          # quarantine band is configurable
    assert near(newer)["passed"] and near(newer)["reason"] == ""   # same family: not a duplicate


def test_rollback_restores_archived_version(lake):
    """Restore demo: change a document, roll it back from the archive, re-ingest."""
    lake.run(lake.a)
    f = lake.a / "fees-2026.md"
    original = f.read_text(encoding="utf-8")
    original_hash = next(r.doc_hash for r in lake.registry().values() if r.source == "fees-2026.md")
    f.write_text(original.replace("after day 20.", "after day 25."), encoding="utf-8")
    lake.run(lake.a)
    archived = [p for p in (lake.tmp / "archive_dir").glob("*.txt")
                if "after day 20" in p.read_text(encoding="utf-8")]
    assert archived, "previous version must be archived before it is replaced"

    archive_mod.restore_document(str(archived[0]), str(f))      # rollback
    lake.run(lake.a)
    fees_text = " ".join(d for d, _, m in lake.rows().values() if m["source"] == "fees-2026.md")
    assert "after day 20" in fees_text and "after day 25" not in fees_text
    rec = next(r for r in lake.registry().values() if r.source == "fees-2026.md")
    assert rec.doc_hash == original_hash
    assert_registry_consistent(lake)


def test_unchanged_rerun_makes_no_embedding_calls(lake, monkeypatch):
    """Re-running unchanged ingestion embeds zero chunks and zero documents."""
    import pipeline.embed as embed_mod

    lake.run(lake.a)
    lake.run(lake.b)
    calls: list[int] = []
    real = embed_mod.embed
    monkeypatch.setattr(embed_mod, "embed", lambda texts, s: calls.append(len(texts)) or real(texts, s))
    assert lake.run(lake.a) == 0 and lake.run(lake.b) == 0
    assert calls == [], f"unchanged runs called the embedder: {calls}"
