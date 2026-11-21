"""The indexer Job's entry point. It runs once and exits (section 4.2).

A03 upgrade: full RAGOps pipeline with verification, incremental indexing,
data lake management, quarantine and archival.

    python -m pipeline.run --dummy        # three fake chunks, no model needed
    python -m pipeline.run                # real pipeline
    python -m pipeline.run --full-reindex # wipe Chroma and rebuild from scratch
    python -m pipeline.run --corpus-dir /corpus/batch_b  # ingest a specific batch

Exit code 0: index written.
Exit code 1: unrecoverable error.
Exit code 2: stage not yet implemented (kept for compatibility).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from .config import IndexerSettings, get_settings
from .models import Chunk, DataLakeRecord


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer", "msg": msg, **fields}), flush=True)


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index."""
    root = Path(corpus_dir)
    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        return [(root / e["file"], str(e.get("doc_version", "unknown")),
                 bool(e.get("superseded", False))) for e in entries]
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append((path, meta.get("doc_version", "unknown"),
                      meta.get("superseded", "false").lower() == "true"))
    return found


def run_pipeline(settings: IndexerSettings, full_reindex: bool = False) -> int:
    """Full RAGOps pipeline with verification, incremental indexing and data lake."""
    from .archive import archive_document
    from .chunk import chunk
    from .clean import clean
    from .datalake import (
        find_orphan_chunk_ids,
        load_registry,
        mark_superseded,
        save_registry,
        upsert_record,
    )
    from .embed import embed
    from .extract import extract
    from .quarantine import quarantine_document
    from .store import store
    from .verify import generate_report, verify

    run_id = uuid.uuid4().hex[:8]
    log("pipeline starting", run_id=run_id, full_reindex=full_reindex,
        corpus_dir=settings.corpus_dir)

    # ---- 1. Discover documents ------------------------------------------------
    documents_meta = discover(settings.corpus_dir)
    if not documents_meta:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")
    log("discovered", count=len(documents_meta))

    # ---- 2. Load data lake registry -------------------------------------------
    registry = load_registry()
    log("registry loaded", records=len(registry))

    # ---- 3. Extract all documents first (needed for near-dup cross-checks) ----
    from .extract import extract as _extract
    all_docs = []
    for path, doc_version, superseded in documents_meta:
        doc = clean(_extract(str(path), doc_version, superseded))
        # Read front matter for extra metadata
        fm = _front_matter(path) if path.suffix.lower() == ".md" else {}
        doc.date = fm.get("date", "")
        doc.title = fm.get("title", "")
        doc.populate_derived()
        all_docs.append((path, doc))

    # ---- 4. Verification pass -------------------------------------------------
    ver_results = []
    for path, doc in all_docs:
        vr = verify(doc, registry, [d for _, d in all_docs])
        ver_results.append(vr)
        if vr.quarantined:
            log("quarantined", source=doc.source, reasons=vr.quarantine_reason)
            quarantine_document(str(path), [vr.quarantine_reason], doc.to_dict() if hasattr(doc, "to_dict") else {})
        else:
            log("verified_ok", source=doc.source)

    # Write verification report
    report = generate_report(ver_results, run_id)
    report_path = Path(settings.corpus_dir) / f"verification_report_{run_id}.json"
    try:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log("verification_report_written", path=str(report_path))
    except Exception as exc:
        log("verification_report_write_failed", error=str(exc))

    # ---- 5. Incremental logic -------------------------------------------------
    # Active doc IDs in this run (excluding quarantined)
    active_doc_ids: set[str] = set()
    docs_to_index: list[tuple[Path, object]] = []  # (path, Document)

    for vr, (path, doc) in zip(ver_results, all_docs):
        if vr.quarantined:
            continue
        active_doc_ids.add(doc.doc_id)
        existing = registry.get(doc.doc_id)
        if existing is None:
            # Brand new document
            log("new_document", source=doc.source, version=doc.doc_version)
            docs_to_index.append((path, doc))
        elif existing.doc_hash != doc.doc_hash:
            # Content changed → archive old version, re-index new
            log("changed_document", source=doc.source,
                old_version=existing.doc_version, new_version=doc.doc_version)
            full_text_old = existing.source  # we don't store full text in registry
            archive_path = archive_document(existing, full_text_old)
            mark_superseded(registry, doc.doc_id)
            docs_to_index.append((path, doc))
        else:
            log("unchanged", source=doc.source, version=doc.doc_version)
            # Update superseded flag in Chroma if it changed
            if existing.superseded != doc.superseded:
                docs_to_index.append((path, doc))

    # Find orphan chunk IDs from documents no longer in the corpus
    orphan_ids = find_orphan_chunk_ids(registry, active_doc_ids)
    if orphan_ids:
        log("orphan_chunks_to_delete", count=len(orphan_ids))

    if not docs_to_index and not orphan_ids:
        log("incremental_no_changes", message="Zero new chunks embedded – corpus unchanged")
        return 0

    # ---- 6. Chunk + embed only changed documents ------------------------------
    all_chunks: list[Chunk] = []
    for _path, doc in docs_to_index:
        pieces = chunk(doc, settings.chunk_size, settings.chunk_overlap)
        # Stamp doc_id onto each chunk
        for piece in pieces:
            piece.doc_id = doc.doc_id
        log("chunked", source=doc.source, chunks=len(pieces))
        all_chunks.extend(pieces)

    if all_chunks:
        vectors = embed([c.text for c in all_chunks], settings)
        count = store(all_chunks, vectors, settings,
                      full_reindex=full_reindex,
                      orphan_chunk_ids=orphan_ids)
    else:
        # Only orphan deletion
        from .store import store as _store
        count = _store([], [], settings, full_reindex=False, orphan_chunk_ids=orphan_ids)

    # ---- 7. Update data lake registry -----------------------------------------
    # Group chunks by doc_id
    chunk_ids_by_doc: dict[str, list[str]] = {}
    for c in all_chunks:
        chunk_ids_by_doc.setdefault(c.doc_id, []).append(c.id)

    for _path, doc in docs_to_index:
        record = DataLakeRecord(
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            doc_version=doc.doc_version,
            doc_hash=doc.doc_hash,
            date=doc.date,
            page_count=doc.page_count,
            chunk_ids=chunk_ids_by_doc.get(doc.doc_id, []),
            superseded=doc.superseded,
        )
        upsert_record(registry, record)

    # Mark documents removed from corpus as superseded in registry
    for doc_id, rec in registry.items():
        if doc_id not in active_doc_ids and not rec.archived:
            rec.superseded = True

    save_registry(registry)
    log("registry_saved", records=len(registry))

    return count


def invalidate_cache(settings: IndexerSettings) -> None:
    """Invalidate semantic cache across the cluster after indexing (section 3.3)."""
    try:
        import redis
        redis_host = os.environ.get("REDIS_HOST", "redis")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        r = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=2.0)
        r.flushdb()
        log("redis_cache_flushed")
    except Exception as exc:
        log("redis_flush_skipped", error=str(exc))

    try:
        import httpx
        api_url = os.environ.get("API_URL", "http://rahbar-api:8000")
        httpx.post(f"{api_url}/api/cache/invalidate", timeout=3.0)
        log("api_cache_invalidated")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    parser.add_argument("--full-reindex", action="store_true",
                        help="wipe Chroma collection and rebuild from scratch")
    parser.add_argument("--corpus-dir", default=None,
                        help="override CORPUS_DIR for this run (e.g. for Batch B)")
    args = parser.parse_args()

    settings = get_settings()
    if args.corpus_dir:
        import dataclasses
        settings = dataclasses.replace(settings, corpus_dir=args.corpus_dir)

    log("indexer starting", dummy=args.dummy,
        full_reindex=args.full_reindex,
        embed_model=settings.embed_model,
        collection=settings.collection,
        chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        corpus_dir=settings.corpus_dir,
        pod=os.environ.get("HOSTNAME", "-"))

    started = time.perf_counter()
    try:
        if args.dummy:
            from .dummy import write_dummy_index
            from .store import get_client
            count = write_dummy_index(settings, get_client(settings))
        else:
            count = run_pipeline(settings, full_reindex=args.full_reindex)
        invalidate_cache(settings)
    except NotImplementedError as exc:
        log("stage_not_implemented", error=str(exc))
        return 2
    except Exception as exc:
        log("indexer_error", error=str(exc))
        return 1

    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
