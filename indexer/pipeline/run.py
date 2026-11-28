"""The indexer Job's entry point. It runs once and exits (section 4.2).

A03 upgrade: full RAGOps pipeline with verification, incremental indexing,
data lake management, quarantine and archival.

    python -m pipeline.run --dummy        # three fake chunks, no model needed
    python -m pipeline.run                # real pipeline
    python -m pipeline.run --full-reindex # rebuild this batch (wipes Chroma only if no other batch)
    python -m pipeline.run --corpus-dir /corpus/batch_b  # ingest Batch B alongside Batch A

Batches: every registry record belongs to a batch (--batch, INGEST_BATCH, or the
corpus folder name). A run only removes documents of its own batch.

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
from .models import Chunk, DataLakeRecord, Document


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
        meta = _front_matter(path) if path.suffix.lower() in {".md", ".txt"} else {}
        version = meta.get("doc_version", "") or _year_from_name(path.name) or "unknown"
        found.append((path, version, meta.get("superseded", "false").lower() == "true"))
    return found


def _year_from_name(name: str) -> str:
    """Fallback version for files without front matter: library-policy-2026.txt -> 2026."""
    import re

    match = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", name)
    return match.group(0) if match else ""


def batch_name(corpus_dir: str, batch: str | None = None) -> str:
    """The ingestion batch: --batch, else INGEST_BATCH, else the corpus folder name."""
    return batch or os.environ.get("INGEST_BATCH", "") or Path(corpus_dir).name or "corpus"


def run_pipeline(settings: IndexerSettings, full_reindex: bool = False,
                 batch: str | None = None) -> int:
    """Full RAGOps pipeline with verification, incremental indexing and data lake.

    Batch-scoped: a run only deletes or supersedes registry documents of its own
    batch, so ingesting Batch B never removes Batch A's documents or chunks.
    """
    from .archive import archive_document
    from .chunk import chunk
    from .clean import clean
    from .datalake import (
        find_orphan_chunk_ids,
        in_batch,
        load_registry,
        load_snapshot,
        mark_superseded,
        save_registry,
        save_snapshot,
        upsert_record,
    )
    from .embed import embed
    from .quarantine import quarantine_document
    from .store import store
    from .verify import generate_report, verify

    run_id = uuid.uuid4().hex[:8]
    batch_id = batch_name(settings.corpus_dir, batch)
    # Records written before batches existed belong to the primary CORPUS_DIR.
    legacy_batch = Path(get_settings().corpus_dir).name or "corpus"
    log("pipeline starting", run_id=run_id, full_reindex=full_reindex,
        corpus_dir=settings.corpus_dir, batch=batch_id)

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
        fm = _front_matter(path) if path.suffix.lower() in {".md", ".txt"} else {}
        doc.date = fm.get("date", "")
        doc.title = fm.get("title", "")
        doc.populate_derived()
        all_docs.append((path, doc))

    def _text(d: Document) -> str:
        return "\n".join(p.text for p in d.pages)

    # Texts of documents already in the data lake (any batch), for duplicate checks
    reference_texts: dict[str, tuple[str, str]] = {}
    for rec in registry.values():
        if rec.archived:
            continue
        snap = load_snapshot(rec.doc_id, rec.doc_hash)
        if snap is not None:
            reference_texts[rec.doc_id] = (rec.source, snap)

    # Whole-document embeddings (pinned EMBED_MODEL) for the near-duplicate check,
    # cached in the data lake per content hash: an unchanged re-run embeds nothing.
    from .datalake import load_doc_vector, save_doc_vector

    batch_ids = {d.doc_id for _, d in all_docs}
    wanted: dict[str, tuple[str, str]] = {d.doc_id: (d.doc_hash, _text(d)) for _, d in all_docs}
    for rid, (_src, txt) in reference_texts.items():
        if rid not in batch_ids:
            wanted[rid] = (registry[rid].doc_hash, txt)
    doc_vectors: dict[str, list[float]] = {}
    missing: list[str] = []
    for did, (dhash, _txt) in wanted.items():
        cached = load_doc_vector(did, dhash, settings.embed_model)
        if cached:
            doc_vectors[did] = cached
        else:
            missing.append(did)
    try:
        if missing:
            for did, vec in zip(missing, embed([wanted[d][1] for d in missing], settings)):
                doc_vectors[did] = vec
                save_doc_vector(did, wanted[did][0], settings.embed_model, vec)
        log("near_dup_vectors", method="embedding", model=settings.embed_model,
            docs=len(doc_vectors), newly_embedded=len(missing))
    except Exception as exc:
        doc_vectors = {}
        log("near_dup_vectors", method="bag-of-words", reason=f"embedding unavailable: {exc}")

    # ---- 4. Verification pass -------------------------------------------------
    ver_results = []
    for path, doc in all_docs:
        vr = verify(doc, registry, [d for _, d in all_docs], reference_texts=reference_texts,
                    doc_vectors=doc_vectors)
        ver_results.append(vr)
        if vr.quarantined:
            log("quarantined", source=doc.source, reasons=vr.quarantine_reason)
            quarantine_document(str(path), [vr.quarantine_reason], doc.to_dict() if hasattr(doc, "to_dict") else {})
        else:
            log("verified_ok", source=doc.source)

    # Write verification report
    # Written to the data lake PVC: CORPUS_DIR is a read-only ConfigMap mount in the cluster.
    from . import datalake as _datalake

    report = generate_report(ver_results, run_id)
    report_path = Path(_datalake.DATA_LAKE_DIR) / "reports" / f"verification_report_{run_id}.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log("verification_report_written", path=str(report_path), summary=report["summary"])
    except Exception as exc:
        log("verification_report_write_failed", error=str(exc))

    # ---- 5. Incremental logic (scoped to this batch) ---------------------------
    other_batches_present = any(
        not in_batch(r, batch_id, legacy_batch) and not r.archived and r.chunk_ids
        for r in registry.values()
    )
    # --full-reindex wipes the whole collection only when no other batch has chunks
    # in it; otherwise it rebuilds just this batch and leaves other batches intact.
    wipe_collection = full_reindex and not other_batches_present
    if full_reindex and not wipe_collection:
        log("full_reindex_scoped_to_batch", batch=batch_id,
            reason="other batches are indexed in this collection")

    active_doc_ids: set[str] = set()
    docs_to_index: list[tuple[Path, Document]] = []
    stale_chunk_ids: list[str] = []   # old chunks of changed documents

    for vr, (path, doc) in zip(ver_results, all_docs):
        existing = registry.get(doc.doc_id)
        if vr.quarantined:
            # A rejected update must not delete the version that is already indexed.
            if existing is not None and not existing.archived:
                active_doc_ids.add(doc.doc_id)
                log("quarantined_update_kept_previous_version", source=doc.source,
                    kept_version=existing.doc_version)
            continue
        active_doc_ids.add(doc.doc_id)
        if full_reindex:
            if existing is not None:
                stale_chunk_ids.extend(existing.chunk_ids)
            docs_to_index.append((path, doc))
        elif existing is None:
            # Brand new document
            log("new_document", source=doc.source, version=doc.doc_version)
            docs_to_index.append((path, doc))
        elif existing.doc_hash != doc.doc_hash:
            # Content changed → archive the real previous text, replace its chunks
            log("changed_document", source=doc.source,
                old_version=existing.doc_version, new_version=doc.doc_version)
            old_text = load_snapshot(existing.doc_id, existing.doc_hash)
            archive_document(existing, old_text if old_text is not None else
                             f"[previous text not in data lake; source={existing.source}]")
            mark_superseded(registry, doc.doc_id)
            stale_chunk_ids.extend(existing.chunk_ids)
            docs_to_index.append((path, doc))
        else:
            log("unchanged", source=doc.source, version=doc.doc_version)
            if not existing.batch:
                existing.batch = batch_id   # adopt legacy record into this batch
            # Update superseded flag in Chroma if it changed
            if existing.superseded != doc.superseded:
                docs_to_index.append((path, doc))
        save_snapshot(doc.doc_id, doc.doc_hash, _text(doc))

    # Chunks of this batch's documents that are gone from its corpus, plus stale
    # chunks of changed documents. Other batches' chunks are never touched.
    removed_ids = {
        rec.doc_id for rec in registry.values()
        if in_batch(rec, batch_id, legacy_batch)
        and rec.doc_id not in active_doc_ids and not rec.archived
    }
    orphan_ids = find_orphan_chunk_ids(registry, active_doc_ids,
                                       batch=batch_id, legacy_batch=legacy_batch)
    orphan_ids = list(dict.fromkeys(orphan_ids + stale_chunk_ids))
    if orphan_ids:
        log("orphan_chunks_to_delete", count=len(orphan_ids), batch=batch_id)

    if not docs_to_index and not orphan_ids:
        log("incremental_no_changes", message="Zero new chunks embedded – corpus unchanged")
        save_registry(registry)   # persists batch adoption of legacy records
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
                      full_reindex=wipe_collection,
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
            batch=batch_id,
        )
        upsert_record(registry, record)
        save_snapshot(doc.doc_id, doc.doc_hash, _text(doc))

    # Documents removed from THIS batch's corpus: their chunks were deleted above,
    # so archive them (real text from the data lake) and clear their chunk list.
    for doc_id in removed_ids:
        rec = registry[doc_id]
        snap = load_snapshot(rec.doc_id, rec.doc_hash)
        rec.archive_path = archive_document(
            rec, snap if snap is not None else f"[text not in data lake; source={rec.source}]")
        rec.archived = True
        rec.superseded = True
        rec.chunk_ids = []
        log("removed_document_archived", source=rec.source, batch=batch_id)

    save_registry(registry)
    log("registry_saved", records=len(registry))

    return count


def invalidate_cache(settings: IndexerSettings) -> None:
    """Invalidate semantic cache across the cluster after indexing completes (section 3.3)."""
    import httpx

    api_url = os.environ.get("API_URL", "http://rahbar-api:8000")
    token = os.environ.get("CACHE_INVALIDATION_TOKEN", "")

    if not token:
        raise RuntimeError("CACHE_INVALIDATION_TOKEN is not configured")

    for attempt in range(1, 13):
        try:
            response = httpx.post(
                f"{api_url}/api/cache/invalidate",
                headers={"X-Cache-Invalidation-Token": token},
                timeout=4.0,
            )
            response.raise_for_status()
            log("api cache invalidation completed", attempt=attempt)
            return

        except httpx.HTTPStatusError as exc:
            # A rejected token will not fix itself by retrying: fail fast and say why.
            if exc.response.status_code in (401, 403):
                raise RuntimeError(
                    "cache invalidation rejected by the API (HTTP "
                    f"{exc.response.status_code}): CACHE_INVALIDATION_TOKEN does not "
                    "match the API's value"
                ) from exc
            if attempt == 12:
                raise RuntimeError(
                    "cache invalidation did not complete after indexing"
                ) from exc
            log(
                "waiting for API cache invalidation",
                attempt=attempt,
                error=str(exc),
            )
            time.sleep(2)

        except httpx.HTTPError as exc:
            if attempt == 12:
                raise RuntimeError(
                    "cache invalidation did not complete after indexing"
                ) from exc

            log(
                "waiting for API cache invalidation",
                attempt=attempt,
                error=str(exc),
            )
            time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    parser.add_argument("--full-reindex", action="store_true",
                        help="wipe Chroma collection and rebuild from scratch")
    parser.add_argument("--corpus-dir", default=None,
                        help="override CORPUS_DIR for this run (e.g. for Batch B)")
    parser.add_argument("--batch", default=None,
                        help="ingestion batch name (default: INGEST_BATCH or the corpus "
                             "folder name); only this batch's documents can be removed")
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
            count = run_pipeline(settings, full_reindex=args.full_reindex, batch=args.batch)
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
