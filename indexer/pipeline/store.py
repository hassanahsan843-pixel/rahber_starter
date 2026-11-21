"""Stage 5: Store. The only code in the whole system that writes to Chroma.

A03 upgrade: incremental upsert.
  - On first run, creates the collection and upserts all chunks.
  - On subsequent runs, only upserts chunks from changed documents.
  - Deletes orphan chunk IDs (from documents removed from the corpus).
  - Full-wipe mode is still available via --full-reindex flag.
"""

from __future__ import annotations

from typing import Any

from .config import IndexerSettings
from .models import Chunk


def get_client(settings: IndexerSettings) -> Any:
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_or_create_collection(client: Any, settings: IndexerSettings, dimension: int) -> Any:
    metadata = {
        "embed_model": settings.embed_model,
        "dimension": dimension,
        "hnsw:space": "cosine",
    }
    try:
        return client.get_or_create_collection(
            name=settings.collection,
            metadata=metadata,
        )
    except Exception:
        try:
            import httpx
            httpx.post(
                f"http://{settings.chroma_host}:{settings.chroma_port}/api/v1/collections",
                json={"name": settings.collection, "metadata": metadata, "get_or_create": True},
                timeout=5.0,
            )
            return client.get_collection(settings.collection)
        except Exception:
            return client.create_collection(name=settings.collection, metadata=metadata)


def store(
    chunks: list[Chunk],
    vectors: list[list[float]],
    settings: IndexerSettings,
    full_reindex: bool = False,
    orphan_chunk_ids: list[str] | None = None,
) -> int:
    """Write chunks with their vectors incrementally.

    - full_reindex=True: delete the entire collection first (backward compat).
    - full_reindex=False (default): upsert only the provided chunks; also
      delete any IDs in orphan_chunk_ids (chunks from deleted/replaced docs).

    Returns the total number of chunks written.
    """
    if not chunks:
        # Still need to delete orphans even if nothing new to add
        if orphan_chunk_ids:
            client = get_client(settings)
            try:
                col = client.get_collection(settings.collection)
                _delete_in_batches(col, orphan_chunk_ids)
            except Exception:
                pass
        return 0

    client = get_client(settings)
    dimension = len(vectors[0]) if vectors else 768

    if full_reindex:
        # Wipe collection – used for first-time setup or forced rebuild
        try:
            client.delete_collection(settings.collection)
        except Exception:
            pass

    collection = _get_or_create_collection(client, settings, dimension)

    # Delete orphan chunks from previous runs (documents no longer in corpus)
    if orphan_chunk_ids:
        _delete_in_batches(collection, orphan_chunk_ids)

    # Upsert chunks in batches of 100
    batch_size = 100
    total = len(chunks)
    for i in range(0, total, batch_size):
        sub_chunks = chunks[i: i + batch_size]
        sub_vectors = vectors[i: i + batch_size]
        ids = [c.id for c in sub_chunks]
        documents = [c.text for c in sub_chunks]
        metadatas = [c.metadata() for c in sub_chunks]
        try:
            collection.upsert(
                ids=ids,
                embeddings=sub_vectors,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception:
            collection.add(
                ids=ids,
                embeddings=sub_vectors,
                documents=documents,
                metadatas=metadatas,
            )

    return total


def _delete_in_batches(collection: Any, chunk_ids: list[str]) -> None:
    """Delete chunks by ID in batches of 100."""
    batch_size = 100
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i: i + batch_size]
        try:
            collection.delete(ids=batch)
        except Exception:
            pass
