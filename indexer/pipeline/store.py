"""Stage 5: Store. The only code in the whole system that writes to Chroma."""

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


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count.
    
    The indexer is the ONLY writer to the Chroma collection. The API only reads.
    Collection metadata carries:
      - embed_model: settings.embed_model
      - dimension: length of embedding vector
    """
    if not chunks:
        return 0

    client = get_client(settings)
    dimension = len(vectors[0]) if vectors else 768

    metadata = {
        "embed_model": settings.embed_model,
        "dimension": dimension,
        "hnsw:space": "cosine",
    }

    # Delete existing collection if present to guarantee clean metadata & contents
    try:
        client.delete_collection(settings.collection)
    except Exception:
        pass

    # Get or create collection with embedding metadata pinned
    collection = client.get_or_create_collection(
        name=settings.collection,
        metadata=metadata,
    )

    # Batch upsert chunks into Chroma
    batch_size = 100
    total = len(chunks)

    for i in range(0, total, batch_size):
        sub_chunks = chunks[i : i + batch_size]
        sub_vectors = vectors[i : i + batch_size]

        ids = [c.id for c in sub_chunks]
        documents = [c.text for c in sub_chunks]
        metadatas = [c.metadata() for c in sub_chunks]

        collection.upsert(
            ids=ids,
            embeddings=sub_vectors,
            documents=documents,
            metadatas=metadatas,
        )

    return total
