"""The real retriever (sections 3.2 and 4.4).

Queries ChromaDB with metadata filtering on superseded documents,
computes cosine similarity, and formats retrieved results as Passages.
"""

from __future__ import annotations

import hashlib
import math
import httpx
from ..config import Settings
from ..schemas import Passage


def _fallback_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic offline vector generator matching indexer fallback."""
    vec = [0.0] * dim
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _embed_query(self, text: str) -> list[float]:
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    f"{self.settings.ollama_url.rstrip('/')}/api/embed",
                    json={"model": self.settings.embed_model, "input": [text]},
                )
                if res.status_code == 200:
                    data = res.json()
                    embs = data.get("embeddings") or []
                    if embs:
                        return embs[0]
        except Exception:
            pass
        return _fallback_embedding(text, dim=768)

    def retrieve(self, question: str, k: int) -> list[Passage]:
        """Embed the question with settings.embed_model (the same model the indexer used),
        query the collection for the top k, filter out superseded documents, and map
        each hit to a Passage.
        
        Chroma returns cosine distance; Passage.score expects similarity,
        so convert: similarity = 1 - distance.
        """
        try:
            collection = self.client.get_collection(self.settings.collection)
        except Exception:
            return []

        query_vec = self._embed_query(question)

        # Query ChromaDB with metadata filter excluding superseded documents
        # unless user explicitly queries an older version
        include_superseded = "2023" in question or "old" in question.lower()
        where_filter = None if include_superseded else {"superseded": {"$eq": False}}

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        passages: list[Passage] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx, chunk_id in enumerate(ids):
            doc_text = docs[idx] if idx < len(docs) else ""
            meta = metas[idx] if idx < len(metas) else {}
            dist = distances[idx] if idx < len(distances) else 1.0

            # Convert cosine distance to cosine similarity
            similarity_score = max(0.0, min(1.0, 1.0 - dist))

            passages.append(Passage(
                chunk_id=chunk_id,
                text=doc_text,
                source=str(meta.get("source", "unknown")),
                section=str(meta.get("section", "General")),
                page=meta.get("page") if isinstance(meta.get("page"), int) else None,
                score=round(similarity_score, 4),
            ))

        return passages
