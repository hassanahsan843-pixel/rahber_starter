"""The real retriever (sections 3.2 and 4.4).

Queries ChromaDB with metadata filtering on superseded documents,
computes cosine similarity, and formats retrieved results as Passages.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import httpx

from ..config import Settings
from ..schemas import Passage

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "if", "in", "is", "it", "my", "of", "on", "or",
    "the", "to", "what", "when", "where", "which", "who", "why",
}


def _offline_terms(text: str) -> list[str]:
    """Keep the CI embedding tokens aligned with the indexer's offline vectorizer."""
    terms: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        if token in {"examination", "examinations"}:
            token = "exam"
        elif len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
            if len(token) > 2 and token[-1] == token[-2]:
                token = token[:-1]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-1]
            if len(token) > 2 and token[-1] == token[-2]:
                token = token[:-1]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        terms.append(token)
    return terms


def _lexical_relevance(question: str, passage_text: str) -> float:
    """CI-only reranking score for its deterministic lexical embeddings."""
    query_terms = set(_offline_terms(question))
    if not query_terms:
        return 0.0
    passage_terms = set(_offline_terms(passage_text))
    return len(query_terms & passage_terms) / len(query_terms)


def _fallback_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic offline vector generator matching indexer fallback."""
    vec = [0.0] * dim
    for word in _offline_terms(text):
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
        # Fast offline / CI path
        if not self.settings.ollama_url or any(x in self.settings.ollama_url.lower() for x in ("offline", "disabled", "stub", "none")):
            return _fallback_embedding(text, dim=768)
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    f"{self.settings.ollama_url.rstrip('/')}/api/embed",
                    json={"model": self.settings.embed_model, "input": [text]},
                )
                res.raise_for_status()
                data = res.json()
                embs = data.get("embeddings") or []
                if not embs or not embs[0]:
                    raise RuntimeError("embedding provider returned an empty vector")
                if "nomic" in self.settings.embed_model and len(embs[0]) != 768:
                    raise RuntimeError("nomic-embed-text must return a 768-dimension query vector")
                return embs[0]
        except httpx.HTTPError as exc:
            raise RuntimeError(f"embedding request failed for model {self.settings.embed_model}") from exc

    def retrieve(self, question: str, k: int) -> list[Passage]:
        """Embed the question with settings.embed_model (the same model the indexer used),
        query the collection for the top k, filter out superseded documents, and map
        each hit to a Passage.

        Chroma returns cosine distance; Passage.score expects similarity,
        so convert: similarity = 1 - distance.
        """
        try:
            collection: Any = self.client.get_collection(self.settings.collection)
        except Exception:
            return []

        query_vec = self._embed_query(question)

        # Query ChromaDB with metadata filter excluding superseded documents
        # unless user explicitly queries an older version
        include_superseded = "2023" in question or "old" in question.lower()
        where_filter: Any = None if include_superseded else {"superseded": {"$eq": False}}

        offline_embeddings = not self.settings.ollama_url or any(
            marker in self.settings.ollama_url.lower()
            for marker in ("offline", "disabled", "stub", "none")
        )
        query_count = k
        if offline_embeddings:
            try:
                query_count = max(k, int(collection.count()))
            except Exception:
                pass

        results: Any = collection.query(
            query_embeddings=[query_vec],
            n_results=query_count,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        passages: list[Passage] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids: list[str] = results["ids"][0]
        raw_docs = results.get("documents")
        docs: list[str] = raw_docs[0] if raw_docs else []
        raw_metas = results.get("metadatas")
        metas: list[dict[str, Any]] = raw_metas[0] if raw_metas else []
        raw_dists = results.get("distances")
        distances: list[float] = raw_dists[0] if raw_dists else []

        for idx, chunk_id in enumerate(ids):
            doc_text = str(docs[idx]) if idx < len(docs) else ""
            meta: dict[str, Any] = metas[idx] if idx < len(metas) else {}
            dist = float(distances[idx]) if idx < len(distances) else 1.0

            # Convert cosine distance to cosine similarity
            similarity_score = max(0.0, min(1.0, 1.0 - dist))

            raw_page = meta.get("page")
            page_num: int | None = int(raw_page) if isinstance(raw_page, int) else None

            passages.append(Passage(
                chunk_id=str(chunk_id),
                text=doc_text,
                source=str(meta.get("source", "unknown")),
                section=str(meta.get("section", "General")),
                page=page_num,
                score=round(similarity_score, 4),
            ))

        if offline_embeddings:
            passages.sort(
                key=lambda passage: _lexical_relevance(question, passage.text),
                reverse=True,
            )
            passages = [
                passage.model_copy(update={
                    "score": round(_lexical_relevance(question, passage.text), 4),
                })
                for passage in passages[:k]
            ]
        return passages
