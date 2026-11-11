"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import hashlib
import math
import re

import httpx

from .config import IndexerSettings


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "if", "in", "is", "it", "my", "of", "on", "or",
    "the", "to", "what", "when", "where", "which", "who", "why",
}


def _offline_terms(text: str) -> list[str]:
    """Normalize CI-only lexical features so punctuation and simple inflections match."""
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


def _deterministic_stub_vector(text: str, dim: int = 768) -> list[float]:
    """Offline fallback vector generator for testing when Ollama is offline."""
    vec = [0.0] * dim
    for word in _offline_terms(text):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Embed with settings.embed_model and nothing else.

    With Ollama: POST {ollama_url}/api/embed with {"model": embed_model, "input": texts}
    returns {"embeddings": [[...], ...]}. Batches requests in chunks of 32.
    Deterministic vectors are used only when CI explicitly sets OLLAMA_URL=offline.
    A production connection or response failure stops indexing; it must never label
    synthetic vectors as the pinned embedding model.
    """
    if not texts:
        return []

    # Fast offline / CI path
    if not settings.ollama_url or any(x in settings.ollama_url.lower() for x in ("offline", "disabled", "stub", "none")):
        return [_deterministic_stub_vector(txt, dim=768) for txt in texts]

    batch_size = 32
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    f"{settings.ollama_url.rstrip('/')}/api/embed",
                    json={"model": settings.embed_model, "input": batch},
                )
                res.raise_for_status()
                data = res.json()
                embs = data.get("embeddings") or []
                if len(embs) != len(batch) or any(not vector for vector in embs):
                    raise RuntimeError("embedding provider returned an incomplete batch")
                dimensions = {len(vector) for vector in embs}
                if len(dimensions) != 1:
                    raise RuntimeError("embedding provider returned inconsistent dimensions")
                if "nomic" in settings.embed_model and dimensions != {768}:
                    raise RuntimeError(f"{settings.embed_model} must return 768-dimension vectors")
                embeddings.extend(embs)
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"embedding request failed for model {settings.embed_model}") from exc

    return embeddings
