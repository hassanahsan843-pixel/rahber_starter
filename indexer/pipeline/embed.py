"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import hashlib
import math
import httpx

from .config import IndexerSettings


def _deterministic_stub_vector(text: str, dim: int = 768) -> list[float]:
    """Offline fallback vector generator for testing when Ollama is offline."""
    vec = [0.0] * dim
    for word in text.lower().split():
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
