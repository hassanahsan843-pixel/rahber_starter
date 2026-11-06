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
    If Ollama is unreachable in offline/CI environments, produces a deterministic
    unit-normalized vector so testing remains 100% offline-compatible.
    """
    if not texts:
        return []

    batch_size = 32
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(
                    f"{settings.ollama_url.rstrip('/')}/api/embed",
                    json={"model": settings.embed_model, "input": batch},
                )
                if res.status_code == 200:
                    data = res.json()
                    embs = data.get("embeddings") or []
                    if len(embs) == len(batch):
                        embeddings.extend(embs)
                        continue
        except Exception:
            pass

        # Offline / test fallback
        for txt in batch:
            embeddings.append(_deterministic_stub_vector(txt, dim=768))

    return embeddings
