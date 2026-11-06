"""Semantic cache layer (section 3.3).

Provides semantic cache lookup with cosine similarity thresholding,
never-cache enforcement for follow-ups and abstentions, and cache invalidation.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from ..schemas import AskResponse

log = logging.getLogger("rahbar.cache")


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


class SemanticCache:
    """Semantic Cache with similarity thresholding and strict never-cache guardrails."""

    def __init__(self, threshold: float = 0.88, redis_host: str = "redis", redis_port: int = 6379) -> None:
        self.threshold = threshold
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.generation_id = 1
        self._in_memory_store: list[dict[str, Any]] = []
        self._redis_client = None

        try:
            import redis
            self._redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                socket_connect_timeout=2.0,
                decode_responses=True,
            )
            self._redis_client.ping()
        except Exception:
            self._redis_client = None

    def ping(self) -> bool:
        if self._redis_client is not None:
            try:
                return bool(self._redis_client.ping())
            except Exception:
                return False
        return True

    def lookup(self, query_vector: list[float], is_followup: bool = False) -> AskResponse | None:
        """Lookup query in cache. Never check or return cache for follow-up questions."""
        if is_followup or not query_vector:
            return None

        best_score = 0.0
        best_hit: dict[str, Any] | None = None

        # Check in-memory store (or Redis entries)
        for item in self._in_memory_store:
            stored_vector = item.get("vector")
            if not stored_vector:
                continue
            sim = _cosine_similarity(query_vector, stored_vector)
            if sim > best_score:
                best_score = sim
                best_hit = item

        # Check threshold
        if best_hit and best_score >= self.threshold:
            log.info("semantic cache hit", extra={"similarity": round(best_score, 4), "threshold": self.threshold})
            resp_dict = best_hit["response"]
            return AskResponse.model_validate(resp_dict)

        return None

    def store(
        self,
        query: str,
        query_vector: list[float],
        response: AskResponse,
        is_followup: bool = False,
    ) -> None:
        """Store turn in cache.
        
        Strict never-cache rules:
        - Never cache follow-up questions
        - Never cache abstentions (answered_from_context == False)
        - Never cache low confidence or fallback responses
        """
        if is_followup:
            return
        if not response.answered_from_context:
            return
        if response.confidence == "low":
            return

        entry = {
            "query": query,
            "vector": query_vector,
            "response": response.model_dump(),
            "generation_id": self.generation_id,
        }
        self._in_memory_store.append(entry)

        if self._redis_client:
            try:
                key = f"cache:v{self.generation_id}:{abs(hash(query))}"
                self._redis_client.set(key, json.dumps(entry), ex=86400)
            except Exception as exc:
                log.warning("redis cache write error", extra={"error": str(exc)})

    def invalidate(self) -> None:
        """Invalidate cache on re-index.
        
        When the indexing Job finishes, expire all old cache entries
        so the system never serves obsolete answers from prior indexes.
        """
        self.generation_id += 1
        self._in_memory_store.clear()
        if self._redis_client:
            try:
                self._redis_client.flushdb()
            except Exception as exc:
                log.warning("redis flushdb error", extra={"error": str(exc)})
        log.info("semantic cache invalidated", extra={"generation_id": self.generation_id})


_CACHE_INSTANCE: SemanticCache | None = None


def get_cache(threshold: float = 0.88, redis_host: str = "redis", redis_port: int = 6379) -> SemanticCache:
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        _CACHE_INSTANCE = SemanticCache(threshold=threshold, redis_host=redis_host, redis_port=redis_port)
    return _CACHE_INSTANCE
