"""Prometheus metrics at /metrics (section 4.8).

Exposes request metrics, retrieval latency, generation latency,
cache hit/miss counts, abstentions, safe fallbacks, and token usage.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

router = APIRouter()

REQUESTS = Counter("rahbar_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "rahbar_request_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

RETRIEVAL_LATENCY = Histogram(
    "rahbar_retrieval_seconds",
    "Time spent in vector retrieval stage",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

GENERATION_LATENCY = Histogram(
    "rahbar_generation_seconds",
    "Time spent in LLM answer generation stage",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30),
)

CACHE_HITS = Counter(
    "rahbar_cache_hits_total",
    "Semantic cache hits and misses",
    ["result"],  # "hit" | "miss"
)

ABSTENTIONS = Counter(
    "rahbar_abstentions_total",
    "Count of queries declined / abstained",
    ["reason"],  # "relevance_floor" | "no_passages" | "input_length"
)

FALLBACKS = Counter(
    "rahbar_fallbacks_total",
    "Safe fallbacks triggered on validation failure",
)

TOKENS = Counter(
    "rahbar_tokens_total",
    "Estimated tokens consumed",
    ["kind"],  # "prompt" | "completion"
)


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
