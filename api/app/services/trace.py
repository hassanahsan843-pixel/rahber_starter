"""D1 – Query trace record writer (section D1).

Every query generates one JSON-line trace record containing all pipeline fields.
Traces are written to:
  1. Standard output (visible via `kubectl logs`)
  2. TRACE_FILE (persistent storage, default /traces/traces.jsonl)

Trace schema
-----------
trace_id          : UUID hex string
timestamp         : ISO-8601 UTC
query             : original user question
query_rewritten   : rewritten query (if any; null otherwise)
embed_model       : embedding model name
collection        : Chroma collection name
retrieved_chunks  : [{chunk_id, source, section, page, score, doc_id, doc_version}]
retrieval_scores  : [float]
filters_applied   : dict of Chroma metadata filters used
reranked_order    : [chunk_id] after reranking (null if no reranker)
prompt_hash       : sha256 of the prompt sent to the LLM
llm_model         : LLM model name
llm_temperature   : generation temperature
response          : final answer text
citations         : [{source, section, page}]
answered_from_context : bool
confidence        : high | medium | low
guardrail_actions : [str] – e.g. ["injection_detected"]
latency           : {retrieval_ms, generation_ms, cache_lookup_ms, total_ms}
cache_hit         : bool
provider          : LLM provider name
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("rahbar.trace")

TRACE_FILE = os.environ.get("TRACE_FILE", "/traces/traces.jsonl")
_TRACE_ENABLED = os.environ.get("TRACE_ENABLED", "true").lower() not in ("0", "false", "no")


def _ensure_trace_dir() -> Path | None:
    """Create trace directory if it doesn't exist. Return path or None on error."""
    try:
        p = Path(TRACE_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as exc:
        log.warning("cannot create trace dir: %s", exc)
        return None


def new_trace_id() -> str:
    return uuid.uuid4().hex


def write_trace(
    trace_id: str,
    query: str,
    settings: Any,
    retrieved: list[dict],
    prompt: str,
    response_text: str,
    citations: list[dict],
    answered_from_context: bool,
    confidence: str,
    provider_name: str,
    guardrail_actions: list[str],
    latency_retrieval_ms: float,
    latency_generation_ms: float,
    latency_cache_ms: float,
    cache_hit: bool,
    filters_applied: dict | None = None,
    query_rewritten: str | None = None,
    final_order: list[str] | None = None,
) -> None:
    """Write one trace record to stdout and to the persistent trace file.

    `prompt` is the exact text sent to the model (system + user message, "" when
    no model call was made); it is stored in full so scripts/trace_inspect.py can
    reconstruct the context, and `prompt_hash` is its SHA-256.
    """
    if not _TRACE_ENABLED:
        return

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    total_ms = latency_retrieval_ms + latency_generation_ms + latency_cache_ms

    record: dict[str, Any] = {
        "trace_id": trace_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": query,
        "query_rewritten": query_rewritten,
        "embed_model": getattr(settings, "embed_model", "unknown"),
        "collection": getattr(settings, "collection", "unknown"),
        "retrieved_chunks": retrieved,
        "retrieval_scores": [r.get("score") for r in retrieved],
        "filters_applied": filters_applied or {},
        # Final chunk order handed to the model (after the offline lexical rerank when
        # OLLAMA_URL=offline; otherwise the retrieval order, as there is no reranker).
        "reranked_order": final_order,
        "prompt_hash": prompt_hash,
        "prompt": prompt,
        "llm_model": getattr(settings, "llm_model", "unknown"),
        "llm_temperature": getattr(settings, "temperature", 0.2),
        "response": response_text,
        "citations": citations,
        "answered_from_context": answered_from_context,
        "confidence": confidence,
        "guardrail_actions": guardrail_actions,
        "latency": {
            "retrieval_ms": round(latency_retrieval_ms, 1),
            "generation_ms": round(latency_generation_ms, 1),
            "cache_lookup_ms": round(latency_cache_ms, 1),
            "total_ms": round(total_ms, 1),
        },
        "cache_hit": cache_hit,
        "provider": provider_name,
    }

    line = json.dumps(record)

    # 1. Write to stdout (picked up by kubectl logs)
    print(line, flush=True)

    # 2. Append to persistent file
    trace_path = _ensure_trace_dir()
    if trace_path:
        try:
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            log.warning("trace file write error: %s", exc)
