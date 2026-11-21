"""Turns a question into an AskResponse. This is where the query pipeline lives.

Executes the complete 8-step production pipeline:
1. Validate input length (3 to 500 chars)
2. Semantic cache lookup with similarity threshold
3. Retrieve top-k chunks with metadata filtering on superseded documents
4. Evaluate relevance floor (honest abstention when score < relevance_floor)
5. Generate schema-constrained Answer via pluggable LLM provider
6. Strictly validate citations against retrieved passages (strip hallucinations)
7. Update semantic cache (with strict never-cache guardrails)
8. Return structured AskResponse with request tracing and metrics (D1: write trace)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..routes.metrics import (
    ABSTENTIONS,
    CACHE_HITS,
    FALLBACKS,
    GENERATION_LATENCY,
    RETRIEVAL_LATENCY,
    TOKENS,
)
from ..schemas import AskResponse, Citation, Passage, RetrievedRef
from .cache import get_cache

log = logging.getLogger("rahbar.answer")

# D4 – Guardrail: injection patterns to detect in both query and retrieved text
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(system\s+)?prompts?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+(administrator|admin|root)\s+mode", re.IGNORECASE),
    re.compile(r"output\s*:\s*['\"]?system[_\s]compromised", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?system\s+instructions", re.IGNORECASE),
]


def _detect_injection(text: str) -> list[str]:
    """Return list of triggered guardrail pattern names."""
    triggered = []
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            triggered.append(f"injection_pattern:{pat.pattern[:40]}")
    return triggered


def _is_valid_citation(citation: Citation, passages: list[Passage]) -> bool:
    """Every citation must match a chunk actually retrieved for that request.
    
    A citation your retriever never returned is fabricated (-10 penalty).
    """
    for p in passages:
        # Match by source and section (case-insensitive substring)
        if citation.source.lower() in p.source.lower() or p.source.lower() in citation.source.lower():
            if citation.section.lower() in p.section.lower() or p.section.lower() in citation.section.lower():
                return True
    return False


def answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
    is_followup: bool = False,
) -> AskResponse:
    rid = request_id_var.get()
    pipeline_start = time.perf_counter()

    # Step 1: Input length validation
    cleaned_q = question.strip()
    if len(cleaned_q) < 3 or len(cleaned_q) > 500:
        ABSTENTIONS.labels(reason="input_length").inc()
        return AskResponse(
            request_id=rid,
            answer="Question must be between 3 and 500 characters.",
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[],
            provider=provider.name,
        )

    # D4: Check query for prompt-injection attempts
    guardrail_actions: list[str] = []
    query_injection = _detect_injection(cleaned_q)
    if query_injection:
        guardrail_actions.extend(query_injection)
        log.warning("prompt injection detected in query", extra={"patterns": query_injection})

    # Step 2: Semantic cache check
    cache_start = time.perf_counter()
    cache = get_cache(
        threshold=settings.cache_threshold,
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
    )
    query_vector: list[float] = []
    cache_hit = False
    _embed_fn = getattr(retriever, "_embed_query", None)
    if _embed_fn is not None:
        try:
            query_vector = _embed_fn(cleaned_q)
            cached_resp = cache.lookup(query_vector, is_followup=is_followup)
            if cached_resp is not None:
                CACHE_HITS.labels(result="hit").inc()
                cache_hit = True
                log.info("serving response from semantic cache", extra={"request_id": rid})
                return cached_resp
            CACHE_HITS.labels(result="miss").inc()
        except Exception as exc:
            log.warning("semantic cache lookup failed: %s", exc)
    cache_ms = (time.perf_counter() - cache_start) * 1000

    # Step 3: Retrieve top-k passages
    retrieval_start = time.perf_counter()
    passages = retriever.retrieve(cleaned_q, k=settings.top_k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    RETRIEVAL_LATENCY.observe(retrieval_ms / 1000)

    # D4: Check retrieved text for injected instructions
    for p in passages:
        retrieved_injection = _detect_injection(p.text)
        if retrieved_injection:
            guardrail_actions.extend([f"retrieved_injection:{a}" for a in retrieved_injection])
            log.warning("injection pattern found in retrieved chunk", extra={
                "chunk_id": p.chunk_id, "patterns": retrieved_injection})

    retrieved_refs = [
        RetrievedRef(
            chunk_id=p.chunk_id,
            source=p.source,
            section=p.section,
            page=p.page,
            score=round(p.score, 4),
        )
        for p in passages
    ]

    # Step 4: Relevance floor check and honest abstention
    max_score = max((p.score for p in passages), default=0.0)
    log.info(
        "retrieval evaluated",
        extra={
            "request_id": rid,
            "top_k": settings.top_k,
            "chunk_ids": [p.chunk_id for p in passages],
            "scores": [round(p.score, 4) for p in passages],
            "max_score": round(max_score, 4),
            "relevance_floor": settings.relevance_floor,
        },
    )

    if not passages or max_score < settings.relevance_floor:
        reason = "no_passages" if not passages else "relevance_floor"
        ABSTENTIONS.labels(reason=reason).inc()
        log.info("honest abstention triggered", extra={"max_score": round(max_score, 4)})
        abstain_resp = AskResponse(
            request_id=rid,
            answer="The university documents in the corpus do not contain information to answer this question.",
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=retrieved_refs,
            provider=provider.name,
        )
        _write_trace(rid, cleaned_q, settings, passages, retrieved_refs, "",
                     abstain_resp, guardrail_actions, retrieval_ms, 0.0, cache_ms,
                     cache_hit, provider.name, pipeline_start)
        return abstain_resp

    # Step 5: Generate schema-constrained answer
    gen_start = time.perf_counter()
    answer = provider.generate(cleaned_q, passages)
    generation_ms = (time.perf_counter() - gen_start) * 1000
    GENERATION_LATENCY.observe(generation_ms / 1000)

    prompt_tokens = len(cleaned_q.split()) + sum(len(p.text.split()) for p in passages)
    completion_tokens = len(answer.answer.split()) + len(answer.reasoning.split())
    TOKENS.labels(kind="prompt").inc(prompt_tokens)
    TOKENS.labels(kind="completion").inc(completion_tokens)

    if answer.confidence == "low" and not answer.citations and not answer.answered_from_context:
        FALLBACKS.inc()

    # Step 6: Validate citations
    valid_citations = [
        cit for cit in answer.citations if _is_valid_citation(cit, passages)
    ]
    if len(valid_citations) < len(answer.citations):
        log.warning("dropped hallucinated citations", extra={
            "original": len(answer.citations), "valid": len(valid_citations)})

    answered_context = answer.answered_from_context
    if answered_context and not valid_citations:
        answered_context = False

    response = AskResponse(
        request_id=rid,
        answer=answer.answer,
        citations=valid_citations,
        answered_from_context=answered_context,
        confidence=answer.confidence,
        retrieved=retrieved_refs,
        provider=provider.name,
    )

    # Step 7: Cache
    if query_vector and answered_context:
        cache.store(cleaned_q, query_vector, response, is_followup=is_followup)

    # Step 8: Log + write trace
    log.info(
        "answered",
        extra={
            "request_id": rid,
            "retriever": retriever.name,
            "provider": provider.name,
            "top_k": settings.top_k,
            "chunk_ids": [p.chunk_id for p in passages],
            "scores": [round(p.score, 4) for p in passages],
            "answered_from_context": response.answered_from_context,
            "citations_count": len(response.citations),
            "guardrail_actions": guardrail_actions,
        },
    )

    # Build prompt text for hash (approximate)
    prompt_text = cleaned_q + " ".join(p.text[:100] for p in passages)
    _write_trace(rid, cleaned_q, settings, passages, retrieved_refs, prompt_text,
                 response, guardrail_actions, retrieval_ms, generation_ms, cache_ms,
                 cache_hit, provider.name, pipeline_start)

    return response


def _write_trace(
    rid: str,
    query: str,
    settings: Any,
    passages: list[Passage],
    retrieved_refs: list[RetrievedRef],
    prompt_text: str,
    response: AskResponse,
    guardrail_actions: list[str],
    retrieval_ms: float,
    generation_ms: float,
    cache_ms: float,
    cache_hit: bool,
    provider_name: str,
    pipeline_start: float,
) -> None:
    try:
        from .trace import write_trace
        retrieved_dicts = [
            {
                "chunk_id": r.chunk_id,
                "source": r.source,
                "section": r.section,
                "page": r.page,
                "score": r.score,
            }
            for r in retrieved_refs
        ]
        citations_dicts = [
            {"source": c.source, "section": c.section, "page": c.page}
            for c in response.citations
        ]
        write_trace(
            trace_id=rid,
            query=query,
            settings=settings,
            retrieved=retrieved_dicts,
            prompt=prompt_text,
            response_text=response.answer,
            citations=citations_dicts,
            answered_from_context=response.answered_from_context,
            confidence=response.confidence,
            provider_name=provider_name,
            guardrail_actions=guardrail_actions,
            latency_retrieval_ms=retrieval_ms,
            latency_generation_ms=generation_ms,
            latency_cache_ms=cache_ms,
            cache_hit=cache_hit,
        )
    except Exception as exc:
        log.warning("trace write failed: %s", exc)
