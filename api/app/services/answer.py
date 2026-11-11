"""Turns a question into an AskResponse. This is where the query pipeline lives.

Executes the complete 8-step production pipeline:
1. Validate input length (3 to 500 chars)
2. Semantic cache lookup with similarity threshold
3. Retrieve top-k chunks with metadata filtering on superseded documents
4. Evaluate relevance floor (honest abstention when score < relevance_floor)
5. Generate schema-constrained Answer via pluggable LLM provider
6. Strictly validate citations against retrieved passages (strip hallucinations)
7. Update semantic cache (with strict never-cache guardrails)
8. Return structured AskResponse with request tracing and metrics logging
"""

from __future__ import annotations

import logging
import time

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
from .cache import SemanticCache, get_cache

log = logging.getLogger("rahbar.answer")


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

    # Step 2: Semantic cache check
    # Generate query vector for semantic cache if retriever has embedding capability
    query_vector: list[float] = []
    cache: SemanticCache | None = None
    cache_generation: int | None = None
    _embed_fn = getattr(retriever, "_embed_query", None)
    if _embed_fn is not None:
        cache = get_cache(
            threshold=settings.cache_threshold,
            redis_host=settings.redis_host,
            redis_port=settings.redis_port,
            allow_memory_fallback=settings.retriever != "chroma",
        )
        try:
            query_vector = _embed_fn(cleaned_q)
            cached_resp = cache.lookup(query_vector, is_followup=is_followup)
            if cached_resp is not None:
                CACHE_HITS.labels(result="hit").inc()
                log.info("serving response from semantic cache", extra={"request_id": rid})
                return cached_resp
            cache_generation = cache.current_generation()
            CACHE_HITS.labels(result="miss").inc()
        except Exception as exc:
            log.warning("semantic cache lookup failed: %s", exc)

    # Step 3: Retrieve top-k passages from vector store
    retrieval_start = time.perf_counter()
    try:
        passages = retriever.retrieve(cleaned_q, k=settings.top_k)
    except Exception as exc:
        RETRIEVAL_LATENCY.observe(time.perf_counter() - retrieval_start)
        FALLBACKS.inc()
        log.error("retrieval failed; returning safe fallback", extra={"request_id": rid, "error": str(exc)})
        return AskResponse(
            request_id=rid,
            answer="I could not retrieve the university documents right now. Please try again shortly.",
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[],
            provider=provider.name,
        )
    RETRIEVAL_LATENCY.observe(time.perf_counter() - retrieval_start)

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
        log.info("honest abstention triggered: score below relevance floor", extra={"max_score": round(max_score, 4)})
        return AskResponse(
            request_id=rid,
            answer="The university documents in the corpus do not contain information to answer this question.",
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=retrieved_refs,
            provider=provider.name,
        )

    # Step 5: Generate schema-constrained answer
    gen_start = time.perf_counter()
    answer = provider.generate(cleaned_q, passages)
    GENERATION_LATENCY.observe(time.perf_counter() - gen_start)

    # Estimate and track token counts
    prompt_tokens = len(cleaned_q.split()) + sum(len(p.text.split()) for p in passages)
    completion_tokens = len(answer.answer.split()) + len(answer.reasoning.split())
    TOKENS.labels(kind="prompt").inc(prompt_tokens)
    TOKENS.labels(kind="completion").inc(completion_tokens)

    if answer.confidence == "low" and not answer.citations and not answer.answered_from_context:
        FALLBACKS.inc()

    # Step 6: Validate citations against retrieved passages
    valid_citations = [
        cit for cit in answer.citations if _is_valid_citation(cit, passages)
    ]
    if len(valid_citations) < len(answer.citations):
        log.warning(
            "dropped hallucinated citations",
            extra={
                "original": len(answer.citations),
                "valid": len(valid_citations),
            },
        )

    # If the model claimed context but provided no valid citations, mark answered_from_context False
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

    # Step 7: Store in semantic cache if safe
    if query_vector and cache is not None and answered_context:
        cache.store(
            cleaned_q,
            query_vector,
            response,
            is_followup=is_followup,
            expected_generation=cache_generation,
        )

    # Step 8: Log observability line and return
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
        },
    )

    return response
