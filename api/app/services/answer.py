"""Turns a question into an AskResponse. This is where the query pipeline lives.

Executes the complete 8-step production pipeline:
1. Validate input length (3 to 500 chars)
2. Semantic cache lookup with similarity threshold
3. Retrieve top-k chunks with metadata filtering on superseded documents
4. Evaluate relevance floor (honest abstention when score < relevance_floor)
5. Generate schema-constrained Answer via pluggable LLM provider
6. Strictly validate citations against retrieved passages (strip hallucinations)
7. Update semantic cache (with strict never-cache guardrails)
8. Return structured AskResponse with request tracing and metrics
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
from .cache import SemanticCache, get_cache

log = logging.getLogger("rahbar.answer")


# D4 - Guardrail: injection patterns to detect in both query and retrieved text
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


# Evidence check: before asking the LLM, at least one salient question term must
# occur in the retrieved text. Embedding similarity alone lets loosely related
# passages through (e.g. an unknown programme matched to attendance rules at ~0.58),
# and a small model will then answer from them. Generic words that occur in every
# document ("university", "student", ...) do not count as evidence.
_STOPWORDS = frozenset("""
a an and are as at be been by can could do does did for from had has have how i if in into is
it its may me might must my no not of on or our shall should so than that the their them then
there these they this to was we were what when where which while who whom why will with would
you your about after any all also am before being both each few get got more most much need
needs other over same some such too under until up very via ever many per
""".split())
_GENERIC_TERMS = frozenset({"university", "student", "students", "fast", "nuces", "campus",
                            "rule", "rules", "policy"})
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+|\d+")


def _stem(word: str) -> str:
    # 5-letter prefix: "eligibility"/"eligible", "examination"/"exam" -> comparable stems
    return word[:5] if len(word) >= 5 else word


def _salient_terms(text: str) -> set[str]:
    terms = set()
    for raw in _WORD_RE.findall(text):
        w = raw.lower().removesuffix("'s").strip("'")
        if w in _STOPWORDS or w in _GENERIC_TERMS or (len(w) < 3 and not w.isdigit()):
            continue
        terms.add(_stem(w))
    return terms


def _has_evidence_overlap(question: str, passages: list[Passage]) -> bool:
    """True when some salient question term appears in the retrieved passages.
    Questions with no salient terms are not blocked."""
    wanted = _salient_terms(question)
    if not wanted:
        return True
    found = {_stem(w.lower().removesuffix("'s").strip("'"))
             for p in passages for w in _WORD_RE.findall(p.text)}
    return bool(wanted & found)


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
    use_cache: bool = True,
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
        log.warning(
            "prompt injection detected in query",
            extra={"patterns": query_injection},
        )

    # Step 2: Semantic cache check
    cache_start = time.perf_counter()
    query_vector: list[float] = []
    cache: SemanticCache | None = None
    cache_generation: int | None = None
    cache_hit = False

    _embed_fn = getattr(retriever, "_embed_query", None)
    if _embed_fn is not None and use_cache:
        cache = get_cache(
            threshold=settings.cache_threshold,
            redis_host=settings.redis_host,
            redis_port=settings.redis_port,
            allow_memory_fallback=settings.retriever != "chroma",
            namespace=settings.collection,
        )
        try:
            query_vector = _embed_fn(cleaned_q)
            cached_resp = cache.lookup(query_vector, is_followup=is_followup)
            if cached_resp is not None:
                CACHE_HITS.labels(result="hit").inc()
                cache_hit = True
                log.info(
                    "serving response from semantic cache",
                    extra={"request_id": rid},
                )
                # The stored entry carries the original request's id; echo this one.
                hit_response = cached_resp.model_copy(update={"request_id": rid})
                # D1: cache hits are traced too, so D3's cache-hit rate is real.
                _write_trace(
                    rid,
                    cleaned_q,
                    settings,
                    [],
                    hit_response.retrieved,
                    "",
                    hit_response,
                    guardrail_actions,
                    0.0,
                    0.0,
                    (time.perf_counter() - cache_start) * 1000,
                    True,
                    provider.name,
                    pipeline_start,
                )
                return hit_response

            cache_generation = cache.current_generation()
            CACHE_HITS.labels(result="miss").inc()
        except Exception as exc:
            log.warning("semantic cache lookup failed: %s", exc)

    cache_ms = (time.perf_counter() - cache_start) * 1000

    # Step 3: Retrieve top-k passages from vector store
    retrieval_start = time.perf_counter()
    try:
        passages = retriever.retrieve(cleaned_q, k=settings.top_k)
    except Exception as exc:
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        RETRIEVAL_LATENCY.observe(retrieval_ms / 1000)
        FALLBACKS.inc()
        log.error(
            "retrieval failed; returning safe fallback",
            extra={"request_id": rid, "error": str(exc)},
        )

        fallback_response = AskResponse(
            request_id=rid,
            answer="I could not retrieve the university documents right now. Please try again shortly.",
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=[],
            provider=provider.name,
        )

        _write_trace(
            rid,
            cleaned_q,
            settings,
            [],
            [],
            "",
            fallback_response,
            guardrail_actions,
            retrieval_ms,
            0.0,
            cache_ms,
            cache_hit,
            provider.name,
            pipeline_start,
        )
        return fallback_response

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    RETRIEVAL_LATENCY.observe(retrieval_ms / 1000)

    # D4: Check retrieved text for injected instructions
    for p in passages:
        retrieved_injection = _detect_injection(p.text)
        if retrieved_injection:
            guardrail_actions.extend(
                [f"retrieved_injection:{action}" for action in retrieved_injection]
            )
            log.warning(
                "injection pattern found in retrieved chunk",
                extra={
                    "chunk_id": p.chunk_id,
                    "patterns": retrieved_injection,
                },
            )

    retrieved_refs = [
        RetrievedRef(
            chunk_id=p.chunk_id,
            source=p.source,
            section=p.section,
            page=p.page,
            score=round(p.score, 4),
            text=p.text,
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

    # The stub retriever returns the same placeholder passages for every question by
    # design (A02 week-1 mode), so the evidence check applies to real retrieval only.
    evidence_ok = bool(passages) and (
        settings.retriever == "stub" or _has_evidence_overlap(cleaned_q, passages)
    )
    if not passages or max_score < settings.relevance_floor or not evidence_ok:
        if not passages:
            reason = "no_passages"
        elif max_score < settings.relevance_floor:
            reason = "relevance_floor"
        else:
            reason = "no_evidence_overlap"
        ABSTENTIONS.labels(reason=reason).inc()
        log.info(
            "honest abstention triggered",
            extra={"max_score": round(max_score, 4), "reason": reason},
        )

        abstain_response = AskResponse(
            request_id=rid,
            answer="The university documents in the corpus do not contain information to answer this question.",
            citations=[],
            answered_from_context=False,
            confidence="low",
            retrieved=retrieved_refs,
            provider=provider.name,
        )

        _write_trace(
            rid,
            cleaned_q,
            settings,
            passages,
            retrieved_refs,
            "",
            abstain_response,
            guardrail_actions,
            retrieval_ms,
            0.0,
            cache_ms,
            cache_hit,
            provider.name,
            pipeline_start,
        )
        return abstain_response

    # Step 5: Generate schema-constrained answer
    gen_start = time.perf_counter()
    answer = provider.generate(cleaned_q, passages)
    generation_ms = (time.perf_counter() - gen_start) * 1000
    GENERATION_LATENCY.observe(generation_ms / 1000)

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

    # If the model claimed context but provided no valid citations,
    # mark answered_from_context False
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

    # Step 8: Log observability line and write trace
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

    # The exact prompt the provider sent (system + user message), so the trace
    # tool can reconstruct it and the hash identifies it precisely.
    prompt_text = _exact_prompt(provider, cleaned_q, passages)
    _write_trace(
        rid,
        cleaned_q,
        settings,
        passages,
        retrieved_refs,
        prompt_text,
        response,
        guardrail_actions,
        retrieval_ms,
        generation_ms,
        cache_ms,
        cache_hit,
        provider.name,
        pipeline_start,
    )

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

        meta = {p.chunk_id: p for p in passages}
        retrieved_dicts = [
            {
                "chunk_id": r.chunk_id,
                "doc_id": meta[r.chunk_id].doc_id if r.chunk_id in meta else "",
                "doc_version": meta[r.chunk_id].doc_version if r.chunk_id in meta else "",
                "superseded": meta[r.chunk_id].superseded if r.chunk_id in meta else False,
                "source": r.source,
                "section": r.section,
                "page": r.page,
                "score": r.score,
            }
            for r in retrieved_refs
        ]
        filters: dict[str, Any] = {}
        if getattr(settings, "retriever", "") == "chroma":
            from ..retrieval.chroma import superseded_filter

            filters = {"where": superseded_filter(query)}

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
            filters_applied=filters,
            final_order=[r.chunk_id for r in retrieved_refs],
        )
    except Exception as exc:
        log.warning("trace write failed: %s", exc)


def _exact_prompt(provider: LLMProvider, question: str, passages: list[Passage]) -> str:
    """System prompt + user message exactly as the provider builds them (first attempt)."""
    import sys

    module = sys.modules.get(type(provider).__module__)
    system = str(getattr(module, "SYSTEM_PROMPT", "") or "")
    builder = getattr(module, "_build_prompt", None)
    user = builder(question, passages) if callable(builder) else question
    return f"{system}\n\n{user}" if system else str(user)
