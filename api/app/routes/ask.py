"""POST /api/ask. The Ingress sends /api/* here unchanged, so the prefix stays."""

from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, HTTPException, Request

from ..schemas import AskRequest, AskResponse
from ..services.answer import answer_question

router = APIRouter(prefix="/api")
log = logging.getLogger("rahbar.ask")


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    # `Cache-Control: no-cache` (RFC 9111) bypasses the semantic cache for this one
    # request: no lookup, no write. Used by the B4 stochasticity experiment so repeated
    # runs measure the model, not cached answers. Caching stays on for everyone else.
    no_cache = "no-cache" in request.headers.get("cache-control", "").lower()
    return answer_question(body.question, state.retriever, state.provider, state.settings,
                           use_cache=not no_cache)


@router.post("/cache/invalidate")
def invalidate_cache(request: Request) -> dict[str, str]:
    """Invalidate semantic cache across the cluster on re-indexing (section 3.3)."""
    expected_token = request.app.state.settings.cache_invalidation_token
    provided_token = request.headers.get("X-Cache-Invalidation-Token", "")

    # Compare bytes: hmac.compare_digest raises TypeError on non-ASCII str input,
    # which would turn a bad header into a 500 instead of a 403.
    if not expected_token or not hmac.compare_digest(
        expected_token.encode("utf-8"), provided_token.encode("utf-8")
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid cache invalidation token",
        )

    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        try:
            cache.invalidate()
        except Exception as exc:
            log.error("cache invalidation failed: %s", exc)
            # 503 tells the indexer to retry; it never reports success here.
            raise HTTPException(
                status_code=503,
                detail="Cache invalidation could not be confirmed",
            ) from exc

    return {"status": "invalidated"}
