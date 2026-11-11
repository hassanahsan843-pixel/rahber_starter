"""POST /api/ask. The Ingress sends /api/* here unchanged, so the prefix stays."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request, status

from ..schemas import AskRequest, AskResponse
from ..services.answer import answer_question

router = APIRouter(prefix="/api")


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    return answer_question(body.question, state.retriever, state.provider, state.settings)


@router.post("/cache/invalidate")
def invalidate_cache(request: Request) -> dict[str, str]:
    """Invalidate semantic cache across the cluster on re-indexing (section 3.3)."""
    settings = request.app.state.settings
    expected = settings.cache_invalidation_token
    supplied = request.headers.get("X-Cache-Invalidation-Token", "")
    if not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        cache.invalidate()
    return {"status": "invalidated"}
