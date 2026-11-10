"""POST /api/ask. The Ingress sends /api/* here unchanged, so the prefix stays."""

from __future__ import annotations

from fastapi import APIRouter, Request

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
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        cache.invalidate()
    return {"status": "invalidated"}
