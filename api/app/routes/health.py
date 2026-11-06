"""Liveness and readiness (section 4.3). They answer different questions."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Request, Response, status

router = APIRouter()
log = logging.getLogger("rahbar.health")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM.
    
    A slow dependency must never cause a restart storm.
    """
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    """Can it take traffic now? Returns 200 only if all dependencies are verified.
    
    Returns 503, naming the failed dependency, unless Chroma is reachable,
    the collection exists and is non-empty (collection.count() > 0), Redis is reachable,
    and the collection's embedding model matches EMBED_MODEL.
    """
    settings = getattr(request.app.state, "settings", None)
    if not settings:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "error": "settings uninitialized"}

    checks: dict[str, str] = {}
    failed_dependencies: list[str] = []

    # Check Vector Store (Chroma) if not running in stub mode
    if settings.retriever == "chroma":
        try:
            retriever = getattr(request.app.state, "retriever", None)
            if not retriever or not hasattr(retriever, "client"):
                raise RuntimeError("Chroma client not initialized")

            client = retriever.client
            collection = client.get_collection(settings.collection)

            # Check collection is non-empty
            total_items = collection.count()
            if total_items <= 0:
                checks["chroma"] = "collection empty"
                failed_dependencies.append("chroma_empty_collection")
            else:
                checks["chroma"] = f"ok ({total_items} items)"

            # Guardrail: embedding model pin verification
            meta = collection.metadata or {}
            stored_model = meta.get("embed_model")
            if stored_model and stored_model != settings.embed_model:
                msg = f"embed_model mismatch: collection uses '{stored_model}' but EMBED_MODEL is '{settings.embed_model}'"
                log.error(msg)
                checks["embed_model_pin"] = msg
                failed_dependencies.append("embed_model_mismatch")
            else:
                checks["embed_model_pin"] = "ok"

        except Exception as exc:
            msg = f"Chroma unreachable or error: {exc}"
            log.warning(msg)
            checks["chroma"] = msg
            failed_dependencies.append("chroma")
    else:
        checks["retriever"] = "stub"

    # Check Semantic Cache (Redis) if configured
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        try:
            if not cache.ping():
                checks["redis"] = "ping failed"
                failed_dependencies.append("redis")
            else:
                checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"unreachable: {exc}"
            failed_dependencies.append("redis")
    else:
        checks["redis"] = "in-memory / disabled"

    if failed_dependencies:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "failed_dependencies": failed_dependencies,
            "checks": checks,
        }

    return {"status": "ready", "checks": checks}
