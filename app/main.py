from __future__ import annotations

import time
import uuid

import structlog
from fastapi import FastAPI, Request

from app.api.routes import chat, documents, ui
from app.config import get_settings
from app.infra.observability.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(title="FitScope", description="Career Intelligence Assistant")


@app.middleware("http")
async def bind_request_context(request: Request, call_next):
    """Give every log line emitted while handling a request the same
    request_id, and log exactly one completion event per request -- the
    minimal amount of tracing needed to correlate "this upload" with "this
    chat call" in the logs without pulling in a full APM agent for a
    single-service app this size.
    """
    request_id = str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed", method=request.method, path=request.url.path)
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(ui.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}
