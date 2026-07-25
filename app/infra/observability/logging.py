"""Structured logging setup.

One place to configure structlog so every log line is a single JSON object
(timestamp, level, event, bound context) instead of ad-hoc print()/logging
calls scattered through the codebase. Uses PrintLoggerFactory (writes
straight to stdout) rather than bridging into stdlib logging -- this app had
no existing stdlib logging calls to stay compatible with, so there was
nothing to bridge (uvicorn's own access logs are separate and keep their own
format). A container platform (ECS, Cloud Run, k8s) picks up stdout directly
and ships it to CloudWatch/Cloud Logging/whatever log backend is configured,
so JSON-on-stdout is the deployment target, not an interim step before "real"
observability.
"""
from __future__ import annotations

import logging

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "app") -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)
