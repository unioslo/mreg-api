"""Context variables for tracking request metadata."""

from __future__ import annotations

from contextvars import ContextVar

last_request_url: ContextVar[str | None] = ContextVar("last_request_url", default=None)
last_request_method: ContextVar[str | None] = ContextVar("last_request_method", default=None)
