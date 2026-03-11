"""
Nexus — ASGI middleware components.

RequestIDMiddleware:   Attaches a unique request ID to every request for log
                       correlation across the multi-agent pipeline.

SecurityHeadersMiddleware: Adds standard HTTP security headers to every response.
"""
from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attach a request ID to every incoming request.

    Sources the ID from the ``X-Request-ID`` request header when present,
    otherwise generates a 16-character hex UUID. The ID is:
    - stored on ``request.state.request_id``
    - bound to structlog's context-var store so every log line emitted during
      the request automatically includes it
    - echoed back in the ``X-Request-ID`` response header
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add standard browser security headers to every HTTP response.

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: no-referrer
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
