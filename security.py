"""Basic safeguards for the local-first incident copilot.

This is a single-instance API key and rate limiter, not an identity system.
For multi-worker or public deployments use an identity provider and a shared rate-limit store.
"""
from collections import defaultdict, deque
import hmac
import os
import re
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_SENSITIVE_FIELD = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|authorization)\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")


def redact_sensitive(value: str) -> str:
    """Remove common secrets/identifiers before embedding, AI calls or persistence."""
    value = _SENSITIVE_FIELD.sub(lambda m: f"{m.group(1)}=[REDACTED]", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _IPV4.sub("[REDACTED_IP]", value)
    return value


class FixedWindowLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = defaultdict(deque)

    def allow(self, key: tuple[str, str], limit: int, window: int = 60) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            times = self._requests[key]
            while times and times[0] <= now - window:
                times.popleft()
            if len(times) >= limit:
                return False, max(1, int(window - (now - times[0])) + 1)
            times.append(now)
            # Avoid unbounded growth for many distinct visitors over time.
            if len(self._requests) > 4096:
                for old_key in list(self._requests):
                    history = self._requests[old_key]
                    if not history or history[-1] <= now - window:
                        del self._requests[old_key]
            return True, 0

limiter = FixedWindowLimiter()


class ApiSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in ("/", "/health"):
            return await call_next(request)
        configured = os.getenv("COPILOT_API_KEY", "")
        if not configured:
            return JSONResponse(status_code=503, content={"detail": "API authentication is not configured"})
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied, configured):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"},
                                headers={"WWW-Authenticate": "ApiKey"})

        if request.method in ("POST", "PUT", "PATCH") and request.headers.get("content-length"):
            try:
                if int(request.headers["content-length"]) > 32768:
                    return JSONResponse(status_code=413, content={"detail": "Request too large"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid content length"})

        client_ip = request.client.host if request.client else "unknown"
        category = "incident-create" if path == "/incident" and request.method == "POST" else "other"
        limit = 20 if category == "incident-create" else 120
        ok, retry_after = limiter.allow((client_ip, category), limit=limit)
        if not ok:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"},
                                headers={"Retry-After": str(retry_after)})
        return await call_next(request)
