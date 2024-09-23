"""Request-size limit, enforced before the body is buffered."""

from __future__ import annotations

from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_BODY_BYTES = 8 * 1024 * 1024


def _too_large() -> HTTPException:
    return HTTPException(413, f"request body exceeds {MAX_BODY_BYTES // (1024 * 1024)} MB")


class BodyLimitMiddleware:
    """Reject request bodies over ``MAX_BODY_BYTES`` with 413.

    A declared ``Content-Length`` is refused up front; chunked uploads are counted as they stream
    in, so a client cannot make the server spool an unbounded file to disk.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            error = _too_large()
            await JSONResponse({"detail": error.detail}, status_code=413)(scope, receive, send)
            return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise _too_large()
            return message

        await self.app(scope, counting_receive, send)
