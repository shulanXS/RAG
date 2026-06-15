"""
request_context_middleware.py — ASGI 中间件 (P2-8)
================================================================================
P2-8: 在每个 HTTP 请求入口构造 RequestContext scope，
供下游 middleware / handler / 子任务继承。

Headers:
- 请求可携带 X-Request-Id (网关 / 客户端透传)，否则生成 12 字符 UUID
- X-Tenant-Id 透传到 ContextVar (rate limiter / orchestrator 都可读)
- X-Session-Id 透传 (chat history 查询)
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.middleware.request_context import RequestContext

logger = logging.getLogger(__name__)

_HEADER_REQUEST_ID = "x-request-id"
_HEADER_TENANT_ID = "x-tenant-id"
_HEADER_SESSION_ID = "x-session-id"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """P2-8: 入口中间件 — 建立 request scope 的 ContextVar 上下文"""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(_HEADER_REQUEST_ID) or self._gen_request_id()
        tenant = request.headers.get(_HEADER_TENANT_ID)
        session = request.headers.get(_HEADER_SESSION_ID)

        with RequestContext(request_id=rid, tenant_id=tenant, session_id=session):
            response = await call_next(request)
            # 回写 request_id 到响应头 (便于客户端关联)
            response.headers[_HEADER_REQUEST_ID] = rid
            return response

    @staticmethod
    def _gen_request_id() -> str:
        import uuid
        return uuid.uuid4().hex[:12]
