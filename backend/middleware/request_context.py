"""
request_context.py — 请求级上下文日志 (P2-8)
================================================================================
P2-8 改造:
- 新增 RequestContext (ContextVar-based)，保存 request_id / tenant_id / session_id
- RequestContextMiddleware 在每个 HTTP 请求入口生成/读取 request_id
- 注入 logging Filter，自动把 ContextVar 写到每条 log record

用法:
    from backend.middleware.request_context import (
        current_request_id, current_tenant_id, current_session_id,
    )
    logger.info("处理查询", extra={"tenant": current_tenant_id()})
"""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

# P2-8: ContextVar 存请求级上下文，跨 async 任务继承。
# 默认 "-" 表示无请求上下文（如启动期 / 后台任务）。
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="-")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


def current_request_id() -> str:
    return _request_id_var.get()


def current_tenant_id() -> str:
    return _tenant_id_var.get()


def current_session_id() -> str:
    return _session_id_var.get()


class RequestContext:
    """
    请求上下文管理器 (contextmanager)

    用法:
        with RequestContext(request_id="abc", tenant_id="t1"):
            logger.info("...")  # 自动带 request_id / tenant_id
    """

    def __init__(
        self,
        request_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
    ):
        self._request_id = request_id or str(uuid.uuid4())[:12]
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._request_id_token = None
        self._tenant_id_token = None
        self._session_id_token = None

    def __enter__(self):
        # 保存旧值 (供 __exit__ 还原)
        self._request_id_token = _request_id_var.set(self._request_id)
        self._tenant_id_token = None
        self._session_id_token = None
        if self._tenant_id is not None:
            self._tenant_id_token = _tenant_id_var.set(self._tenant_id)
        if self._session_id is not None:
            self._session_id_token = _session_id_var.set(self._session_id)
        return self

    def __exit__(self, *exc):
        # 还原旧值。Token.old_value 在无前置值时是 Token.MISSING sentinel，
        # 此时把 var 设回它的 default ('-')。
        from contextvars import Token as _Token  # 局部 import 避免顶层依赖

        for token, var, default in (
            (self._request_id_token, _request_id_var, "-"),
            (self._tenant_id_token, _tenant_id_var, "-"),
            (self._session_id_token, _session_id_var, "-"),
        ):
            if token is None:
                continue
            if token.old_value is _Token.MISSING:
                var.set(default)
            else:
                var.set(token.old_value)


class RequestContextFilter(logging.Filter):
    """
    P2-8: logging Filter — 把 ContextVar 注入每条 log record。

    安装方法 (在 _setup_logging 里):
        for h in handlers:
            h.addFilter(RequestContextFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 不覆盖已有字段（应用层可显式指定）
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_var.get()
        if not hasattr(record, "tenant_id"):
            record.tenant_id = _tenant_id_var.get()
        if not hasattr(record, "session_id"):
            record.session_id = _session_id_var.get()
        return True
