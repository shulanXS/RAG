"""
health.py — 健康检查 API 路由
================================================================================
认证策略 (P0 修复):
- /health        详细健康检查，JWT 可选 (optional) — K8s probe 不带 token
                  时也能 200；JWT 用于"我想看更详细的 authenticated 视图"
- /health/auth   同 /health 但强制要求 JWT (legacy 兼容)
- /ready         K8s readiness probe，无认证，并发检查所有依赖
- /live          K8s liveness probe，无认证，仅检查进程存活

技术决策:
- /ready 复用 backend.observability.health.HealthChecker，
  通过 asyncio.gather 并发执行依赖检查（Qdrant / Redis / LLM），
  避免 K8s 在依赖未就绪时引入流量。
- 修复 P0 阶段发现的硬伤：之前 /health 强制 require JWT，导致 K8s
  livenessProbe 调它会 401，Pod 永远进不了 Ready 状态。
  现在 /health 走 optional auth（自己解析 Bearer），/ready 和 /live
  都不需要认证（K8s probe 的标准做法）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from backend.config import get_config
from backend.observability.health import HealthChecker, HealthState
from backend.security.auth import (
    decode_access_token,
    require_current_user,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

_start_time = time.time()


class ComponentHealth(BaseModel):
    status: str
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float | None = None
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    authenticated: bool = False
    user: str | None = None


def _build_health_checker() -> HealthChecker:
    cfg = get_config()
    return HealthChecker(
        qdrant_url=cfg.vector_db.url,
        redis_host=cfg.cache.host,
        redis_port=cfg.cache.port,
        llm_api_base=None,
        check_timeout=5.0,
    )


async def _run_component_checks() -> dict[str, ComponentHealth]:
    """
    P1-5: 并发执行依赖检查。

    原实现串行 `await checker.check_qdrant()` + `await checker.check_redis()`，
    总耗时 = t_qdrant + t_redis（K8s probe 默认 10s 一次，单次 2-4s 阻塞 503 决策）。
    改用 asyncio.gather 并发，总耗时 = max(t_qdrant, t_redis)。
    """
    checker = _build_health_checker()
    qdrant_status, redis_status = await asyncio.gather(
        checker.check_qdrant(),
        checker.check_redis(),
    )
    return {
        "vector_db": ComponentHealth(
            status=qdrant_status.status.value,
            latency_ms=qdrant_status.latency_ms,
            error=qdrant_status.error,
        ),
        "cache": ComponentHealth(
            status=redis_status.status.value,
            latency_ms=redis_status.latency_ms,
            error=redis_status.error,
        ),
    }


def _aggregate_status(components: dict[str, ComponentHealth]) -> str:
    degraded = sum(1 for c in components.values() if c.status in ("degraded", "unhealthy"))
    return "healthy" if degraded == 0 else "degraded"


def _try_extract_token(request: Request) -> dict | None:
    """Optional auth: parse Bearer token if present, return None if absent/invalid."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None
    try:
        token = auth_header.split(" ", 1)[1].strip()
        return decode_access_token(token)
    except Exception:
        return None


@router.get("", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    详细系统健康检查接口 (JWT 可选)。

    行为:
    - 无 JWT: 200, authenticated=false
    - 有 JWT: 200, authenticated=true, user 字段填入 sub
    - JWT 过期/无效: 仍然 200 (K8s probe 不带 token 时也能用)
    """
    token_payload = _try_extract_token(request)
    components = await _run_component_checks()
    return HealthResponse(
        status=_aggregate_status(components),
        uptime_seconds=time.time() - _start_time,
        components=components,
        authenticated=bool(token_payload),
        user=token_payload.get("sub") if token_payload else None,
    )


@router.get("/auth", response_model=HealthResponse)
async def health_check_auth(
    token_payload: dict = Depends(require_current_user),
    request: Request = None,
) -> HealthResponse:
    """强制 JWT 认证的详细健康检查 — 保留给"我想看 user 视角的 health"用例。"""
    components = await _run_component_checks()
    return HealthResponse(
        status=_aggregate_status(components),
        uptime_seconds=time.time() - _start_time,
        components=components,
        authenticated=True,
        user=token_payload.get("sub"),
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """
    K8s readiness probe — 并发检查所有依赖，核心依赖失败时返回 503。
    """
    checker = _build_health_checker()
    result = await checker.get_readiness()

    if result.status == HealthState.UNHEALTHY:
        logger.warning(f"Readiness check failed: {result.error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "ready": False,
                "status": result.status.value,
                "details": result.details,
            },
        )

    body: dict[str, Any] = {
        "ready": True,
        "status": result.status.value,
        "latency_ms": result.latency_ms,
        "details": result.details,
    }
    if result.status == HealthState.DEGRADED:
        body["degraded"] = True
    return body


@router.get("/live")
async def liveness_check() -> dict:
    """K8s liveness probe — 仅检查进程存活 (无认证)"""
    return {"alive": True}
