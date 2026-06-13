"""
health.py — 健康检查 API 路由
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

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


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    系统健康检查接口。

    检查各组件状态:
    - vector_db: Qdrant 连接
    - cache: Redis 连接
    - embedding: Embedder 初始化
    """
    uptime = time.time() - _start_time
    components: dict[str, ComponentHealth] = {}

    # Qdrant 健康检查
    try:
        import time as t

        start = t.perf_counter()
        from backend.retrieval.vector_retriever import VectorRetriever

        vr = VectorRetriever()
        await vr.collection_exists()
        components["vector_db"] = ComponentHealth(
            status="healthy",
            latency_ms=(t.perf_counter() - start) * 1000,
        )
    except Exception as e:
        components["vector_db"] = ComponentHealth(
            status="degraded", error=str(e)
        )
        logger.warning(f"Vector DB health check failed: {e}")

    # Redis 健康检查
    try:
        import time as t

        start = t.perf_counter()
        from backend.cache import RedisSemanticCache

        cache = RedisSemanticCache(host="localhost", port=6379)
        await cache.client.ping()
        components["cache"] = ComponentHealth(
            status="healthy",
            latency_ms=(t.perf_counter() - start) * 1000,
        )
    except Exception as e:
        components["cache"] = ComponentHealth(status="degraded", error=str(e))

    # Embedder 健康检查
    try:
        import time as t

        start = t.perf_counter()
        from backend.ingestion.embedder import Embedder

        emb = Embedder()
        emb.embed("health check")
        components["embedding"] = ComponentHealth(
            status="healthy",
            latency_ms=(t.perf_counter() - start) * 1000,
        )
    except Exception as e:
        components["embedding"] = ComponentHealth(status="degraded", error=str(e))

    # 汇总状态
    degraded_count = sum(1 for c in components.values() if c.status == "degraded")
    overall = "healthy" if degraded_count == 0 else "degraded"

    return HealthResponse(
        status=overall,
        uptime_seconds=uptime,
        components=components,
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """K8s readiness probe"""
    return {"ready": True}


@router.get("/live")
async def liveness_check() -> dict:
    """K8s liveness probe"""
    return {"alive": True}
