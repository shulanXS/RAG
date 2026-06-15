"""
health.py — Health and readiness endpoints
================================================================================
技术决策记录:
- /health: 只检查进程存活（Kubernetes liveness probe）
- /ready: 检查所有依赖（Qdrant/Redis/LLM API）（Kubernetes readiness probe）
- 使用 async/await 支持 FastAPI 集成
- 每个依赖有独立健康检查函数，支持并发检查
- 超时控制：单次检查超时 5s，总超时 10s
- 降级策略：部分依赖不可用时仍返回 200（graceful degradation）

业务价值:
- Kubernetes probes: 支持容器编排的存活/就绪探测
- 依赖监控: 提前发现下游服务故障
- SLO 报告: 健康检查数据用于 SLO 计算

实现细节:
- HealthStatus dataclass: 统一返回格式
- HealthChecker: 单例模式，封装所有检查逻辑
- async def get_readiness(): FastAPI endpoint handler
- async def get_liveness(): FastAPI endpoint handler
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Health Status Enum
# =============================================================================


class HealthState(str, Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# 2. Health Status Dataclass
# =============================================================================


@dataclass
class HealthStatus:
    """
    健康检查结果

    字段说明:
    - status: 健康状态（healthy/degraded/unhealthy）
    - latency_ms: 检查延迟（毫秒）
    - error: 错误信息（如果有）
    - details: 附加详情（依赖各自的健康状态）
    """
    status: HealthState
    latency_ms: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（用于 JSON 响应）"""
        return {
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            **self.details,
        }

    @property
    def is_healthy(self) -> bool:
        """是否健康（healthy 或 degraded 都视为可用）"""
        return self.status in (HealthState.HEALTHY, HealthState.DEGRADED)


# =============================================================================
# 3. HealthChecker Singleton
# =============================================================================


class HealthChecker:
    """
    健康检查器

    设计模式: Facade Pattern
    - 封装所有依赖的健康检查逻辑
    - 提供统一的接口

    技术决策:
    - 为什么使用 async?
      → 健康检查通常是 I/O 操作（网络请求），async 可以并发执行多个检查
      → 支持 FastAPI 框架集成
    - 为什么每个依赖独立检查?
      → 细粒度的健康状态，支持 graceful degradation
      → 便于告警规则配置（单独告警 Redis 而非整体）
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        llm_api_base: str | None = None,
        check_timeout: float = 5.0,
    ):
        """
        初始化健康检查器

        Args:
            qdrant_url: Qdrant 服务地址
            redis_host: Redis 主机地址
            redis_port: Redis 端口
            llm_api_base: LLM API 基础地址（用于健康检查）
            check_timeout: 单次检查超时时间（秒）
        """
        self._qdrant_url = qdrant_url
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._llm_api_base = llm_api_base
        self._check_timeout = check_timeout

    # -------------------------------------------------------------------------
    # Individual Health Checks
    # -------------------------------------------------------------------------

    async def check_qdrant(self) -> HealthStatus:
        """
        检查 Qdrant 矢量数据库健康状态

        检查方式: GET /readyz

        超时处理: 5s 超时视为不可用
        """
        start = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self._check_timeout) as client:
                response = await client.get(f"{self._qdrant_url}/readyz")

                latency_ms = (time.perf_counter() - start) * 1000

                if response.status_code == 200:
                    return HealthStatus(
                        status=HealthState.HEALTHY,
                        latency_ms=latency_ms,
                        details={"qdrant": "connected"},
                    )
                else:
                    return HealthStatus(
                        status=HealthState.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=f"Qdrant returned status {response.status_code}",
                        details={"qdrant": "error"},
                    )

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"Qdrant 健康检查超时: {self._qdrant_url}")
            return HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=latency_ms,
                error="Timeout connecting to Qdrant",
                details={"qdrant": "timeout"},
            )

        except httpx.ConnectError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"Qdrant 连接失败: {e}")
            return HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=latency_ms,
                error=f"Cannot connect to Qdrant: {str(e)[:100]}",
                details={"qdrant": "connection_error"},
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Qdrant 健康检查异常: {e}")
            return HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(e)[:200],
                details={"qdrant": "unknown_error"},
            )

    async def check_redis(self) -> HealthStatus:
        """
        检查 Redis 缓存健康状态

        检查方式: PING 命令

        超时处理: 5s 超时视为不可用
        """
        start = time.perf_counter()

        try:
            import redis.asyncio as aioredis

            client = aioredis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                socket_timeout=self._check_timeout,
                socket_connect_timeout=self._check_timeout,
            )

            try:
                response = await client.ping()
                latency_ms = (time.perf_counter() - start) * 1000

                if response:
                    await client.aclose()
                    return HealthStatus(
                        status=HealthState.HEALTHY,
                        latency_ms=latency_ms,
                        details={"redis": "connected"},
                    )
                else:
                    await client.aclose()
                    return HealthStatus(
                        status=HealthState.UNHEALTHY,
                        latency_ms=latency_ms,
                        error="Redis PING returned False",
                        details={"redis": "error"},
                    )

            except Exception:
                await client.aclose()
                raise

        except ImportError:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("redis [async] 未安装，跳过 Redis 健康检查")
            return HealthStatus(
                status=HealthState.DEGRADED,
                latency_ms=latency_ms,
                error="redis.asyncio not installed",
                details={"redis": "not_available"},
            )

        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"Redis 连接失败: {e}")
            return HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=latency_ms,
                error=f"Cannot connect to Redis: {str(e)[:100]}",
                details={"redis": "connection_error"},
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Redis 健康检查异常: {e}")
            return HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(e)[:200],
                details={"redis": "unknown_error"},
            )

    # -------------------------------------------------------------------------
    # Aggregate Health Checks
    # -------------------------------------------------------------------------

    async def get_liveness(self) -> HealthStatus:
        """
        Liveness 检查（Kubernetes liveness probe）

        只检查进程是否存活，不检查依赖。

        Returns:
            HealthStatus: 总是返回 healthy（除非进程已崩溃）
        """
        return HealthStatus(
            status=HealthState.HEALTHY,
            latency_ms=0.0,
            details={"process": "alive"},
        )

    async def get_readiness(self) -> HealthStatus:
        """
        Readiness 检查（Kubernetes readiness probe）

        检查核心依赖：Qdrant、Redis

        技术决策:
        - 使用 asyncio.gather 并发执行所有检查
        - 只要有一个核心依赖（Qdrant）可用，就返回 degraded 而非 unhealthy
        - 这是 graceful degradation 策略：部分降级仍可服务
        """
        start = time.perf_counter()

        # 并发执行所有检查
        results = await asyncio.gather(
            self.check_qdrant(),
            self.check_redis(),
            return_exceptions=True,
        )

        qdrant_status = results[0] if not isinstance(results[0], Exception) else None
        redis_status = results[1] if not isinstance(results[1], Exception) else None

        # 解析异常
        if isinstance(results[0], Exception):
            qdrant_status = HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=0.0,
                error=str(results[0])[:200],
                details={"qdrant": "exception"},
            )
        if isinstance(results[1], Exception):
            redis_status = HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=0.0,
                error=str(results[1])[:200],
                details={"redis": "exception"},
            )

        total_latency_ms = (time.perf_counter() - start) * 1000

        # 聚合状态判断
        overall_status = self._aggregate_status(qdrant_status, redis_status)

        return HealthStatus(
            status=overall_status,
            latency_ms=total_latency_ms,
            details={
                "dependencies": {
                    "qdrant": qdrant_status.to_dict() if qdrant_status else None,
                    "redis": redis_status.to_dict() if redis_status else None,
                }
            },
        )

    def _aggregate_status(
        self,
        qdrant: HealthStatus | None,
        redis: HealthStatus | None,
    ) -> HealthState:
        """
        聚合多个依赖的健康状态

        规则:
        - Qdrant 是核心依赖：必须 healthy 或 degraded
        - Redis 可选：可用性降级不影响整体健康
        """
        if qdrant and qdrant.status == HealthState.UNHEALTHY:
            return HealthState.UNHEALTHY

        # 检查是否有任何 healthy 依赖
        all_degraded = all(
            s and s.status in (HealthState.DEGRADED, HealthState.UNHEALTHY)
            for s in [qdrant, redis]
        )
        if all_degraded:
            return HealthState.UNHEALTHY

        # 有 degraded 状态
        has_degraded = any(
            s and s.status == HealthState.DEGRADED
            for s in [qdrant, redis]
        )
        if has_degraded:
            return HealthState.DEGRADED

        return HealthState.HEALTHY


# =============================================================================
# 4. Module-Level Singleton (P3.2: 真单例)
# =============================================================================

_health_checker_instance: HealthChecker | None = None
_health_checker_lock = threading.Lock()


def get_health_checker(
    qdrant_url: str = "http://localhost:6333",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    llm_api_base: str | None = None,
) -> HealthChecker:
    """
    获取 HealthChecker 单例 (P3.2 修复: 之前 lru_cache + 多参数会导致每次调参不同实例)
    真正的单例：第一次调用时确定参数，后续忽略。
    """
    global _health_checker_instance
    if _health_checker_instance is None:
        with _health_checker_lock:
            if _health_checker_instance is None:
                _health_checker_instance = HealthChecker(
                    qdrant_url=qdrant_url,
                    redis_host=redis_host,
                    redis_port=redis_port,
                    llm_api_base=llm_api_base,
                )
    return _health_checker_instance


def reset_health_checker() -> None:
    """
    重置单例（仅供测试使用）。

    用法示例:
        from backend.observability.health import get_health_checker

        checker = get_health_checker(
            qdrant_url="http://qdrant:6333",
            redis_host="redis",
        )

        # FastAPI endpoint
        @app.get("/health")
        async def health():
            return await checker.get_liveness()

        @app.get("/ready")
        async def ready():
            return await checker.get_readiness()
    """
    global _health_checker_instance
    _health_checker_instance = None


# =============================================================================
# 5. FastAPI 端点统一在 backend/api/health.py，模块层不再提供便捷函数
# =============================================================================
