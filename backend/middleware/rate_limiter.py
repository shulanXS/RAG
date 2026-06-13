"""
rate_limiter.py — Redis 令牌桶限流
================================================================================
技术决策记录:
- 基于 Redis 的令牌桶算法，支持 per-tenant 限流
- 支持 tier 分组：free/pro/enterprise
- 滑动窗口计数，避免 burst 问题
- 限流在 API 网关层或 middleware 层处理，不在业务逻辑层
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Literal

logger = logging.getLogger(__name__)


# Redis 依赖检查
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis 未安装，限流不可用。请运行: pip install redis")


@dataclass
class RateLimitTier:
    """限流 Tier 配置"""
    name: str
    requests_per_minute: int
    burst_size: int


DEFAULT_TIERS: dict[str, RateLimitTier] = {
    "free": RateLimitTier(name="free", requests_per_minute=10, burst_size=5),
    "pro": RateLimitTier(name="pro", requests_per_minute=60, burst_size=20),
    "enterprise": RateLimitTier(name="enterprise", requests_per_minute=300, burst_size=100),
}


class RateLimiter:
    """
    Redis 滑动窗口限流器

    技术要点:
    - 使用 Redis ZSET 实现滑动窗口计数器
    - 每个 tenant_id 有独立的限流 key
    - tier 决定请求速率上限
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        tiers: dict[str, RateLimitTier] | None = None,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("需要安装 redis: pip install redis")
        
        self._client = redis.Redis(host=host, port=port, decode_responses=False)
        self._tiers = tiers or DEFAULT_TIERS

    def _get_key(self, tenant_id: str) -> str:
        return f"ratelimit:{tenant_id}"

    async def check_rate_limit(
        self,
        tenant_id: str,
        tier: str = "free",
    ) -> tuple[bool, int, int]:
        """
        检查限流状态

        Args:
            tenant_id: 租户 ID
            tier: 限流 tier

        Returns:
            (allowed, remaining, reset_in_seconds)
            allowed: 是否允许请求
            remaining: 窗口内剩余请求数
            reset_in_seconds: 窗口重置时间
        """
        tier_config = self._tiers.get(tier, self._tiers["free"])
        key = self._get_key(tenant_id)
        now = time.time()
        window_start = now - 60  # 60秒滑动窗口

        pipe = self._client.pipeline()

        # 移除窗口外的记录
        pipe.zremrangebyscore(key, 0, window_start)

        # 获取当前窗口内请求数
        pipe.zcard(key)

        # 添加当前请求
        pipe.zadd(key, {f"{now}": now})

        # 设置过期时间
        pipe.expire(key, 120)

        results = pipe.execute()
        current_count = results[1]  # zcard result

        remaining = max(0, tier_config.requests_per_minute - current_count - 1)
        allowed = current_count < tier_config.requests_per_minute

        # 如果不允许，不要将当前请求计入
        if not allowed:
            self._client.zrem(key, f"{now}")

        logger.debug(
            f"RateLimit: tenant={tenant_id} tier={tier} "
            f"allowed={allowed} remaining={remaining}"
        )

        return allowed, remaining, 60

    def get_window_count(self, tenant_id: str) -> int:
        """获取当前窗口内的请求数"""
        key = self._get_key(tenant_id)
        now = time.time()
        window_start = now - 60
        return self._client.zcount(key, window_start, now)


# 全局限流器实例（延迟初始化）
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(host: str = "localhost", port: int = 6379) -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(host=host, port=port)
    return _rate_limiter


def check_rate_limit(tenant_id: str, tier: str = "free"):
    """
    限流检查装饰器

    用法:
        @check_rate_limit("tenant_123", "pro")
        async def my_handler():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()
            allowed, remaining, _ = await limiter.check_rate_limit(tenant_id, tier)
            if not allowed:
                raise RateLimitExceeded(f"Rate limit exceeded for tenant {tenant_id}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """限流超出异常"""
    pass


class RateLimitMiddleware:
    """
    FastAPI/Starlette 限流中间件

    从请求头中提取 tenant_id，默认 fallback 到 IP。
    将限流结果以响应头返回: X-RateLimit-Remaining, X-RateLimit-Reset
    """

    def __init__(self, app, limiter: RateLimiter | None = None):
        self.app = app
        self._limiter = limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limiter = self._limiter or get_rate_limiter()
        tenant_id = self._extract_tenant_id(scope)
        tier = self._extract_tier(scope)

        try:
            allowed, remaining, reset_in = limiter.check_rate_limit(tenant_id, tier)
        except Exception:
            allowed = True
            remaining = -1
            reset_in = 60

        if not allowed:
            response = [
                b"HTTP/1.1 429 Too Many Requests\r\n",
                b"Content-Type: application/json\r\n",
                b"Retry-After: 60\r\n",
                b"\r\n",
                b'{"detail":"Rate limit exceeded"}',
            ]
            async for chunk in receive:
                pass
            for item in response:
                await send({"type": "http.response.body", "body": item})
            return

        status_sent = False

        async def send_wrapper(message):
            nonlocal status_sent
            if message["type"] == "http.response.start" and not status_sent:
                status_sent = True
                await send({
                    "type": "http.response.start",
                    "status": message["status"],
                    "headers": [
                        *message.get("headers", []),
                        (b"x-ratelimit-remaining", str(remaining).encode()),
                        (b"x-ratelimit-reset", str(reset_in).encode()),
                    ],
                })
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)

    def _extract_tenant_id(self, scope) -> str:
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer "):
            import base64, json
            try:
                payload = auth[7:].split(".")[1]
                padded = payload + "=" * (4 - len(payload) % 4)
                claims = json.loads(base64.urlsafe_b64decode(padded))
                return claims.get("sub", "anonymous")
            except Exception:
                pass
        client = scope.get("client")
        if client:
            return f"ip:{client[0]}"
        return "unknown"

    def _extract_tier(self, scope) -> str:
        headers = dict(scope.get("headers", []))
        tier_header = headers.get(b"x-rate-limit-tier", b"").decode()
        if tier_header in ("free", "pro", "enterprise"):
            return tier_header
        return "free"
