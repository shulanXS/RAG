"""
rate_limiter.py — Redis 滑动窗口限流（per-tenant）
================================================================================
P1-2 简化:
- 移除 3-tier 抽象（free/pro/enterprise）：无前端传 X-Rate-Limit-Tier header
- 固定 requests_per_minute=60（从原 pro 档）
- 移除 check_rate_limit 装饰器（无消费方，只用 RateLimitMiddleware）
- 保留 RateLimitExceeded 异常 + RateLimitMiddleware
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


# Redis 依赖检查
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis 未安装，限流不可用。请运行: pip install redis")


DEFAULT_REQUESTS_PER_MINUTE = 60


class RateLimiter:
    """
    Redis 滑动窗口限流器（per-tenant）

    P1-2: 固定 requests_per_minute=60，移除 tier 抽象。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("需要安装 redis: pip install redis")

        import redis.asyncio as aioredis
        self._client = aioredis.Redis(host=host, port=port, decode_responses=False)
        self._requests_per_minute = requests_per_minute

    def _get_key(self, tenant_id: str) -> str:
        return f"ratelimit:{tenant_id}"

    async def check_rate_limit(
        self,
        tenant_id: str,
    ) -> tuple[bool, int, int]:
        """
        检查限流状态

        Args:
            tenant_id: 租户 ID

        Returns:
            (allowed, remaining, reset_in_seconds)
        """
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

        results = await pipe.execute()
        current_count = results[1]  # zcard result

        remaining = max(0, self._requests_per_minute - current_count - 1)
        allowed = current_count < self._requests_per_minute

        # 如果不允许，不要将当前请求计入
        if not allowed:
            await self._client.zrem(key, f"{now}")

        logger.debug(
            f"RateLimit: tenant={tenant_id} "
            f"allowed={allowed} remaining={remaining}"
        )

        return allowed, remaining, 60

    async def get_window_count(self, tenant_id: str) -> int:
        """获取当前窗口内的请求数"""
        key = self._get_key(tenant_id)
        now = time.time()
        window_start = now - 60
        return await self._client.zcount(key, window_start, now)


# 全局限流器实例（延迟初始化）
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(host: str = "localhost", port: int = 6379) -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(host=host, port=port)
    return _rate_limiter


class RateLimitExceeded(Exception):
    """限流超出异常"""
    pass


class RateLimitMiddleware:
    """
    FastAPI/Starlette 限流中间件

    从请求头中提取 tenant_id，默认 fallback 到 IP。
    将限流结果以响应头返回: X-RateLimit-Remaining, X-RateLimit-Reset

    P1-2: 移除 _extract_tier() — 无前端传 X-Rate-Limit-Tier header。
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

        try:
            allowed, remaining, reset_in = await limiter.check_rate_limit(tenant_id)
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
        """
        从 JWT 提取 tenant_id（user sub）。
        """
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer "):
            try:
                from backend.security.auth import decode_token
                claims = decode_token(auth[7:])
                return claims.get("sub", "anonymous")
            except Exception:
                pass
        client = scope.get("client")
        if client:
            return f"ip:{client[0]}"
        return "unknown"
