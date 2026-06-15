"""
retry.py — 异步重试策略（指数退避 + jitter）
================================================================================
技术决策记录:
- 指数退避 + jitter: 多个客户端同时重试时（thundering herd），
  jitter 随机化退避时间可避免请求尖刺。
- 配置化: 不同 LLM provider / endpoint 可配置不同的重试参数。
- 异常分类: 区分「可重试」与「不可重试」异常，避免对 4xx 类错误做无意义重试。
- 失败传播: 重试用尽后异常向上抛，由调用方决定降级策略（Circuit Breaker）。
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 不可重试的异常（重试无意义）
NON_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (ValueError, TypeError, KeyError)


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: float = 0.5  # ±50% 随机抖动


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否可重试"""
    if isinstance(exc, NON_RETRYABLE_EXCEPTIONS):
        return False
    # 常见可重试异常: TimeoutError, ConnectionError, OSError
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    # httpx 异常（httpx 可选依赖）
    try:
        import httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
            return True
    except ImportError:
        pass
    # 其他异常默认可重试（保守策略）
    return True


def _compute_delay(attempt: int, cfg: RetryConfig) -> float:
    """计算第 N 次重试的退避时间（含 jitter）"""
    base = min(cfg.base_delay * (cfg.exponential_base ** attempt), cfg.max_delay)
    jitter_range = base * cfg.jitter
    return base + random.uniform(-jitter_range, jitter_range)


async def with_retry(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
) -> T:
    """
    异步重试包装器：指数退避 + jitter。

    Args:
        func: 返回 Awaitable 的零参 callable（每次重试都创建新的 awaitable）
        config: 重试配置，None 则使用默认

    Returns:
        函数执行结果

    Raises:
        最后一次重试的异常

    Examples:
        async def _call():
            return await client.generate_async(prompt)

        result = await with_retry(_call, RetryConfig(max_attempts=3))
    """
    cfg = config or RetryConfig()

    for attempt in range(cfg.max_attempts):
        try:
            return await func()
        except Exception as e:
            if not _is_retryable(e):
                logger.debug(f"Non-retryable exception, not retrying: {e}")
                raise

            is_last = attempt == cfg.max_attempts - 1
            if is_last:
                logger.error(
                    f"LLM call failed on final attempt ({attempt + 1}/{cfg.max_attempts}): {e}"
                )
                raise

            delay = _compute_delay(attempt, cfg)
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{cfg.max_attempts}), "
                f"retrying in {delay:.2f}s: {type(e).__name__}: {e}"
            )
            await asyncio.sleep(delay)

    # 逻辑上不可达（最后一次失败已 raise），但保持类型安全
    raise RuntimeError("with_retry: unexpected control flow")
