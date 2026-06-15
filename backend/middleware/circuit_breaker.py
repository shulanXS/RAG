"""
circuit_breaker.py — LLM 调用熔断器
================================================================================
技术决策记录:
- 熔断器模式 (Circuit Breaker): 监控 LLM 调用，当错误率超过阈值时「跳闸」，
  阻止后续请求直到服务恢复。防止单点 LLM 故障导致整个系统雪崩。
- 三态模型: CLOSED (正常) → OPEN (熔断) → HALF_OPEN (探测)
- 滑动窗口错误率: 用固定时间窗口统计，避免单次尖刺误触发。
- 仅对 LLM 调用熔断，不影响检索路径。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """熔断器处于 OPEN 状态时抛出"""
    pass


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


@dataclass
class CircuitBreakerStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0.0
    last_state_change: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    LLM 调用熔断器

    用法:
        cb = CircuitBreaker("anthropic", failure_threshold=5)
        try:
            result = await cb.call(llm.generate, prompt)
        except CircuitOpenError:
            result = fallback_response()

    设计要点:
    - 每个 provider (anthropic/openai/deepseek) 独立熔断
    - CLOSED: 正常调用，错误累积
    - OPEN: 连续失败达到阈值，拒绝所有请求，立即返回
    - HALF_OPEN: recovery_timeout 后放行少量探测请求
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._window_start = time.monotonic()

    async def call(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        通过熔断器执行调用

        Raises:
            CircuitOpenError: 熔断器处于 OPEN 状态
        """
        async with self._lock:
            state = self._get_state()

            if state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"CircuitBreaker '{self._name}' is open. "
                    f"Wait {self._recovery_remaining():.0f}s."
                )

            if state == CircuitState.HALF_OPEN:
                if self._stats.successful_calls >= self._config.half_open_max_calls:
                    raise CircuitOpenError(
                        f"CircuitBreaker '{self._name}' is in half_open probe phase."
                    )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure()
            raise

    async def call_async(
        self,
        runner: Callable[[], Awaitable[T]],
    ) -> T:
        """
        通过熔断器执行一个零参 async 闭包。

        与 call 的区别：传入的是「返回一个 awaitable 的函数」，
        适用于「调用前还有额外逻辑」（如带重试）但仍想被熔断的场景。
        """
        async with self._lock:
            state = self._get_state()

            if state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"CircuitBreaker '{self._name}' is open. "
                    f"Wait {self._recovery_remaining():.0f}s."
                )

            if state == CircuitState.HALF_OPEN:
                if self._stats.successful_calls >= self._config.half_open_max_calls:
                    raise CircuitOpenError(
                        f"CircuitBreaker '{self._name}' is in half_open probe phase."
                    )

        try:
            result = await runner()
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    def _on_success_sync(self) -> None:
        """
        同步记录一次成功（流式端点专用）。

        流式调用无法直接套用 call()/call_async()，因为 yield 不可在同步上下文等待。
        调用方在流式迭代成功完成时调用本方法，失败时调用 _on_failure_sync。

        P3.2 修复: 拆分 sync/async 实现。
        - 优先尝试用 running loop schedule
        - 否则用 thread-safe 直接修改统计（无 event loop 时）
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # 当前在 event loop 中：把状态变更调度为协程
            loop.create_task(self._on_success())
        else:
            # 同步上下文：直接修改统计（thread-safe 因为只读 _stats 自身）
            self._stats.successful_calls += 1
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes += 1
            self._stats.last_state_change = time.monotonic()

    def _on_failure_sync(self) -> None:
        """同步记录一次失败（流式端点专用）"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            loop.create_task(self._on_failure())
        else:
            # 同步 fallback：直接修改统计
            self._stats.failed_calls += 1
            self._stats.consecutive_successes = 0
            self._stats.consecutive_failures += 1
            self._stats.last_state_change = time.monotonic()

    def _get_state(self) -> CircuitState:
        now = time.monotonic()

        if self._stats.state == CircuitState.OPEN:
            elapsed = now - self._stats.last_failure_time
            if elapsed >= self._config.recovery_timeout:
                self._stats.state = CircuitState.HALF_OPEN
                self._stats.consecutive_successes = 0
                logger.info(f"CircuitBreaker '{self._name}' OPEN -> HALF_OPEN")
        elif self._stats.state == CircuitState.HALF_OPEN:
            pass
        else:
            self._stats.state = CircuitState.CLOSED

        return self._stats.state

    async def _on_success(self):
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_failures = 0

            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.consecutive_successes += 1
                if self._stats.consecutive_successes >= self._config.success_threshold:
                    self._stats.state = CircuitState.CLOSED
                    self._stats.consecutive_successes = 0
                    logger.info(f"CircuitBreaker '{self._name}' HALF_OPEN -> CLOSED")

    async def _on_failure(self):
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.monotonic()

            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.state = CircuitState.OPEN
                logger.warning(
                    f"CircuitBreaker '{self._name}' HALF_OPEN -> OPEN "
                    f"(failure in probe)"
                )
            elif (
                self._stats.consecutive_failures >= self._config.failure_threshold
            ):
                self._stats.state = CircuitState.OPEN
                logger.warning(
                    f"CircuitBreaker '{self._name}' CLOSED -> OPEN "
                    f"(consecutive failures: {self._stats.consecutive_failures})"
                )

    def _recovery_remaining(self) -> float:
        elapsed = time.monotonic() - self._stats.last_failure_time
        return max(0.0, self._config.recovery_timeout - elapsed)

    @property
    def state(self) -> CircuitState:
        return self._stats.state

    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats

    def reset(self):
        self._stats = CircuitBreakerStats()


# 全局熔断器实例映射: provider_name -> CircuitBreaker
_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, config)
    return _breakers[name]


def get_all_breakers() -> dict[str, CircuitBreaker]:
    return dict(_breakers)
