"""
test_circuit_breaker.py — P3.1 补充测试
================================================================================
"""

from __future__ import annotations

import asyncio

import pytest

from backend.middleware.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)


def _cfg(fail_threshold: int = 3, recovery_timeout: float = 0.5) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_threshold=fail_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=1,
        half_open_max_calls=1,
    )


@pytest.mark.asyncio
async def test_breaker_starts_closed():
    cb = CircuitBreaker("test", _cfg())
    assert cb.state == CircuitState.CLOSED

    async def ok():
        return 1

    assert await cb.call_async(ok) == 1
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold():
    cb = CircuitBreaker("test", _cfg(fail_threshold=3))

    async def fail():
        raise RuntimeError("boom")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call_async(fail)

    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        await cb.call_async(fail)


@pytest.mark.asyncio
async def test_breaker_half_open_then_closed():
    cb = CircuitBreaker("test", _cfg(fail_threshold=2, recovery_timeout=0.1))
    async def fail():
        raise RuntimeError("boom")
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call_async(fail)
    assert cb.state == CircuitState.OPEN
    await asyncio.sleep(0.15)
    # recovery_timeout 已过 -> HALF_OPEN
    async def ok():
        return "ok"
    result = await cb.call_async(ok)
    assert result == "ok"
    # success_threshold=1 -> 立即回到 CLOSED
    assert cb.state == CircuitState.CLOSED


def test_breaker_sync_fallback_path():
    """无 event loop 时 _on_failure_sync 不应崩"""
    cb = CircuitBreaker("test", _cfg(fail_threshold=5))
    cb._on_failure_sync()
    cb._on_failure_sync()
    assert cb._stats.failed_calls == 2
    cb._on_success_sync()
    assert cb._stats.successful_calls == 1


def test_breaker_recovery_remaining():
    cb = CircuitBreaker("test", _cfg(recovery_timeout=10.0))
    cb._stats.state = CircuitState.OPEN
    remaining = cb._recovery_remaining()
    assert 0 <= remaining <= 10
