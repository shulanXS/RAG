"""
test_generation.py — Generation 层核心测试
================================================================================
覆盖现存的 2 个组件:
1. PromptBuilder — 真实 yaml 加载 + prompt 组装 (system/context)
2. retry.with_retry — 指数退避 + 可重试/不可重试分类

历史: P0 阶段删除 grounded_generator / citation_generator,
Phase 1.2 进一步删除 CitationVerifier（无前端消费）— 此文件相应清理。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.generation.prompt_builder import PromptBuilder
from backend.generation.retry import (
    NON_RETRYABLE_EXCEPTIONS,
    RetryConfig,
    _compute_delay,
    _is_retryable,
    with_retry,
)


# =============================================================================
# 1. PromptBuilder
# =============================================================================


class TestPromptBuilder:
    """PromptBuilder 真实 prompt 组装测试 — 读真实 yaml, 验证关键段"""

    def test_prompt_version_is_set(self):
        """prompt_version 应从 yaml 加载, 非空"""
        builder = PromptBuilder()
        assert builder.prompt_version
        assert builder.prompt_version != "unknown"

    def test_prompt_hash_is_stable(self):
        """prompt_hash 应是 12 字符 hex (SHA256 前 12 位)"""
        builder = PromptBuilder()
        assert len(builder.prompt_hash) == 12
        assert all(c in "0123456789abcdef" for c in builder.prompt_hash)

    def test_prompt_hash_matches_content(self):
        """同 yaml 两次构建 prompt_hash 一致 (deterministic)"""
        b1 = PromptBuilder()
        b2 = PromptBuilder()
        assert b1.prompt_hash == b2.prompt_hash

    def test_build_context_with_chunks(self):
        """带 chunks 时输出 [N] 来源 标注 + 分隔符"""
        builder = PromptBuilder()
        chunks = [
            {"doc_id": "doc-1", "section_path": "§1", "text": "first content"},
            {"doc_id": "doc-2", "section_path": "", "text": "second"},
        ]
        ctx = builder.build_context(chunks)

        assert "[1] 来源: doc-1 / §1" in ctx
        assert "[2] 来源: doc-2" in ctx  # 无 section 时不显示
        assert "first content" in ctx
        assert "---" in ctx  # 分隔符

    def test_build_context_empty(self):
        """无 chunks 时返回明确提示"""
        builder = PromptBuilder()
        assert "未检索到" in builder.build_context([])

    def test_build_context_truncates_long_text(self):
        """>500 字符的 chunk 文本被截断到 500 + '..'."""
        builder = PromptBuilder()
        long_text = "x" * 1000
        chunks = [{"doc_id": "d", "text": long_text}]
        ctx = builder.build_context(chunks)

        # 500 字符 + "..." = 截断后总长 503
        assert "..." in ctx
        assert len(ctx) < len(long_text) + 100  # 应明显短于原文

    def test_build_prompt_includes_query_and_context(self):
        """build_prompt 输出含 query, context, 引用要求, 置信度要求"""
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            query="X产品的价格",
            context="[1] 价格: ¥100",
        )

        assert "X产品的价格" in prompt
        assert "[1] 价格: ¥100" in prompt
        assert "system" in prompt.lower() or "你" in prompt  # system prompt 段

    def test_build_prompt_with_citations_disabled(self):
        """require_citations=False 时不含引用要求段"""
        builder = PromptBuilder()
        prompt_full = builder.build_prompt(
            "q", "c", require_citations=True, require_confidence=True
        )
        prompt_no_cite = builder.build_prompt(
            "q", "c", require_citations=False, require_confidence=False
        )

        # 不含引用要求时应更短 (至少短 50 字符)
        assert len(prompt_no_cite) < len(prompt_full) - 50

    def test_structured_prompt_methods_removed(self):
        """Phase 1.3: build_structured_prompt / _format_schema / _schema_to_text 已删除（无调用方）"""
        builder = PromptBuilder()
        assert not hasattr(builder, "build_structured_prompt")
        assert not hasattr(builder, "_format_schema")
        assert not hasattr(builder, "_schema_to_text")


# =============================================================================
# 3. retry
# =============================================================================


class TestRetry:
    """retry.with_retry + _is_retryable 单元测试"""

    def test_is_retryable_value_error(self):
        """ValueError 不可重试 (NON_RETRYABLE_EXCEPTIONS 白名单)"""
        assert _is_retryable(ValueError("bad input")) is False
        assert _is_retryable(TypeError("type mismatch")) is False
        assert _is_retryable(KeyError("missing")) is False

    def test_is_retryable_connection_error(self):
        """ConnectionError / TimeoutError / OSError 可重试"""
        assert _is_retryable(ConnectionError("refused")) is True
        assert _is_retryable(TimeoutError("slow")) is True
        assert _is_retryable(OSError("disk")) is True

    def test_is_retryable_unknown_defaults_retryable(self):
        """未分类异常默认可重试 (保守策略)"""
        class WeirdError(Exception):
            pass

        assert _is_retryable(WeirdError("???")) is True

    def test_non_retryable_constant(self):
        """NON_RETRYABLE_EXCEPTIONS 是元组"""
        assert isinstance(NON_RETRYABLE_EXCEPTIONS, tuple)
        assert ValueError in NON_RETRYABLE_EXCEPTIONS

    def test_compute_delay_increases_with_attempt(self):
        """delay 随 attempt 指数增长 (无 jitter 时)"""
        cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=0.0)
        d0 = _compute_delay(0, cfg)
        d1 = _compute_delay(1, cfg)
        d2 = _compute_delay(2, cfg)
        # 1, 2, 4 — 严格递增
        assert d0 == 1.0
        assert d1 == 2.0
        assert d2 == 4.0

    def test_compute_delay_capped_at_max_delay(self):
        """delay 上限是 max_delay"""
        cfg = RetryConfig(base_delay=1.0, exponential_base=10.0, max_delay=5.0, jitter=0.0)
        # attempt=10 → 10^10 = 1e10, 远超 max_delay=5
        d = _compute_delay(10, cfg)
        assert d == 5.0

    def test_compute_delay_jitter_within_range(self):
        """jitter 范围 = base * jitter (±50%)"""
        cfg = RetryConfig(base_delay=2.0, exponential_base=2.0, jitter=0.5, max_delay=100.0)
        # 多采样, 验证总在 [base - base*jitter, base + base*jitter] = [1, 3]
        for _ in range(50):
            d = _compute_delay(0, cfg)
            assert 1.0 <= d <= 3.0

    @pytest.mark.asyncio
    async def test_with_retry_success_first_attempt(self):
        """首次就成功 → 1 次调用, 不重试"""
        call_count = 0

        async def _call():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(_call, RetryConfig(max_attempts=3))
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_with_retry_succeeds_after_failures(self):
        """前 2 次 ConnectionError, 第 3 次成功"""
        call_count = 0

        async def _call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("flake")
            return "recovered"

        result = await with_retry(
            _call,
            RetryConfig(max_attempts=3, base_delay=0.001, jitter=0.0),  # 加速测试
        )
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_gives_up_on_non_retryable(self):
        """ValueError 不重试, 立即抛出"""
        call_count = 0

        async def _call():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await with_retry(
                _call,
                RetryConfig(max_attempts=5, base_delay=0.001, jitter=0.0),
            )
        assert call_count == 1  # 只调 1 次就 raise

    @pytest.mark.asyncio
    async def test_with_retry_raises_after_max_attempts(self):
        """可重试异常用尽 max_attempts 后抛出最后一次的异常"""
        call_count = 0

        async def _call():
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"attempt {call_count}")

        with pytest.raises(ConnectionError, match="attempt 3"):
            await with_retry(
                _call,
                RetryConfig(max_attempts=3, base_delay=0.001, jitter=0.0),
            )
        assert call_count == 3
