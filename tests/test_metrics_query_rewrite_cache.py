"""
test_metrics_query_rewrite_cache.py — P3.1(plan §4.4) 单元测试

验证:
- MetricsCollector.record_query_rewrite_cache 增加 hit/miss 计数
- QueryRewriter.rewrite_async 命中 LRU 缓存时调用 hit=True
- QueryRewriter.rewrite_async 未命中时调用 hit=False
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.domain.retrieval.query_rewriter import QueryRewriter
from backend.observability.metrics import (
    create_metrics_collector,
    rag_query_rewrite_cache_hit,
    rag_query_rewrite_cache_miss,
)


def test_record_query_rewrite_cache_hit_increments():
    """record_query_rewrite_cache(hit=True) 应让 hit 计数增加。"""
    collector = create_metrics_collector()
    before = rag_query_rewrite_cache_hit._value.get()
    collector.record_query_rewrite_cache(hit=True)
    after = rag_query_rewrite_cache_hit._value.get()
    assert after == before + 1


def test_record_query_rewrite_cache_miss_increments():
    """record_query_rewrite_cache(hit=False) 应让 miss 计数增加。"""
    collector = create_metrics_collector()
    before = rag_query_rewrite_cache_miss._value.get()
    collector.record_query_rewrite_cache(hit=False)
    after = rag_query_rewrite_cache_miss._value.get()
    assert after == before + 1


@pytest.mark.asyncio
async def test_query_rewriter_cache_hit_records_metric():
    """Rewrite cache 命中时,应通过 create_metrics_collector 调用 record_query_rewrite_cache(hit=True)。"""
    rw = QueryRewriter(llm_client=None)
    history = [{"role": "user", "content": "first"}]
    key = rw._make_cache_key("第二点呢", history)

    # 预填 cache
    from backend.domain.retrieval.query_rewriter import RewrittenQuery
    cached = RewrittenQuery(rewritten="第二点的内容", was_rewritten=True, confidence=0.9, original="第二点呢")
    rw._cache_put(key, cached)

    with patch("backend.domain.retrieval.query_rewriter.create_metrics_collector") as mock_factory:
        mock_collector = MagicMock()
        mock_factory.return_value = mock_collector
        result = await rw.rewrite_async("第二点呢", conversation_history=history)

    assert result.rewritten == "第二点的内容"
    # cache 命中时,record_query_rewrite_cache 被以 hit=True 调用
    mock_collector.record_query_rewrite_cache.assert_called_once_with(hit=True)


@pytest.mark.asyncio
async def test_query_rewriter_cache_miss_records_metric():
    """Rewrite cache miss 时,应记录 hit=False。"""
    rw = QueryRewriter(llm_client=None)

    with patch("backend.domain.retrieval.query_rewriter.create_metrics_collector") as mock_factory:
        mock_collector = MagicMock()
        mock_factory.return_value = mock_collector
        # 代词查询, 无 LLM, 触发 cache miss 后写 cache
        await rw.rewrite_async("第二点呢", conversation_history=[])

    # 至少调用一次 record_query_rewrite_cache, hit=False(miss 路径)
    calls = mock_collector.record_query_rewrite_cache.call_args_list
    assert any(c.kwargs.get("hit") is False for c in calls) or any(
        len(c.args) >= 1 and c.args[0] is False for c in calls
    )
