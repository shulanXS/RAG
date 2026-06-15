"""
test_query_rewriter.py — P3.1 补充测试
================================================================================
P0-6: QueryIntent / QueryType 已从 query_rewriter 移除。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.retrieval.query_rewriter import QueryRewriter


def test_needs_rewriting_short_query():
    rw = QueryRewriter(llm_client=None)
    assert rw._needs_rewriting("第二点呢") is True
    assert rw._needs_rewriting("hi") is True
    assert rw._needs_rewriting("What about it?") is True


def test_needs_rewriting_full_query():
    rw = QueryRewriter(llm_client=None)
    # 完整问句不需要重写
    assert rw._needs_rewriting("RAG 系统的核心组件是什么") is False
    assert rw._needs_rewriting("Explain the difference between BM25 and dense retrieval.") is False


def test_cache_key_is_deterministic():
    rw = QueryRewriter(llm_client=None)
    k1 = rw._make_cache_key("hi", None)
    k2 = rw._make_cache_key("hi", None)
    assert k1 == k2
    k3 = rw._make_cache_key("hi", [{"role": "user", "content": "old"}])
    assert k1 != k3


def test_rewriter_cache_get_put():
    rw = QueryRewriter(llm_client=None)
    key = "test_key"
    assert rw._cache_get(key) is None
    rw._cache_put(key, "value")
    assert rw._cache_get(key) == "value"


def test_rewriter_rewrites_with_history():
    """多轮对话场景：第二点呢 -> 包含前文实体"""
    from backend.retrieval.query_rewriter import RewrittenQuery

    rw = QueryRewriter(llm_client=None)
    history = [{"role": "user", "content": "RAG 系统有 3 个核心组件"}]
    # 无 LLM 的情况下应回退到原 query
    result = rw.rewrite("第二点呢", conversation_history=history)
    assert isinstance(result, RewrittenQuery)
    assert result.original == "第二点呢"


def test_cache_stats():
    rw = QueryRewriter(llm_client=None)
    stats = rw.get_cache_stats()
    assert "size" in stats
    assert stats["size"] == 0
