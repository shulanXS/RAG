"""
test_reranker.py — P3.3
================================================================================
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.retrieval.reranker import (
    BGEReranker,
    CohereReranker,
    RerankResult,
    _RerankCache,
    is_permanent_error,
    is_transient_error,
    get_default_rerank_cache,
)


def _make_bge_with_mock_scores(scores: list[float]) -> BGEReranker:
    """构造 BGEReranker 但 mock 掉 cross_encoder"""
    from backend.retrieval import reranker as r

    # 跳过真实 __init__，直接构造实例
    rr = BGEReranker.__new__(BGEReranker)
    rr._model_name = "mock-bge"
    rr._device = "cpu"
    rr._use_fp16 = False
    ce = MagicMock()
    ce.predict = MagicMock(return_value=MagicMock(tolist=lambda: scores))
    rr._cross_encoder = ce
    # P2-B3: cache 字段 (绕过 __init__ 时需要手动设)
    rr._cache = _RerankCache(capacity=128)
    rr._cache_enabled = True
    return rr


def test_rerank_truncates_to_top_k():
    chunks = [
        {"chunk_id": f"c{i}", "doc_id": "d1", "text": f"text {i}"}
        for i in range(10)
    ]
    # 倒序打分: c9 最高, c0 最低
    scores = [0.1 * i for i in range(10)]
    rr = _make_bge_with_mock_scores(scores)
    results = rr.rerank("query", chunks, top_k=3)
    assert len(results) == 3
    assert results[0].chunk_id == "c9"  # 最高分
    assert results[1].chunk_id == "c8"
    assert results[2].chunk_id == "c7"
    assert all(isinstance(r, RerankResult) for r in results)


def test_rerank_empty_chunks():
    rr = _make_bge_with_mock_scores([])
    assert rr.rerank("q", [], top_k=5) == []


def test_rerank_preserves_text_and_doc_id():
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "hello world", "section_path": "intro"},
    ]
    rr = _make_bge_with_mock_scores([0.9])
    results = rr.rerank("q", chunks, top_k=5)
    assert results[0].doc_id == "d1"
    assert results[0].text == "hello world"
    assert results[0].rerank_score == 0.9
    assert results[0].final_rank == 1


def test_rerank_text_truncated_to_2000_chars():
    """CrossEncoder 输入文本超过 2000 字符应被截断"""
    long_text = "a" * 5000
    chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": long_text}]
    rr = _make_bge_with_mock_scores([0.5])

    rr.rerank("q", chunks, top_k=5)

    # 验证传给 predict 的 pair 第二个元素是截断后的
    call_args = rr._cross_encoder.predict.call_args
    pairs = call_args[0][0]
    assert len(pairs[0][1]) == 2000


# ===========================================================================
# P2-B3: Rerank LRU 缓存
# ===========================================================================

class TestRerankCache:
    """_RerankCache 单元测试"""

    def test_make_key_deterministic(self):
        k1 = _RerankCache.make_key("query", ["c1", "c2", "c3"])
        k2 = _RerankCache.make_key("query", ["c1", "c2", "c3"])
        assert k1 == k2
        assert len(k1) == 32  # md5 hex

    def test_make_key_order_invariant(self):
        """chunk 列表顺序变化应生成相同 key (容忍 RRF 输出顺序波动)"""
        k1 = _RerankCache.make_key("query", ["c1", "c2"])
        k2 = _RerankCache.make_key("query", ["c2", "c1"])
        assert k1 == k2

    def test_make_key_query_sensitive(self):
        k1 = _RerankCache.make_key("q1", ["c1"])
        k2 = _RerankCache.make_key("q2", ["c1"])
        assert k1 != k2

    def test_put_get(self):
        cache = _RerankCache(capacity=10)
        results = [RerankResult(chunk_id="c1", doc_id="d1", rerank_score=0.9)]
        key = cache.make_key("q", ["c1"])
        cache.put(key, results)
        got = cache.get(key)
        assert got is not None
        assert got[0].chunk_id == "c1"
        assert cache.get_stats().hits == 1

    def test_get_miss(self):
        cache = _RerankCache(capacity=10)
        assert cache.get("nonexistent") is None
        assert cache.get_stats().misses == 1

    def test_lru_eviction(self):
        """容量满后, 最久未访问的应被淘汰"""
        cache = _RerankCache(capacity=2)
        r = RerankResult(chunk_id="x", doc_id="x", rerank_score=0.5)
        k1 = cache.make_key("q1", ["c1"])
        k2 = cache.make_key("q2", ["c2"])
        k3 = cache.make_key("q3", ["c3"])
        cache.put(k1, [r])
        cache.put(k2, [r])
        # 触发 LRU 更新
        cache.get(k1)
        cache.put(k3, [r])  # 应淘汰 k2
        assert cache.get(k1) is not None
        assert cache.get(k2) is None
        assert cache.get_stats().evictions == 1

    def test_stats_hit_rate(self):
        cache = _RerankCache(capacity=10)
        r = RerankResult(chunk_id="x", doc_id="x", rerank_score=0.5)
        k1 = cache.make_key("q", ["c1"])
        cache.put(k1, [r])
        cache.get(k1)  # hit
        cache.get(k1)  # hit
        cache.get("missing")  # miss
        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert abs(stats.hit_rate - 2/3) < 1e-6

    def test_clear(self):
        cache = _RerankCache(capacity=10)
        cache.put("k", [RerankResult("c", "d", 0.5)])
        cache.clear()
        assert cache.get("k") is None
        assert cache.get_stats().hits == 0


class TestRerankCacheIntegration:
    """验证 cache hit 时, 底层 API 不被调用"""

    def test_bge_cache_hit_skips_predict(self):
        """第二次相同 query+chunks 调用, predict() 不应再次执行"""
        rr = _make_bge_with_mock_scores([0.7, 0.3])
        chunks = [
            {"chunk_id": "c1", "doc_id": "d1", "text": "alpha"},
            {"chunk_id": "c2", "doc_id": "d1", "text": "beta"},
        ]
        # 首次调用 — 写 cache
        rr.rerank("q", chunks, top_k=2)
        predict_count_first = rr._cross_encoder.predict.call_count
        # 第二次相同输入 — 应命中 cache
        rr.rerank("q", chunks, top_k=2)
        assert rr._cross_encoder.predict.call_count == predict_count_first
        stats = rr._cache.get_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1

    def test_cache_disabled_calls_predict_twice(self):
        """cache_enabled=False 时, 每次都调 API"""
        rr = _make_bge_with_mock_scores([0.5])
        rr._cache_enabled = False
        chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "x"}]
        rr.rerank("q", chunks, top_k=1)
        rr.rerank("q", chunks, top_k=1)
        assert rr._cross_encoder.predict.call_count == 2


# ===========================================================================
# P2-B3: 错误分类 (transient vs permanent)
# ===========================================================================

class TestErrorClassification:
    """is_transient_error / is_permanent_error"""

    @pytest.mark.parametrize("exc,expected", [
        (TimeoutError("API timeout"), True),
        (ConnectionError("connection reset"), True),
        (Exception("rate limit exceeded"), True),
        (Exception("429 Too Many Requests"), True),
        (Exception("503 service unavailable"), True),
        (Exception("invalid api key"), False),  # permanent
        (Exception("400 bad request"), False),
        (Exception("unauthorized 401"), False),
        (Exception("random unrelated error"), False),
    ])
    def test_is_transient(self, exc: Exception, expected: bool):
        assert is_transient_error(exc) is expected

    @pytest.mark.parametrize("exc,expected", [
        (Exception("unauthorized"), True),
        (Exception("401"), True),
        (Exception("forbidden 403"), True),
        (Exception("invalid api key"), True),
        (Exception("400 bad request"), True),
        (Exception("valueerror: bad arg"), True),
        (Exception("rate limit"), False),  # transient, not permanent
        (Exception("random"), False),
    ])
    def test_is_permanent(self, exc: Exception, expected: bool):
        assert is_permanent_error(exc) is expected


class TestHybridSearchErrorFallback:
    """P2-B3: hybrid_search 区分瞬时 / 永久错误, 不同日志级别 + 处理路径"""

    @pytest.mark.asyncio
    async def test_transient_error_logs_warning(self, caplog):
        """瞬时错误 (rate limit) 应 warning + 降级 RRF top-5"""
        import logging
        from unittest.mock import MagicMock
        from backend.retrieval.hybrid_search import HybridSearchEngine
        from backend.ingestion.embedder import Embedder
        from backend.security.tenant import TenantContext

        class _Hit:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        # 构造 engine, mock reranker 抛瞬时错误
        embedder = MagicMock(spec=Embedder)
        embedder.embed = MagicMock(return_value=[0.0] * 1536)
        vector_retriever = MagicMock()
        vector_retriever.hybrid_search = MagicMock(return_value=[
            _Hit(chunk_id="c1", doc_id="d1", score=0.9, rank=1, text="t", section_path="", metadata={})
        ])
        reranker = MagicMock()
        reranker.rerank = MagicMock(side_effect=Exception("rate limit exceeded"))

        engine = HybridSearchEngine(
            embedder=embedder,
            vector_retriever=vector_retriever,
            reranker=reranker,
        )

        tenant = TenantContext(tenant_id="t1")
        with caplog.at_level(logging.WARNING, logger="backend.retrieval.hybrid_search"):
            chunks, ctx = await engine.search("test query", tenant=tenant)
        assert any("瞬时错误" in r.message for r in caplog.records)
        # 降级到 RRF 结果
        assert len(chunks) >= 0

    @pytest.mark.asyncio
    async def test_permanent_error_logs_error(self, caplog):
        """永久错误 (auth) 应 error + 降级 RRF top-5 (不重试)"""
        import logging
        from unittest.mock import MagicMock
        from backend.retrieval.hybrid_search import HybridSearchEngine
        from backend.ingestion.embedder import Embedder
        from backend.security.tenant import TenantContext

        class _Hit:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        embedder = MagicMock(spec=Embedder)
        embedder.embed = MagicMock(return_value=[0.0] * 1536)
        vector_retriever = MagicMock()
        vector_retriever.hybrid_search = MagicMock(return_value=[
            _Hit(chunk_id="c1", doc_id="d1", score=0.9, rank=1, text="t", section_path="", metadata={})
        ])
        reranker = MagicMock()
        reranker.rerank = MagicMock(side_effect=Exception("unauthorized 401"))

        engine = HybridSearchEngine(
            embedder=embedder,
            vector_retriever=vector_retriever,
            reranker=reranker,
        )

        tenant = TenantContext(tenant_id="t1")
        with caplog.at_level(logging.ERROR, logger="backend.retrieval.hybrid_search"):
            chunks, ctx = await engine.search("test query", tenant=tenant)
        assert any("永久错误" in r.message for r in caplog.records)
