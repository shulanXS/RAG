"""
test_retrieval.py — 检索链路测试
================================================================================
P0-7: BM25Retriever / BM25Result 已删除。RRF 测试中的 bm25_results 改用
types.SimpleNamespace 作为属性访问的最小替身（fusion.fuse 用 result.chunk_id
等属性访问；dict 会触发 AttributeError）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from backend.retrieval.fusion import (
    DynamicRRFFusion,
    DEFAULT_K_BY_COMPLEXITY,
    FusionResult,
)
from backend.retrieval.vector_retriever import VectorSearchResult


def _bm25_result(chunk_id, doc_id, score, rank, text="", metadata=None):
    """BM25Result 字段的 SimpleNamespace 替身（P0-7: BM25Result 类已删除）"""
    return SimpleNamespace(
        chunk_id=chunk_id,
        doc_id=doc_id,
        score=score,
        rank=rank,
        text=text,
        metadata=metadata or {},
    )


def _make_fuser(k=60):
    """Phase1-1.6: RRFFusion 已合并到 DynamicRRFFusion 内联 RRF。"""
    return DynamicRRFFusion(k_default=k)


class TestRRFFusion:
    """RRF 融合测试（Phase1-1.6 合并到 DynamicRRFFusion）"""

    def test_basic_fusion(self):
        """基础融合测试"""
        fusion = _make_fuser()

        bm25_results = [
            _bm25_result("c1", "d1", 5.0, 1, "text1"),
            _bm25_result("c2", "d1", 3.0, 2, "text2"),
            _bm25_result("c3", "d2", 4.0, 3, "text3"),
        ]

        dense_results = [
            VectorSearchResult(chunk_id="c1", doc_id="d1", score=0.95, rank=1, text="text1", section_path="", metadata={}),
            VectorSearchResult(chunk_id="c4", doc_id="d2", score=0.90, rank=2, text="text4", section_path="", metadata={}),
            VectorSearchResult(chunk_id="c2", doc_id="d1", score=0.85, rank=3, text="text2", section_path="", metadata={}),
        ]

        fused = fusion.fuse({"bm25": bm25_results, "dense": dense_results})

        assert len(fused) > 0
        assert fused[0].chunk_id == "c1"  # c1 在两路都排名第一
        assert fused[0].sources == ["bm25", "dense"]  # 来自两路
        assert all(hasattr(r, "fused_score") for r in fused)
        assert all(hasattr(r, "rank") for r in fused)

    def test_single_source_fusion(self):
        """单路融合测试"""
        fusion = _make_fuser()

        results = [
            VectorSearchResult(chunk_id="c1", doc_id="d1", score=0.95, rank=1, text="text1", section_path="", metadata={}),
            VectorSearchResult(chunk_id="c2", doc_id="d1", score=0.85, rank=2, text="text2", section_path="", metadata={}),
        ]

        fused = fusion.fuse({"dense": results})
        assert len(fused) == 2
        assert fused[0].sources == ["dense"]

    def test_rrf_scoring(self):
        """RRF 得分计算测试"""
        fusion = _make_fuser()

        # 两路都排第一的文档应该得分最高
        top_results = [
            VectorSearchResult(chunk_id="c1", doc_id="d1", score=0.95, rank=1, text="", section_path="", metadata={}),
            VectorSearchResult(chunk_id="c2", doc_id="d1", score=0.85, rank=1, text="", section_path="", metadata={}),
        ]
        bottom_results = [
            VectorSearchResult(chunk_id="c3", doc_id="d2", score=0.10, rank=5, text="", section_path="", metadata={}),
            VectorSearchResult(chunk_id="c4", doc_id="d2", score=0.05, rank=5, text="", section_path="", metadata={}),
        ]

        fused = fusion.fuse({"top": top_results, "bottom": bottom_results})

        # c1 和 c2 的得分应该高于 c3 和 c4
        assert fused[0].fused_score > fused[-1].fused_score


# ===========================================================================
# DynamicRRFFusion
# ===========================================================================

class TestDynamicRRFFusion:
    """按 query complexity 动态选 k 的 RRF 融合器"""

    def test_k_by_complexity_simple_uses_small_k(self):
        """simple → k=30 (小 k, 看重头部)"""
        d = DynamicRRFFusion()
        assert d.k_for_complexity("simple") == 30

    def test_k_by_complexity_complex_uses_large_k(self):
        """complex → k=90 (大 k, 各路均衡)"""
        d = DynamicRRFFusion()
        assert d.k_for_complexity("complex") == 90

    def test_k_by_complexity_moderate_uses_default(self):
        """moderate → k=60 (RRF 默认)"""
        d = DynamicRRFFusion()
        assert d.k_for_complexity("moderate") == 60

    def test_k_by_complexity_none_uses_default(self):
        """complexity=None (无路由信号) → fallback k_default"""
        d = DynamicRRFFusion(k_default=60)
        assert d.k_for_complexity(None) == 60

    def test_k_disabled_uses_default(self):
        """config 开关关闭 → 永远用 k_default, 不论 complexity"""
        d = DynamicRRFFusion(k_default=60, enabled=False)
        assert d.k_for_complexity("simple") == 60
        assert d.k_for_complexity("complex") == 60
        assert d.k_for_complexity(None) == 60

    def test_custom_k_by_complexity_mapping(self):
        """config 可覆盖默认 k 映射"""
        custom = {"simple": 10, "complex": 100, "moderate": 50, "beyond_kb": 50}
        d = DynamicRRFFusion(k_by_complexity=custom)
        assert d.k_for_complexity("simple") == 10
        assert d.k_for_complexity("complex") == 100

    def test_fuse_uses_dynamic_k(self):
        """fuse 行为应与 RRF 一致, 但 k 由 complexity 决定"""
        d = DynamicRRFFusion()
        bm25 = [_bm25_result("c1", "d1", 1.0, 1, "x")]
        dense = [VectorSearchResult(chunk_id="c2", doc_id="d1", score=0.9, rank=1, text="y", section_path="", metadata={})]
        results = d.fuse({"bm25": bm25, "dense": dense}, complexity="simple")
        # 简单查询用 k=30, RRF 公式 1/(k+rank)
        # c1: bm25 rank 1 → 1/(30+1) = 1/31
        # c2: dense rank 1 → 1/31
        # 两者得分相等
        assert len(results) == 2
        scores = {r.chunk_id: r.fused_score for r in results}
        assert abs(scores["c1"] - 1/31) < 1e-6
        assert abs(scores["c2"] - 1/31) < 1e-6

    def test_fuse_without_complexity_falls_back_to_k60(self):
        """fuse 不传 complexity → k=60"""
        d = DynamicRRFFusion()
        bm25 = [_bm25_result("c1", "d1", 1.0, 1, "x")]
        dense = [VectorSearchResult(chunk_id="c2", doc_id="d1", score=0.9, rank=1, text="y", section_path="", metadata={})]
        results = d.fuse({"bm25": bm25, "dense": dense})
        # k=60, 1/(60+1) = 1/61
        scores = {r.chunk_id: r.fused_score for r in results}
        assert abs(scores["c1"] - 1/61) < 1e-6

    def test_default_k_mapping_has_all_complexities(self):
        """sanity: 默认 mapping 覆盖所有 QueryComplexity 值"""
        from backend.agentic.query_router import QueryComplexity
        for level in QueryComplexity:
            assert level.value in DEFAULT_K_BY_COMPLEXITY, (
                f"missing k for {level.value!r}"
            )
