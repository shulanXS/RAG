"""
test_retrieval.py — 检索链路测试
"""

import pytest
from backend.retrieval.fusion import (
    RRFFusion,
    DynamicRRFFusion,
    DEFAULT_K_BY_COMPLEXITY,
    FusionResult,
)
from backend.retrieval.bm25_retriever import BM25Result
from backend.retrieval.vector_retriever import VectorSearchResult


class TestRRFFusion:
    """RRF 融合测试"""

    def test_basic_fusion(self):
        """基础融合测试"""
        fusion = RRFFusion(k=60)

        bm25_results = [
            BM25Result(chunk_id="c1", doc_id="d1", score=5.0, rank=1, text="text1", metadata={}),
            BM25Result(chunk_id="c2", doc_id="d1", score=3.0, rank=2, text="text2", metadata={}),
            BM25Result(chunk_id="c3", doc_id="d2", score=4.0, rank=3, text="text3", metadata={}),
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
        fusion = RRFFusion(k=60)

        results = [
            VectorSearchResult(chunk_id="c1", doc_id="d1", score=0.95, rank=1, text="text1", section_path="", metadata={}),
            VectorSearchResult(chunk_id="c2", doc_id="d1", score=0.85, rank=2, text="text2", section_path="", metadata={}),
        ]

        fused = fusion.fuse({"dense": results})
        assert len(fused) == 2
        assert fused[0].sources == ["dense"]

    def test_rrf_scoring(self):
        """RRF 得分计算测试"""
        fusion = RRFFusion(k=60)

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


class TestBM25Result:
    """BM25 结果数据结构测试"""

    def test_bm25_result_creation(self):
        """BM25 结果创建测试"""
        result = BM25Result(
            chunk_id="test_chunk",
            doc_id="test_doc",
            score=5.5,
            rank=1,
            text="This is test content.",
            metadata={"section": "intro"},
        )

        assert result.chunk_id == "test_chunk"
        assert result.doc_id == "test_doc"
        assert result.score == 5.5
        assert result.rank == 1
        assert result.metadata["section"] == "intro"


# ===========================================================================
# P2-B5: DynamicRRFFusion
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
        """fuse 行为应与 RRFFusion 一致, 但 k 由 complexity 决定"""
        d = DynamicRRFFusion()
        bm25 = [BM25Result(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="x", metadata={})]
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
        """fuse 不传 complexity → k=60 (同 RRFFusion 默认)"""
        d = DynamicRRFFusion()
        bm25 = [BM25Result(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="x", metadata={})]
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
