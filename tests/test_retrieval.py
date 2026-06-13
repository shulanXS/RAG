"""
test_retrieval.py — 检索链路测试
"""

import pytest
from backend.retrieval.fusion import RRFFusion, WeightedFusion, FusionResult
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


class TestWeightedFusion:
    """加权融合测试"""

    def test_weighted_fusion(self):
        """加权融合测试"""
        fusion = WeightedFusion(weights={"bm25": 0.7, "dense": 0.3})

        bm25 = [
            BM25Result(chunk_id="c1", doc_id="d1", score=10.0, rank=1, text="", metadata={}),
            BM25Result(chunk_id="c2", doc_id="d1", score=5.0, rank=2, text="", metadata={}),
        ]

        dense = [
            VectorSearchResult(chunk_id="c1", doc_id="d1", score=0.9, rank=1, text="", section_path="", metadata={}),
            VectorSearchResult(chunk_id="c2", doc_id="d1", score=0.85, rank=2, text="", section_path="", metadata={}),
        ]

        fused = fusion.fuse({"bm25": bm25, "dense": dense})

        assert len(fused) >= 1
        # c1 在两路都排第一，加权得分应该最高
        assert fused[0].chunk_id == "c1"


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
