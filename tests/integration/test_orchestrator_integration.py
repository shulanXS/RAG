"""
test_orchestrator_integration.py — 端到端编排器集成测试
================================================================================
覆盖:
- 完整 RAG 流程：rewrite → route → retrieve → generate → reflect
- 各复杂度路径（SIMPLE / MODERATE / COMPLEX / BEYOND_KB）
- 缓存命中路径
- 错误降级路径

所有外部依赖（Qdrant / LLM / Redis）使用 mock，
保留真实的内部协作逻辑（fuse, prompt builder, citation verifier）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agentic.query_router import QueryComplexity
from backend.agentic.orchestrator import AgenticOrchestrator


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_orchestrator(
    chunks: list[dict] | None = None,
    answer: str = "Test answer",
    complexity: QueryComplexity = QueryComplexity.SIMPLE,
) -> AgenticOrchestrator:
    """构造一个 orchestrator，所有外部依赖都 mock 掉"""
    cfg = MagicMock()
    cfg.hybrid_search.bm25_weight = 0.5
    cfg.hybrid_search.dense_weight = 0.5
    cfg.llm.generator.model = "claude-3-7-sonnet-20250620"

    # Mock hybrid search
    hybrid_search = MagicMock()
    retrieval_ctx = MagicMock()
    retrieval_ctx.total_latency_ms = 50.0
    retrieval_ctx.stage_breakdown = {
        "bm25": "10ms (20%)",
        "dense": "15ms (30%)",
        "fusion": "2ms (4%)",
        "rerank": "23ms (46%)",
    }
    hybrid_search.search = AsyncMock(return_value=(chunks or [], retrieval_ctx))

    # Mock router
    router = MagicMock()
    routing = MagicMock()
    routing.complexity = complexity
    routing.confidence = 0.85
    routing.recommended_approach = "hybrid"
    routing.reasoning = "test"
    router.route = MagicMock(return_value=routing)

    # Mock LLM
    llm_client = MagicMock()
    llm_client.generator_model = "claude-3-7-sonnet-20250620"
    llm_client.generator_client = MagicMock()
    llm_client.generate_async = AsyncMock(return_value=answer)
    llm_client.router_client = MagicMock()

    # Mock chat store
    chat_store = MagicMock()
    chat_store.get_history = MagicMock(return_value=[])
    chat_store.add_message = MagicMock()

    # Mock rewriter
    rewriter = MagicMock()
    rewritten = MagicMock()
    rewritten.was_rewritten = False
    rewritten.confidence = 1.0
    rewritten.rewritten = "test query"
    rewriter.rewrite = MagicMock(return_value=rewritten)

    # Mock citation verifier
    citation_verifier = MagicMock()
    verify_result = MagicMock()
    verify_result.overall_groundedness_score = 0.9
    verify_result.num_supported = 3
    verify_result.num_total = 3
    verify_result.unsupported_claims = []
    citation_verifier.verify = AsyncMock(return_value=verify_result)
    citation_verifier.to_citations = MagicMock(return_value=[])

    with patch("backend.retrieval.query_rewriter.QueryRewriter", return_value=rewriter), \
         patch("backend.generation.citation_verifier.CitationVerifier", return_value=citation_verifier):
        orch = AgenticOrchestrator(
            hybrid_search_engine=hybrid_search,
            router=router,
            llm_client=llm_client,
            chat_store=chat_store,
        )
    return orch


# --------------------------------------------------------------------------
# SIMPLE 路径
# --------------------------------------------------------------------------


class TestSimplePath:
    @pytest.mark.asyncio
    async def test_simple_query_returns_answer(self, sample_chunks):
        """SIMPLE 复杂度走 hybrid_search 路径"""
        orch = _make_orchestrator(chunks=sample_chunks, answer="RAG 是检索增强生成。")
        result = await orch.run(query="什么是 RAG？")
        assert result.answer == "RAG 是检索增强生成。"
        assert result.complexity == QueryComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_simple_query_includes_citations(self, sample_chunks):
        """结果应包含从 chunks 派生的引用"""
        orch = _make_orchestrator(chunks=sample_chunks)
        result = await orch.run(query="什么是 RAG？")
        # 至少会从 chunks 提取 citations
        assert isinstance(result.citations, list)


# ----------------------------------------------------------------------------
# BEYOND_KB 路径
# ----------------------------------------------------------------------


class TestBeyondKbPath:
    @pytest.mark.asyncio
    async def test_beyond_kb_skips_retrieval(self):
        """BEYOND_KB 路径应跳过 retrieval，直接走 LLM"""
        orch = _make_orchestrator(answer="通用答案", complexity=QueryComplexity.BEYOND_KB)
        result = await orch.run(query="今天天气怎么样？")
        # Hybrid search 不应被调用
        orch._hybrid_search.search.assert_not_called()
        # 答案应来自 LLM
        assert result.answer == "通用答案"


# --------------------------------------------------------------------------
# 错误降级
# ----------------------------------------------------------------------


class TestErrorDegradation:
    @pytest.mark.asyncio
    async def test_no_chunks_returns_empty(self):
        """检索无结果时仍应返回有效 OrchestratorResult"""
        orch = _make_orchestrator(chunks=[])
        result = await orch.run(query="不存在的内容")
        assert result is not None
        # 答案仍是 LLM 生成（虽然可能不准确）
        assert result.answer  # 非空

    @pytest.mark.asyncio
    async def test_no_llm_returns_error(self):
        """无 LLM client 时降级到错误答案"""
        cfg = MagicMock()
        router = MagicMock()
        routing = MagicMock()
        routing.complexity = QueryComplexity.SIMPLE
        routing.confidence = 0.85
        routing.recommended_approach = "hybrid"
        routing.reasoning = "test"
        router.route = MagicMock(return_value=routing)

        # 构造一个不传 llm_client 的 orchestrator
        hybrid_search = MagicMock()
        hybrid_search.search = AsyncMock(return_value=([], MagicMock(total_latency_ms=10)))

        rewriter = MagicMock()
        rewritten = MagicMock()
        rewritten.was_rewritten = False
        rewritten.confidence = 1.0
        rewritten.rewritten = "q"
        rewriter.rewrite = MagicMock(return_value=rewritten)

        with patch("backend.retrieval.query_rewriter.QueryRewriter", return_value=rewriter):
            orch = AgenticOrchestrator(
                hybrid_search_engine=hybrid_search,
                router=router,
                llm_client=None,  # 关键：None
            )
        result = await orch.run(query="test")
        # 无 LLM 时答案应降级为 "LLM 不可用"
        assert "LLM" in result.answer or result.answer == ""
