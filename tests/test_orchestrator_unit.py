"""
test_orchestrator_unit.py — P3.3 orchestrator 单元测试
================================================================================
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# langgraph 是可选依赖 (CI 环境会装, 离线开发环境可缺)
langgraph = pytest.importorskip("langgraph")

from backend.domain.agent.orchestrator import AgenticOrchestrator, OrchestratorResult


@pytest.fixture
def mock_orchestrator() -> AgenticOrchestrator:
    """构造一个最小可用的 orchestrator，所有外部依赖 mock"""
    mock_hybrid = MagicMock()
    mock_hybrid.search = AsyncMock(return_value=(
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text": "test content",
                "rerank_score": 0.9,
                "rrf_score": 0.85,
            }
        ],
        {"latency_ms": 50.0, "query": "test"},
    ))

    mock_llm = MagicMock()
    mock_llm.generator_client = MagicMock()
    mock_llm.router_client = MagicMock()

    orch = AgenticOrchestrator(
        hybrid_search_engine=mock_hybrid,
        llm_client=mock_llm,
    )
    return orch


@pytest.mark.asyncio
async def test_orchestrator_returns_orchestrator_result(mock_orchestrator: AgenticOrchestrator):
    """基本调用返回 OrchestratorResult"""
    mock_orchestrator._llm.generator_client.generate_async = AsyncMock(
        return_value="Based on [c1], RAG is a technique."
    )

    result = await mock_orchestrator.run("What is RAG?")
    assert isinstance(result, OrchestratorResult)
    assert result.answer is not None
    assert len(result.answer) > 0


@pytest.mark.asyncio
async def test_orchestrator_cache_hit_short_circuits():
    """P1.2: 缓存命中应短路执行"""
    mock_hybrid = MagicMock()
    mock_hybrid.search = AsyncMock(side_effect=Exception("should not be called"))

    orch = AgenticOrchestrator(
        hybrid_search_engine=mock_hybrid,
        semantic_cache_fn=lambda q, response=None: {
            "answer": "cached answer",
            "query": q,
            "sources": [],
        } if response is None else None,
    )

    result = await orch.run("hi")
    assert result.answer == "cached answer"
    # hybrid search 不应被调用
    mock_hybrid.search.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_handles_missing_hybrid_search():
    """无 hybrid search 时应优雅降级"""
    orch = AgenticOrchestrator(hybrid_search_engine=None, llm_client=None)
    chunks = await orch._run_retrieval("test")
    assert chunks == []


@pytest.mark.asyncio
async def test_orchestrator_router_classification_called(mock_orchestrator: AgenticOrchestrator):
    """Router 应对 query 进行分类"""
    with patch("backend.domain.agent.orchestrator.QueryRouter") as MockRouter:
        mock_router_instance = MagicMock()
        mock_router_instance.route = AsyncMock(return_value=MagicMock(
            complexity="simple",
            route="simple_qa",
            confidence=0.9,
            use_cache=True,
            skip_retrieval=False,
        ))
        MockRouter.return_value = mock_router_instance
        mock_orchestrator._router = mock_router_instance

        mock_orchestrator._llm.generator_client.generate_async = AsyncMock(
            return_value="RAG is retrieval augmented generation."
        )

        await mock_orchestrator.run("What is RAG?")
        # router.route 应被调用
        assert mock_router_instance.route.called
