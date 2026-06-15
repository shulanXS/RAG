"""
test_self_rag.py — Self-RAG 兜底单元测试(plan §2.2)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.domain.agent.self_rag import SelfRAGJudge, Verdict


@pytest.fixture
def mock_llm():
    """构造 mock LLM,默认返回高置信度 correct 判定"""
    client = MagicMock()
    client.generate_async = AsyncMock(return_value=json.dumps({
        "confidence": 0.9,
        "verdict": "correct",
        "reason": "matches context",
    }))
    return client


@pytest.mark.asyncio
async def test_correct_passes_through(mock_llm):
    judge = SelfRAGJudge(llm_client=mock_llm)
    result = await judge.evaluate(
        query="什么是 RAG?",
        answer="RAG 是检索增强生成。",
        chunks=[{"chunk_id": "c1", "doc_id": "d1", "text": "RAG 是检索增强生成。", "section_path": ""}],
    )
    assert result.verdict == Verdict.CORRECT
    assert result.confidence >= 0.85
    assert result.refined is False
    assert "RAG" in result.answer


@pytest.mark.asyncio
async def test_ambiguous_adds_hint(mock_llm):
    mock_llm.generate_async = AsyncMock(return_value=json.dumps({
        "confidence": 0.6,
        "verdict": "ambiguous",
        "reason": "信息不足",
    }))
    judge = SelfRAGJudge(llm_client=mock_llm)
    result = await judge.evaluate(
        query="X?",
        answer="可能是 A。",
        chunks=[],
    )
    assert result.verdict == Verdict.AMBIGUOUS
    assert "信息可能不足" in result.answer


@pytest.mark.asyncio
async def test_incorrect_triggers_refine():
    """INCORRECT + 有 retrieve/rewrite 时,应触发 refine,confidence 提升"""
    client = MagicMock()
    # 第一次 judge: low confidence, INCORRECT
    # 第二次 judge(after refine): high confidence, CORRECT
    client.generate_async = AsyncMock(side_effect=[
        json.dumps({"confidence": 0.3, "verdict": "incorrect", "reason": "off-topic"}),
        json.dumps({"confidence": 0.9, "verdict": "correct", "reason": "refined match"}),
    ])
    retrieve_fn = AsyncMock(return_value=[
        {"chunk_id": "c2", "doc_id": "d2", "text": "refined context", "section_path": ""}
    ])
    rewrite_fn = AsyncMock(return_value="refined query")

    judge = SelfRAGJudge(llm_client=client, retrieval_fn=retrieve_fn, rewrite_fn=rewrite_fn)
    result = await judge.evaluate(
        query="X?",
        answer="old answer",
        chunks=[{"chunk_id": "c1", "doc_id": "d1", "text": "old", "section_path": ""}],
    )
    assert result.refined is True
    assert result.refine_attempts == 1
    assert result.confidence == 0.9
    assert retrieve_fn.call_count == 1
    assert rewrite_fn.call_count == 1


@pytest.mark.asyncio
async def test_incorrect_without_refine_capability_falls_back():
    """INCORRECT + 无 retrieve_fn 时,降级返回原结果"""
    client = MagicMock()
    client.generate_async = AsyncMock(return_value=json.dumps({
        "confidence": 0.3, "verdict": "incorrect", "reason": "x"
    }))
    judge = SelfRAGJudge(llm_client=client)  # 无 retrieve/rewrite
    result = await judge.evaluate(query="X?", answer="A", chunks=[])
    assert result.verdict == Verdict.INCORRECT
    assert result.refined is False
    assert result.refine_attempts == 0


@pytest.mark.asyncio
async def test_llm_failure_falls_back_gracefully():
    client = MagicMock()
    client.generate_async = AsyncMock(side_effect=RuntimeError("LLM down"))
    judge = SelfRAGJudge(llm_client=client)
    result = await judge.evaluate(query="X?", answer="A", chunks=[])
    assert result.verdict == Verdict.AMBIGUOUS
    assert result.confidence == 0.5


@pytest.mark.asyncio
async def test_no_llm_returns_ambiguous():
    judge = SelfRAGJudge(llm_client=None)
    result = await judge.evaluate(query="X?", answer="A", chunks=[])
    assert result.verdict == Verdict.AMBIGUOUS
    assert result.confidence == 0.5


def test_verdict_enum_values():
    assert Verdict.CORRECT.value == "correct"
    assert Verdict.INCORRECT.value == "incorrect"
    assert Verdict.AMBIGUOUS.value == "ambiguous"


def test_extract_citations_limits_to_5():
    chunks = [{"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": "x", "section_path": "s"} for i in range(10)]
    cites = SelfRAGJudge._extract_citations(chunks)
    assert len(cites) == 5
    assert cites[0]["chunk_id"] == "c0"


def test_max_refine_attempts_default_is_one():
    """plan §2.2 关键决策:硬编码 max_refine_attempts=1 避免无限 retry"""
    judge = SelfRAGJudge(llm_client=MagicMock())
    assert judge._max_refine == 1
