"""
test_online_eval.py — 在线评估单元测试(plan §2.3)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.domain.evaluation.online import (
    OnlineEvalSample,
    OnlineEvaluator,
    default_ragas_eval,
    query_recent_online_metrics,
)


@pytest.mark.asyncio
async def test_should_sample_respects_rate():
    """1000 次采样, ~5% 命中"""
    ev = OnlineEvaluator(sample_rate=0.1)
    hits = sum(1 for _ in range(1000) if ev.should_sample())
    assert 70 < hits < 130  # 容许 ±3%


@pytest.mark.asyncio
async def test_maybe_evaluate_skipped_when_not_sampled(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.99)  # 远超 0.05
    ev = OnlineEvaluator(sample_rate=0.05)
    result = await ev.maybe_evaluate("q", "a", ["c"], 100, 0.9)
    assert result is None
    assert ev._sampled_count == 0


@pytest.mark.asyncio
async def test_maybe_evaluate_persists_sample(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.01)  # 命中 5% 采样
    persist = AsyncMock()
    ev = OnlineEvaluator(sample_rate=0.05, persist_fn=persist)
    result = await ev.maybe_evaluate("what is RAG?", "RAG is...", ["ctx1", "ctx2"], 250, 0.92)
    assert result is not None
    assert result.query == "what is RAG?"
    assert result.latency_ms == 250
    assert result.confidence == 0.92
    assert persist.call_count == 1
    assert ev._sampled_count == 1


@pytest.mark.asyncio
async def test_maybe_evaluate_with_ragas(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.01)
    ragas = AsyncMock(return_value={
        "faithfulness": 0.9,
        "answer_relevancy": 0.85,
        "context_precision": 0.8,
        "context_recall": 0.88,
        "answer_correctness": 0.86,
    })
    ev = OnlineEvaluator(sample_rate=0.05, ragas_eval_fn=ragas)
    result = await ev.maybe_evaluate("q", "a", ["c"], 100, 0.9)
    assert result is not None
    assert result.faithfulness == 0.9
    assert result.answer_relevancy == 0.85
    assert ev._evaluated_count == 1


@pytest.mark.asyncio
async def test_ragas_failure_falls_back_gracefully(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.01)
    ragas = AsyncMock(side_effect=RuntimeError("RAGAS not installed"))
    persist = AsyncMock()
    ev = OnlineEvaluator(sample_rate=0.05, ragas_eval_fn=ragas, persist_fn=persist)
    result = await ev.maybe_evaluate("q", "a", ["c"], 100, 0.5)
    # 失败兜底: sample 仍持久化,只是指标为 None
    assert result is not None
    assert result.faithfulness is None
    assert persist.call_count == 1


@pytest.mark.asyncio
async def test_persist_failure_does_not_break_main_flow(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.01)
    persist = AsyncMock(side_effect=RuntimeError("DB down"))
    ev = OnlineEvaluator(sample_rate=0.05, persist_fn=persist)
    result = await ev.maybe_evaluate("q", "a", ["c"], 100, 0.5)
    # 持久化失败但 sample 仍返回(主流程不受影响)
    assert result is not None
    assert result.query == "q"


@pytest.mark.asyncio
async def test_default_ragas_eval_returns_metrics():
    """默认 RAGAS(无真实库时)返回 mock 指标"""
    sample = OnlineEvalSample(
        query="q", answer="a", contexts=["c"], latency_ms=100, confidence=0.9
    )
    metrics = await default_ragas_eval(sample)
    assert "faithfulness" in metrics
    assert 0 <= metrics["faithfulness"] <= 1


@pytest.mark.asyncio
async def test_query_recent_metrics_with_data():
    async def fake_persist():
        return [
            OnlineEvalSample(
                query=f"q{i}", answer=f"a{i}", contexts=["c"],
                latency_ms=100 + i, confidence=0.8 + i * 0.01,
                faithfulness=0.9, answer_relevancy=0.85,
            )
            for i in range(20)
        ]
    result = await query_recent_online_metrics(fake_persist, limit=10)
    assert result["count"] == 10
    assert "faithfulness" in result["averages"]
    assert result["time_window"] == "24h"


@pytest.mark.asyncio
async def test_query_recent_metrics_empty():
    async def empty_persist():
        return []
    result = await query_recent_online_metrics(empty_persist)
    assert result["count"] == 0
    assert result["averages"] == {}


def test_stats_reports_ragas_success_rate():
    ev = OnlineEvaluator(sample_rate=0.05)
    ev._sampled_count = 10
    ev._evaluated_count = 8
    stats = ev.get_stats()
    assert stats["sampled_count"] == 10
    assert stats["evaluated_count"] == 8
    assert stats["ragas_success_rate"] == "80.00%"


def test_stats_handles_zero_samples():
    ev = OnlineEvaluator()
    stats = ev.get_stats()
    assert stats["sampled_count"] == 0
    assert stats["ragas_success_rate"] == "N/A"
