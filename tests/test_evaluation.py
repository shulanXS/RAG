"""
test_evaluation.py — Evaluation 模块测试
"""

import pytest
from backend.evaluation.online_evaluator import (
    OnlineEvaluator,
    EvaluationSample,
    QualityMetrics,
)
from backend.evaluation.ragas_metrics import RAGASEvaluator


class TestOnlineEvaluator:
    """Online Evaluator 测试"""

    def test_sampling_decision_high_latency(self):
        """高延迟采样测试"""
        evaluator = OnlineEvaluator(
            latency_threshold_ms=3000,
            confidence_threshold=0.6,
            sample_rate=0.1,
        )

        assert evaluator.should_sample(latency_ms=4000, confidence=0.8) is True

    def test_sampling_decision_low_confidence(self):
        """低置信度采样测试"""
        evaluator = OnlineEvaluator(
            latency_threshold_ms=3000,
            confidence_threshold=0.6,
            sample_rate=0.1,
        )

        assert evaluator.should_sample(latency_ms=500, confidence=0.3) is True

    def test_sampling_decision_high_quality(self):
        """高质量不采样测试"""
        evaluator = OnlineEvaluator(
            latency_threshold_ms=3000,
            confidence_threshold=0.6,
            sample_rate=0.1,
        )

        assert evaluator.should_sample(latency_ms=500, confidence=0.9) is False

    def test_quality_dashboard_empty(self):
        """空数据仪表板测试"""
        evaluator = OnlineEvaluator()

        dashboard = evaluator.get_quality_dashboard("24h")

        assert isinstance(dashboard, QualityMetrics)
        assert dashboard.total_requests == 0

    def test_sample_counter(self):
        """采样计数器测试"""
        evaluator = OnlineEvaluator(sample_rate=0.1)

        assert evaluator._total_requests == 0

        evaluator.should_sample(500, 0.5)
        assert evaluator._total_requests == 1

        evaluator.should_sample(500, 0.5)
        assert evaluator._total_requests == 2


class TestQualityMetrics:
    """Quality Metrics 数据结构测试"""

    def test_metrics_creation(self):
        metrics = QualityMetrics(
            period="24h",
            total_requests=100,
            sampled_requests=15,
            avg_latency_ms=1200.0,
            avg_faithfulness=0.87,
            avg_relevancy=0.78,
            pass_rate=0.85,
            weakest_metric="context_precision",
        )

        assert metrics.period == "24h"
        assert metrics.total_requests == 100
        assert metrics.pass_rate == 0.85
        assert metrics.weakest_metric == "context_precision"


class TestEvaluationSample:
    """Evaluation Sample 数据结构测试"""

    def test_sample_creation(self):
        sample = EvaluationSample(
            query="测试查询",
            answer="测试答案",
            contexts=[{"chunk_id": "c1", "text": "上下文"}],
            latency_ms=1500.0,
            sampled=True,
            confidence=0.75,
        )

        assert sample.query == "测试查询"
        assert sample.latency_ms == 1500.0
        assert sample.sampled is True
        assert sample.confidence == 0.75
