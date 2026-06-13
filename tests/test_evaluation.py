"""
test_evaluation.py — Evaluation 模块测试
"""

import pytest
from backend.evaluation.online_evaluator import (
    OnlineEvaluator,
    EvaluationSample,
    QualityMetrics,
)
from backend.evaluation.ab_testing import (
    ABTestManager,
    Experiment,
    Variant,
    ExperimentFactory,
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


class TestABTestManager:
    """A/B Testing 测试"""

    def test_create_experiment(self):
        """创建实验测试"""
        manager = ABTestManager()

        variants = [
            Variant(id="a", name="Variant A", config={"key": "a"}, traffic_ratio=0.5),
            Variant(id="b", name="Variant B", config={"key": "b"}, traffic_ratio=0.5),
        ]

        experiment = manager.create_experiment(
            name="test_experiment",
            variants=variants,
            metric="faithfulness",
        )

        assert isinstance(experiment, Experiment)
        assert experiment.name == "test_experiment"
        assert len(experiment.variants) == 2
        assert experiment.status == "draft"

    def test_assign_variant_consistent(self):
        """变体分配一致性测试"""
        manager = ABTestManager()

        variants = [
            Variant(id="a", name="A", config={}, traffic_ratio=0.5),
            Variant(id="b", name="B", config={}, traffic_ratio=0.5),
        ]
        exp = manager.create_experiment(
            name="test", variants=variants, metric="latency"
        )
        manager.start_experiment(exp.id)

        # 同一用户始终分到同一变体
        variant1 = manager.assign_variant("user123", exp.id)
        variant2 = manager.assign_variant("user123", exp.id)

        assert variant1 == variant2

    def test_assign_variant_deterministic(self):
        """变体分配确定性测试"""
        manager = ABTestManager()

        variants = [
            Variant(id="a", name="A", config={}, traffic_ratio=0.5),
            Variant(id="b", name="B", config={}, traffic_ratio=0.5),
        ]
        exp = manager.create_experiment(
            name="test", variants=variants, metric="latency"
        )
        manager.start_experiment(exp.id)

        # 不同用户应该能分到不同变体（基于 hash）
        variants_assigned = set()
        for i in range(10):
            v = manager.assign_variant(f"user_{i}", exp.id)
            variants_assigned.add(v)

        assert len(variants_assigned) >= 1

    def test_record_outcome(self):
        """记录实验结果测试"""
        manager = ABTestManager()

        variants = [
            Variant(id="a", name="A", config={}, traffic_ratio=0.5),
            Variant(id="b", name="B", config={}, traffic_ratio=0.5),
        ]
        exp = manager.create_experiment(
            name="test", variants=variants, metric="latency"
        )
        manager.start_experiment(exp.id)

        success = manager.record_outcome(
            user_id="user1",
            experiment_id=exp.id,
            variant_id="a",
            metric_value=0.85,
        )

        assert success is True

    def test_analyze_experiment(self):
        """实验结果分析测试"""
        manager = ABTestManager()

        variants = [
            Variant(id="a", name="A", config={}, traffic_ratio=0.5),
            Variant(id="b", name="B", config={}, traffic_ratio=0.5),
        ]
        exp = manager.create_experiment(
            name="test", variants=variants, metric="faithfulness"
        )
        manager.start_experiment(exp.id)

        for i in range(5):
            manager.record_outcome("user1", exp.id, "a", 0.8 + i * 0.02)
            manager.record_outcome("user2", exp.id, "b", 0.75 + i * 0.01)

        result = manager.analyze(exp.id)

        assert result is not None
        assert result.experiment_id == exp.id
        assert "a" in result.variant_results
        assert "b" in result.variant_results
        assert result.variant_results["a"]["count"] == 5
        assert result.variant_results["b"]["count"] == 5

    def test_conclude_experiment(self):
        """结束实验测试"""
        manager = ABTestManager()

        variants = [Variant(id="a", name="A", config={}, traffic_ratio=1.0)]
        exp = manager.create_experiment(
            name="test", variants=variants, metric="latency"
        )
        manager.start_experiment(exp.id)

        manager.record_outcome("user1", exp.id, "a", 0.9)

        success = manager.conclude_experiment(exp.id)

        assert success is True
        assert manager.get_experiment(exp.id).status == "completed"


class TestExperimentFactory:
    """实验工厂测试"""

    def test_chunk_size_variants(self):
        """Chunk size 实验变体"""
        variants = ExperimentFactory.create_chunk_size_experiment()

        assert len(variants) == 3
        assert all(v.traffic_ratio > 0 for v in variants)
        ids = [v.id for v in variants]
        assert "chunk_256" in ids
        assert "chunk_512" in ids
        assert "chunk_1024" in ids

    def test_llm_comparison_variants(self):
        """LLM 对比实验变体"""
        variants = ExperimentFactory.create_llm_comparison_experiment()

        assert len(variants) == 2
        ids = [v.id for v in variants]
        assert "deepseek" in ids
        assert "claude" in ids

    def test_self_reflection_variants(self):
        """Self-reflection 实验变体"""
        variants = ExperimentFactory.create_self_reflection_experiment()

        assert len(variants) == 2
        ids = [v.id for v in variants]
        assert "no_reflection" in ids
        assert "with_reflection" in ids


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
