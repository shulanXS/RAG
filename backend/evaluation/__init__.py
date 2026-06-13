"""
evaluation 模块 — 评估与可观测性
"""

from backend.evaluation.ragas_metrics import RAGASEvaluator, RAGASResult, EvaluationReport
from backend.evaluation.deepeval_tests import RAGTestSuite
from backend.evaluation.test_dataset import get_test_dataset, get_test_dataset_by_category
from backend.evaluation.online_evaluator import OnlineEvaluator, EvaluationSample, QualityMetrics
from backend.evaluation.ab_testing import (
    ABTestManager,
    Experiment,
    Variant,
    ExperimentResult,
    ExperimentFactory,
)

__all__ = [
    "RAGASEvaluator",
    "RAGASResult",
    "EvaluationReport",
    "RAGTestSuite",
    "get_test_dataset",
    "get_test_dataset_by_category",
    "OnlineEvaluator",
    "EvaluationSample",
    "QualityMetrics",
    "ABTestManager",
    "Experiment",
    "Variant",
    "ExperimentResult",
    "ExperimentFactory",
]
