"""
evaluation 模块 — 评估与可观测性
================================================================================
技术决策记录:
- RAGAS 为主评估器，DeepEval 为 CI pytest 风格备用。
- OnlineEvaluator 在生产采样评估，触发存储到 SQLite (eval_store)。
- 移除项 (P0): ab_testing.py + shadow_testing.py — 没接入主流程，作品集 demo
  不会用；FAANG 真实生产用 EP/FB Experiment 平台，不在 RAG repo 内做。
"""

from backend.evaluation.ragas_metrics import RAGASEvaluator, RAGASResult, EvaluationReport
from backend.evaluation.deepeval_tests import RAGTestSuite
from backend.evaluation.test_dataset import get_test_dataset, get_test_dataset_by_category
from backend.evaluation.online_evaluator import OnlineEvaluator, EvaluationSample, QualityMetrics

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
]
