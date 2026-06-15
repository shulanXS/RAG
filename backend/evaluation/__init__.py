"""评估与监控 — RAGAS + EvalStore（详见 ARCHITECTURE.md）

Phase2-2.1: Online Evaluator 已删除 — 在线跑 RAGAS LLM judge 成本失控，
且无人维护，删除以聚焦离线 RAGAS 评估（CI 跑）。
"""
from backend.evaluation.ragas_metrics import RAGASEvaluator, RAGASResult, EvaluationReport
from backend.evaluation.test_dataset import get_test_dataset, get_test_dataset_by_category
