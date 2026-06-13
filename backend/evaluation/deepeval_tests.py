"""
deepeval_tests.py — DeepEval pytest 测试套件
================================================================================
技术决策记录:
- DeepEval 的核心价值: 将 RAG 评估集成到 CI/CD 流程中，像写单元测试一样写评估。
- 5 指标对应 RAG 的核心质量维度：
  Hallucination（幻觉）= faithfulness 的对立面
  Answer Relevancy = RAGAS answer_relevancy
  Contextual Precision = RAGAS context_precision
  Contextual Recall = RAGAS context_recall
  Factual Consistency = faithfulness

业务难点:
- 测试用例维护: 随着产品迭代，测试用例需要持续更新。
  解决方案: 将测试用例与代码分离，存储在 test_dataset.py 中。
- CI/CD 集成: DeepEval 可直接集成到 GitHub Actions。
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# DeepEval 依赖检查
try:
    from deepeval.metrics import (
        HallucinationMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_run import test_case
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("DeepEval 未安装，跳过 DeepEval 测试。运行: pip install deepeval")


class RAGTestSuite:
    """
    RAG 评估测试套件（pytest 风格）

    用法:
        suite = RAGTestSuite()
        suite.run_all(
            question="X是什么？",
            answer="X是...",
            contexts=["上下文1", "上下文2"],
            ground_truth="参考答案"
        )
    """

    def __init__(self, thresholds: dict | None = None):
        """
        Args:
            thresholds: 各指标阈值配置，如 {"faithfulness": 0.85}
        """
        self._thresholds = thresholds or {
            "faithfulness": 0.85,
            "answer_relevancy": 0.75,
            "context_precision": 0.70,
            "context_recall": 0.70,
            "hallucination": 0.05,
        }
        self._results: list[dict] = []

    def run_all(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        category: Literal["simple", "moderate", "difficult"] = "simple",
    ) -> dict:
        """
        运行完整评估套件

        Args:
            question: 用户问题
            answer: LLM 生成的回答
            contexts: 检索到的上下文
            ground_truth: 参考答案（用于 context_recall）
            category: 查询复杂度分类

        Returns:
            评估结果字典
        """
        if not DEEPEVAL_AVAILABLE:
            logger.warning("DeepEval 不可用，返回占位结果")
            return self._dummy_result(question, answer, category)

        results = {}

        # 1. Faithfulness
        try:
            faithfulness_metric = FaithfulnessMetric(
                threshold=self._thresholds["faithfulness"],
            )
            faithfulness_metric.measure(
                LLMResponse=answer,
                context=contexts,
            )
            results["faithfulness"] = {
                "score": faithfulness_metric.score,
                "passed": faithfulness_metric.success,
                "reason": faithfulness_metric.reason,
            }
        except Exception as e:
            logger.warning(f"Faithfulness 测试失败: {e}")
            results["faithfulness"] = {"score": 0, "passed": False, "reason": str(e)}

        # 2. Answer Relevancy
        try:
            relevancy_metric = AnswerRelevancyMetric(
                threshold=self._thresholds["answer_relevancy"],
            )
            relevancy_metric.measure(
                LLMResponse=answer,
                input=question,
            )
            results["answer_relevancy"] = {
                "score": relevancy_metric.score,
                "passed": relevancy_metric.success,
                "reason": relevancy_metric.reason,
            }
        except Exception as e:
            logger.warning(f"Answer Relevancy 测试失败: {e}")
            results["answer_relevancy"] = {"score": 0, "passed": False, "reason": str(e)}

        # 3. Contextual Precision
        try:
            precision_metric = ContextualPrecisionMetric(
                threshold=self._thresholds["context_precision"],
            )
            precision_metric.measure(
                LLMResponse=answer,
                context=contexts,
                query=question,
            )
            results["context_precision"] = {
                "score": precision_metric.score,
                "passed": precision_metric.success,
                "reason": precision_metric.reason,
            }
        except Exception as e:
            logger.warning(f"Context Precision 测试失败: {e}")
            results["context_precision"] = {"score": 0, "passed": False, "reason": str(e)}

        # 4. Contextual Recall（需要 ground truth）
        if ground_truth:
            try:
                recall_metric = ContextualRecallMetric(
                    threshold=self._thresholds["context_recall"],
                )
                recall_metric.measure(
                    LLMResponse=answer,
                    expected_output=ground_truth,
                    context=contexts,
                )
                results["context_recall"] = {
                    "score": recall_metric.score,
                    "passed": recall_metric.success,
                    "reason": recall_metric.reason,
                }
            except Exception as e:
                logger.warning(f"Context Recall 测试失败: {e}")
                results["context_recall"] = {"score": 0, "passed": False, "reason": str(e)}

        # 5. Hallucination
        try:
            halluc_metric = HallucinationMetric(
                threshold=self._thresholds["hallucination"],
            )
            halluc_metric.measure(
                LLMResponse=answer,
                context=contexts,
            )
            results["hallucination"] = {
                "score": halluc_metric.score,
                "passed": halluc_metric.success,
                "reason": halluc_metric.reason,
            }
        except Exception as e:
            logger.warning(f"Hallucination 测试失败: {e}")
            results["hallucination"] = {"score": 0, "passed": False, "reason": str(e)}

        # 汇总
        all_passed = all(r.get("passed", False) for r in results.values())
        avg_score = sum(r.get("score", 0) for r in results.values()) / len(results)

        result = {
            "question": question,
            "answer": answer[:100] + "...",
            "category": category,
            "all_passed": all_passed,
            "average_score": avg_score,
            "metrics": results,
        }

        self._results.append(result)
        return result

    def run_ci(
        self,
        test_cases: list[dict],
        regression_threshold: float = 0.05,
    ) -> dict:
        """
        CI/CD 集成模式：运行所有测试用例并生成回归报告

        Args:
            test_cases: 测试用例列表
            regression_threshold: 指标下降超过此阈值则触发告警

        Returns:
            CI 报告: {"total": N, "passed": M, "regression_detected": bool}
        """
        results = []
        for case in test_cases:
            result = self.run_all(
                question=case["question"],
                answer=case.get("answer", ""),
                contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth"),
                category=case.get("category", "simple"),
            )
            results.append(result)

        total = len(results)
        passed = sum(1 for r in results if r["all_passed"])
        avg_scores = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "hallucination": [],
        }

        for r in results:
            for metric, data in r["metrics"].items():
                if metric in avg_scores:
                    avg_scores[metric].append(data["score"])

        metric_averages = {
            k: sum(v) / len(v) if v else 0
            for k, v in avg_scores.items()
        }

        # 回归检测：与历史数据对比（简化版：与阈值对比）
        regression_detected = False
        for metric, avg in metric_averages.items():
            threshold = self._thresholds.get(metric, 0.5)
            if avg < threshold - regression_threshold:
                regression_detected = True
                logger.warning(f"指标回归: {metric} = {avg:.3f} < {threshold} - {regression_threshold}")

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "average_scores": metric_averages,
            "regression_detected": regression_detected,
            "should_fail_ci": regression_detected or (passed / total < 0.8),
        }

    def _dummy_result(self, question: str, answer: str, category: str) -> dict:
        return {
            "question": question,
            "answer": answer[:100] + "...",
            "category": category,
            "all_passed": False,
            "average_score": 0.0,
            "metrics": {
                "faithfulness": {"score": 0, "passed": False, "reason": "DeepEval unavailable"},
                "answer_relevancy": {"score": 0, "passed": False, "reason": "DeepEval unavailable"},
                "context_precision": {"score": 0, "passed": False, "reason": "DeepEval unavailable"},
                "hallucination": {"score": 0, "passed": False, "reason": "DeepEval unavailable"},
            },
        }
