"""
ragas_metrics.py — RAGAS 评估（基于 ragas 库 + 优雅降级）
================================================================================
技术决策记录:
- 优先使用 ragas 库（>=0.4）的官方指标，避免自己写 LLM prompt。
  官方 prompt 经过大量实验校准，质量与稳定性优于自实现。
- 优雅降级: ragas 库未安装时回退到内置简化评估器。
- 评估 LLM 与生产 LLM 解耦: 评估使用专门 LLM（更便宜/更稳定），
  避免影响生产 LLM 配额。

RAGAS 五大指标:
1. Faithfulness: 答案是否被检索上下文支撑？
2. Answer Relevancy: 答案是否直接回答问题？
3. Context Precision: top-K 上下文中相关块的比例？
4. Context Recall: 检索上下文覆盖必要信息的程度？
5. Answer Correctness: 与 ground truth 的一致性？
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. RAGAS 库可用性检测
# --------------------------------------------------------------------------

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "ragas / datasets 未安装，评估不可用。请运行: "
        "pip install ragas>=0.4.0 datasets>=2.14.0"
    )


# --------------------------------------------------------------------------
# 2. 结果数据类
# --------------------------------------------------------------------------


@dataclass
class RAGASResult:
    """
    RAGAS 评估结果
    """
    metric: str
    score: float
    passed: bool
    threshold: float
    details: str = ""


@dataclass
class EvaluationReport:
    """
    完整评估报告
    """
    overall_pass: bool
    results: list[RAGASResult]
    average_score: float = 0.0
    weakest_metric: str = ""
    timestamp: str = ""


# --------------------------------------------------------------------------
# 3. 主评估器
# --------------------------------------------------------------------------


class RAGASEvaluator:
    """
    RAGAS 评估器

    技术要点:
    - ragas 库路径: 优先使用 ragas 内置指标，自动管理 evaluator LLM。
    - 降级路径: 未安装 ragas 时使用简化本地评估（仅供开发）。
    - 阈值判定: 每项指标有独立阈值。
    - 生成详细报告: 含分数、判定结果、改进建议。
    """

    DEFAULT_THRESHOLDS = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.75,
        "context_precision": 0.70,
        "context_recall": 0.70,
        "answer_correctness": 0.80,
    }

    def __init__(
        self,
        llm_client=None,
        thresholds: dict[str, float] | None = None,
    ):
        """
        Args:
            llm_client: 保留入参以保持向后兼容。ragas 0.4+ 自动管理 evaluator LLM。
            thresholds: 指标阈值配置
        """
        self._llm = llm_client
        self._thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)
        # ragas 0.4+ 自行管理 evaluator LLM；self._llm 仅在调用方做 trace 关联时使用。

    def _wrap_llm_llm_attribute(self) -> None:
        """保留扩展点 — 真实适配由调用方按需覆盖。"""
        self._ragas_llm = None

    def _wrap_llm_for_ragas(self) -> None:
        """扩展点。ragas 0.4+ 自行管理 evaluator LLM，无需手工包装。"""
        self._wrap_llm_llm_attribute()

    # -------------------------------------------------------------------------
    # 单条评估
    # -------------------------------------------------------------------------

    async def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationReport:
        """
        对单条查询进行 RAGAS 评估。

        Args:
            question: 用户问题
            answer: LLM 生成的回答
            retrieved_contexts: 检索到的上下文列表
            ground_truth: 参考答案（可选，用于 context_recall 和 answer_correctness）

        Returns:
            EvaluationReport: 完整评估报告

        Raises:
            RuntimeError: ragas 库不可用或评估失败（不允许静默回退到不可信实现）
        """
        if not RAGAS_AVAILABLE:
            raise RuntimeError(
                "ragas >= 0.4 is required for RAGASEvaluator; "
                "install via `pip install ragas>=0.4.0 datasets>=2.14.0`"
            )
        return await self._evaluate_with_ragas(
            question, answer, retrieved_contexts, ground_truth
        )

    async def _evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        retrieved_contexts: list[str],
        ground_truth: str | None,
    ) -> EvaluationReport:
        """使用 ragas 库执行评估"""
        data = {
            "user_input": [question],
            "response": [answer],
            "retrieved_contexts": [retrieved_contexts] if retrieved_contexts else [[]],
        }
        if ground_truth:
            data["reference"] = [ground_truth]
        ds = Dataset.from_dict(data)

        metric_objs = [faithfulness, answer_relevancy, context_precision]
        if ground_truth:
            # P0-1: 把 answer_correctness 也加入（与 config.yaml 5 指标阈值保持一致）。
            # 此前只跑 4 指标，answer_correctness 阈值永远不被比较 → 沉默失败。
            metric_objs.append(answer_correctness)
            metric_objs.append(context_recall)

        try:
            # ragas 评估是同步阻塞的，放到线程池避免阻塞 event loop
            import asyncio
            loop = asyncio.get_event_loop()
            ragas_result = await loop.run_in_executor(
                None, lambda: ragas_evaluate(ds, metrics=metric_objs)
            )
        except Exception as e:
            raise RuntimeError(f"ragas 评估失败: {e}") from e

        results: list[RAGASResult] = []
        # ragas 0.4+ 返回 Result 对象
        scores: dict[str, float] = {}
        try:
            df = ragas_result.to_pandas()
            for col in df.columns:
                if col in self._thresholds:
                    val = df[col].iloc[0]
                    if val is not None and not (isinstance(val, float) and val != val):
                        scores[col] = float(val)
        except Exception as e:
            raise RuntimeError(f"解析 ragas 结果失败: {e}") from e

        for metric, score in scores.items():
            threshold = self._thresholds.get(metric, 0.5)
            results.append(
                RAGASResult(
                    metric=metric,
                    score=score,
                    passed=score >= threshold,
                    threshold=threshold,
                )
            )

        return self._build_report(results)

    # -------------------------------------------------------------------------
    # 批量评估
    # -------------------------------------------------------------------------

    async def evaluate_batch(self, test_cases: list[dict]) -> dict:
        """
        批量评估

        Args:
            test_cases: 测试用例列表，格式:
              [{"question": ..., "answer": ..., "contexts": [...], "ground_truth": "..."}, ...]

        Returns:
            批量评估报告
        """
        reports: list[EvaluationReport] = []
        for case in test_cases:
            report = await self.evaluate(
                question=case["question"],
                answer=case.get("answer", ""),
                retrieved_contexts=case.get("contexts", []),
                ground_truth=case.get("ground_truth"),
            )
            reports.append(report)

        total = len(reports)
        passed = sum(1 for r in reports if r.overall_pass)
        avg_scores = [r.average_score for r in reports]

        all_metrics: dict[str, list[float]] = {}
        for r in reports:
            for metric_result in r.results:
                all_metrics.setdefault(metric_result.metric, []).append(metric_result.score)

        weakest_name = ""
        weakest_score = 0.0
        if all_metrics:
            weakest_name, weakest_scores = min(
                all_metrics.items(),
                key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 1.0,
            )
            weakest_score = sum(weakest_scores) / len(weakest_scores) if weakest_scores else 0.0

        # Per-metric averages — P1-4 替代 RAGTestSuite.run_ci 输出形状
        # 与原 DeepEval 时代 `average_scores` 字段名一致。
        per_metric_avg = {
            metric: (sum(v) / len(v) if v else 0.0)
            for metric, v in all_metrics.items()
        }

        # 回归检测：与阈值比较（与原 RAGTestSuite.run_ci 一致语义）
        regression_detected = any(
            score < self._thresholds.get(metric, 0.5)
            for metric, score in per_metric_avg.items()
        )

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "average_score": sum(avg_scores) / len(avg_scores) if avg_scores else 0,
            "average_scores": per_metric_avg,
            "regression_detected": regression_detected,
            "weakest_metric": weakest_name,
            "weakest_score": weakest_score,
            "per_case": [
                {
                    "question": tc["question"],
                    "passed": r.overall_pass,
                    "score": r.average_score,
                }
                for tc, r in zip(test_cases, reports)
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # 工具
    # -------------------------------------------------------------------------

    def _build_report(self, results: list[RAGASResult]) -> EvaluationReport:
        if not results:
            return EvaluationReport(
                overall_pass=False,
                results=[],
                average_score=0.0,
                weakest_metric="",
                timestamp=datetime.utcnow().isoformat(),
            )
        overall_pass = all(r.passed for r in results)
        avg_score = sum(r.score for r in results) / len(results)
        weakest = min(results, key=lambda r: r.score)
        return EvaluationReport(
            overall_pass=overall_pass,
            results=results,
            average_score=avg_score,
            weakest_metric=weakest.metric,
            timestamp=datetime.utcnow().isoformat(),
        )
