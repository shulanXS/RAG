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
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "ragas / datasets 未安装，将回退到简化评估。请运行: "
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
            llm_client: LLM client（ragas 会自动包装为 evaluator LLM；降级路径下用于本地评估）
            thresholds: 指标阈值配置
        """
        self._llm = llm_client
        self._thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)
        self._use_ragas = RAGAS_AVAILABLE

        if self._use_ragas and llm_client is not None:
            self._wrap_llm_for_ragas()

    def _wrap_llm_llm_attribute(self) -> None:
        """ragas >= 0.2 推荐使用 Langchain LLM 接口。
        简单处理：把项目内 LLMClient 的 generator_client 包装成 ragas 可用的形式。
        此处仅做软绑定 — 实际指标调用的内部细节交给 ragas 库。
        """
        # 复杂适配放到使用方按需覆盖；此处保留扩展点
        self._ragas_llm = None

    def _wrap_llm_for_ragas(self) -> None:
        """尝试将内 LLMClient 包装为 ragas 可用的 LLM 接口。"""
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
        对单条查询进行 RAGAS 评估

        Args:
            question: 用户问题
            answer: LLM 生成的回答
            retrieved_contexts: 检索到的上下文列表
            ground_truth: 参考答案（可选，用于 context_recall 和 answer_correctness）

        Returns:
            EvaluationReport: 完整评估报告
        """
        if self._use_ragas and self._llm is not None:
            return await self._evaluate_with_ragas(
                question, answer, retrieved_contexts, ground_truth
            )
        return await self._evaluate_fallback(
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
            metric_objs.append(context_recall)

        try:
            # ragas 评估是同步阻塞的，放到线程池避免阻塞 event loop
            import asyncio
            loop = asyncio.get_event_loop()
            ragas_result = await loop.run_in_executor(
                None, lambda: ragas_evaluate(ds, metrics=metric_objs)
            )
        except Exception as e:
            logger.warning(f"ragas 评估失败，回退到本地评估: {e}")
            return await self._evaluate_fallback(
                question, answer, retrieved_contexts, ground_truth
            )

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
            logger.warning(f"解析 ragas 结果失败: {e}")
            return await self._evaluate_fallback(
                question, answer, retrieved_contexts, ground_truth
            )

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

    async def _evaluate_fallback(
        self,
        question: str,
        answer: str,
        retrieved_contexts: list[str],
        ground_truth: str | None,
    ) -> EvaluationReport:
        """降级路径：使用 LLM 做 LLM-as-Judge（与原实现一致）"""
        if self._llm is None:
            logger.warning("没有 LLM client，无法执行 RAGAS 评估")
            return self._build_report([])

        results: list[RAGASResult] = []

        faithfulness_score = await self._llm_score_faithfulness(
            question, answer, retrieved_contexts
        )
        results.append(
            RAGASResult(
                metric="faithfulness",
                score=faithfulness_score,
                passed=faithfulness_score >= self._thresholds.get("faithfulness", 0.85),
                threshold=self._thresholds.get("faithfulness", 0.85),
            )
        )

        relevancy_score = await self._llm_score_relevancy(question, answer)
        results.append(
            RAGASResult(
                metric="answer_relevancy",
                score=relevancy_score,
                passed=relevancy_score >= self._thresholds.get("answer_relevancy", 0.75),
                threshold=self._thresholds.get("answer_relevancy", 0.75),
            )
        )

        precision_score = await self._llm_score_context_precision(question, retrieved_contexts)
        results.append(
            RAGASResult(
                metric="context_precision",
                score=precision_score,
                passed=precision_score >= self._thresholds.get("context_precision", 0.70),
                threshold=self._thresholds.get("context_precision", 0.70),
            )
        )

        if ground_truth:
            recall_score = await self._llm_score_context_recall(ground_truth, retrieved_contexts)
            results.append(
                RAGASResult(
                    metric="context_recall",
                    score=recall_score,
                    passed=recall_score >= self._thresholds.get("context_recall", 0.70),
                    threshold=self._thresholds.get("context_recall", 0.70),
                )
            )

        return self._build_report(results)

    # -------------------------------------------------------------------------
    # 简化的 LLM-as-Judge（降级路径）
    # -------------------------------------------------------------------------

    async def _llm_score_faithfulness(
        self, question: str, answer: str, contexts: list[str]
    ) -> float:
        if not contexts:
            return 0.0
        ctx_text = "\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
        prompt = f"""评估答案的忠实性（faithfulness，0-1 区间）。
问题: {question}
上下文: {ctx_text}
答案: {answer}
仅返回 0-1 的小数，不要其他内容。"""
        try:
            response = await self._llm.generate_async(prompt, max_tokens=16, temperature=0.1)
            return max(0.0, min(1.0, float(response.strip())))
        except Exception as e:
            logger.warning(f"Faithfulness LLM 评估失败: {e}")
            return 0.0

    async def _llm_score_relevancy(self, question: str, answer: str) -> float:
        prompt = f"""评估答案与问题的相关度（0-1 区间）。
问题: {question}
答案: {answer}
仅返回 0-1 的小数，不要其他内容。"""
        try:
            response = await self._llm.generate_async(prompt, max_tokens=16, temperature=0.1)
            return max(0.0, min(1.0, float(response.strip())))
        except Exception as e:
            logger.warning(f"Relevancy LLM 评估失败: {e}")
            return 0.0

    async def _llm_score_context_precision(
        self, question: str, contexts: list[str]
    ) -> float:
        if not contexts:
            return 0.0
        ctx_text = "\n".join(f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts))
        prompt = f"""评估检索上下文的相关性比例（precision，0-1 区间）。
问题: {question}
上下文列表:
{ctx_text}
仅返回 0-1 的小数。"""
        try:
            response = await self._llm.generate_async(prompt, max_tokens=16, temperature=0.1)
            return max(0.0, min(1.0, float(response.strip())))
        except Exception as e:
            logger.warning(f"Context Precision LLM 评估失败: {e}")
            return 0.0

    async def _llm_score_context_recall(
        self, ground_truth: str, contexts: list[str]
    ) -> float:
        ctx_text = "\n".join(f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts))
        prompt = f"""评估检索上下文对参考答案的覆盖率（recall，0-1 区间）。
参考答案: {ground_truth}
上下文列表:
{ctx_text}
仅返回 0-1 的小数。"""
        try:
            response = await self._llm.generate_async(prompt, max_tokens=16, temperature=0.1)
            return max(0.0, min(1.0, float(response.strip())))
        except Exception as e:
            logger.warning(f"Context Recall LLM 评估失败: {e}")
            return 0.0

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

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "average_score": sum(avg_scores) / len(avg_scores) if avg_scores else 0,
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
