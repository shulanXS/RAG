"""
ragas_metrics.py — RAGAS 五指标评估
================================================================================
技术决策记录:
- RAGAS (Retrieval-Augmented Generation Assessment) 是学术界和工业界公认的
  RAG 评估标准。五大指标覆盖了检索和生成两个阶段的质量。
- 为什么需要评估: RAG 系统的质量不能靠「感觉」。没有量化指标，
  无法发现检索质量退化（corpus 变化 / embedding 模型更新 / 文档格式变化）。
- CI/CD 集成: 每次代码变更或索引更新都必须运行评估套件，
  指标下降超过 threshold 则触发告警或自动回滚。

业务难点:
- 标注数据集: RAGAS 的 context_recall 需要 ground_truth context，
  构建标注数据集是最大的工程成本。
- LLM-as-Judge: 评估本身使用 LLM 判断，存在评估偏差。
  缓解: 使用多个评估模型取平均，使用专用评估 prompt。

RAGAS 五大指标定义:
1. Faithfulness: 答案是否被检索上下文支撑？
2. Answer Relevancy: 答案是否直接回答问题？
3. Context Precision: top-K 上下文中相关块的比例？
4. Context Recall: 检索上下文覆盖必要信息的程度？
5. Answer Correctness: 与 ground truth 的一致性？
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    """
    RAGAS 评估结果

    字段说明:
    - metric: 指标名称
    - score: 分数 (0-1)
    - passed: 是否通过阈值
    - threshold: 判定阈值
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

    字段说明:
    - overall_pass: 是否全部通过
    - results: 各指标详细结果
    - average_score: 平均得分
    - weakest_metric: 得分最低的指标
    """
    overall_pass: bool
    results: list[RAGASResult]
    average_score: float = 0.0
    weakest_metric: str = ""
    timestamp: str = ""


class RAGASEvaluator:
    """
    RAGAS 评估器

    技术要点:
    - 支持离线评估（batch 模式）
    - 阈值判定: 每项指标有独立阈值
    - 生成详细报告: 含分数、判定结果、改进建议

    RAGAS 五指标详解:
    1. Faithfulness: LLM 判断答案中有多少陈述被检索上下文支持。
       公式: supported_statements / total_statements
    2. Answer Relevancy: LLM 判断答案与问题的相关程度（0-1）。
       原理: 如果答案很好地回答了问题，从答案中提取的子问题应该与原问题相似。
    3. Context Precision: 使用 NDCG@K 衡量 top-K 中相关块的比例。
       公式: Σ(G_i / i) / Σ(G_i) 其中 G_i 是第 i 位是否相关（1/0）
    4. Context Recall: 检索到的上下文覆盖了多少必要信息（需要 ground truth）。
       公式: ground_truth_statements / retrieved_statements
    5. Answer Correctness: 综合评估答案与 ground truth 的一致性。
       公式: LLM 判断 answer 与 ground_truth 的相似度 (0-1)

    权衡取舍:
    - context_recall 需要 ground truth 标注，成本高。
      决策: 先用 faithfulness 和 answer_relevancy（有参考答案即可），
      等团队成熟后再加入 context_recall。
    """

    # RAGAS 推荐阈值（2026 生产经验值）
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
            llm_client: LLM client（用于 LLM-as-Judge 评估）
            thresholds: 指标阈值配置
        """
        self._llm = llm_client
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS

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
        if self._llm is None:
            logger.warning("没有 LLM client，无法执行 RAGAS 评估")
            return self._dummy_report()

        results: list[RAGASResult] = []

        # 1. Faithfulness
        faithfulness = await self._eval_faithfulness(question, answer, retrieved_contexts)
        results.append(faithfulness)

        # 2. Answer Relevancy
        relevancy = await self._eval_answer_relevancy(question, answer)
        results.append(relevancy)

        # 3. Context Precision
        precision = await self._eval_context_precision(question, retrieved_contexts)
        results.append(precision)

        # 4. Context Recall（需要 ground truth）
        if ground_truth:
            recall = await self._eval_context_recall(
                ground_truth, retrieved_contexts
            )
            results.append(recall)

        # 5. Answer Correctness（需要 ground truth）
        if ground_truth:
            correctness = await self._eval_answer_correctness(
                answer, ground_truth
            )
            results.append(correctness)

        # 计算整体结果
        overall_pass = all(r.passed for r in results)
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        weakest = min(results, key=lambda r: r.score) if results else None

        return EvaluationReport(
            overall_pass=overall_pass,
            results=results,
            average_score=avg_score,
            weakest_metric=weakest.metric if weakest else "",
        )

    async def evaluate_batch(
        self,
        test_cases: list[dict],
    ) -> dict:
        """
        批量评估

        Args:
            test_cases: 测试用例列表，格式:
              [{"question": ..., "ground_truth_answer": ..., ...}, ...]

        Returns:
            批量评估报告: {"total": N, "passed": M, "pass_rate": 0.x, "per_case": [...]}
        """
        from datetime import datetime

        reports = []
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

        # 找出最弱的指标
        all_metrics = {}
        for r in reports:
            for metric_result in r.results:
                if metric_result.metric not in all_metrics:
                    all_metrics[metric_result.metric] = []
                all_metrics[metric_result.metric].append(metric_result.score)

        weakest = min(
            all_metrics.items(),
            key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 1.0,
        )

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "average_score": sum(avg_scores) / len(avg_scores) if avg_scores else 0,
            "weakest_metric": weakest[0],
            "weakest_score": sum(weakest[1]) / len(weakest[1]) if weakest[1] else 0,
            "per_case": [
                {"question": tc["question"], "passed": r.overall_pass, "score": r.average_score}
                for tc, r in zip(test_cases, reports)
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # 各指标评估实现
    # -------------------------------------------------------------------------

    async def _eval_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RAGASResult:
        """Faithfulness: 答案是否被上下文支撑？"""
        prompt = f"""评估答案的忠实性。

问题: {question}

检索到的上下文:
{chr(10).join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))}

答案: {answer}

请判断答案中有多少陈述被检索上下文支撑。
评估原则:
- 如果答案中的每个关键陈述都能在上下文中找到直接支持，faithfulness = 1.0
- 如果答案中有部分陈述无法从上下文中找到支持，faithfulness 按比例降低
- 如果答案完全是编造的（上下文完全不支持），faithfulness = 0.0

请以 JSON 格式输出:
{{"score": 0.0-1.0, "reasoning": "判断理由（1-2句话）"}}
"""
        try:
            import json
            response = await self._llm.generate_async(prompt, max_tokens=256, temperature=0.1)
            data = json.loads(response.strip())
            score = float(data.get("score", 0.5))
            return RAGASResult(
                metric="faithfulness",
                score=score,
                passed=score >= self._thresholds["faithfulness"],
                threshold=self._thresholds["faithfulness"],
                details=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Faithfulness 评估失败: {e}")
            return self._error_result("faithfulness")

    async def _eval_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> RAGASResult:
        """Answer Relevancy: 答案是否直接回答问题？"""
        prompt = f"""评估答案与问题的相关程度。

问题: {question}
答案: {answer}

评估原则:
- 如果答案直接、完整地回答了问题，relevancy = 1.0
- 如果答案部分相关但不完整，relevancy 按比例降低
- 如果答案完全跑题，relevancy = 0.0

请以 JSON 格式输出:
{{"score": 0.0-1.0, "reasoning": "判断理由"}}
"""
        try:
            import json
            response = await self._llm.generate_async(prompt, max_tokens=128, temperature=0.1)
            data = json.loads(response.strip())
            score = float(data.get("score", 0.5))
            return RAGASResult(
                metric="answer_relevancy",
                score=score,
                passed=score >= self._thresholds["answer_relevancy"],
                threshold=self._thresholds["answer_relevancy"],
                details=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Answer Relevancy 评估失败: {e}")
            return self._error_result("answer_relevancy")

    async def _eval_context_precision(
        self,
        question: str,
        contexts: list[str],
    ) -> RAGASResult:
        """Context Precision: top-K 中相关块的比例（NDCG@K）"""
        if not contexts:
            return self._error_result("context_precision")

        prompt = f"""评估每个检索到的上下文与问题的相关程度。

问题: {question}

检索到的上下文:
{chr(10).join(f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts))}

请判断每个上下文是否与问题相关（1=相关，0=不相关）。
相关性定义: 如果这个上下文对回答问题有帮助，则为相关。

请以 JSON 格式输出:
{{"relevance_scores": [1, 0, 1, ...]}}  # 每个上下文的相关性得分（0或1）
"""
        try:
            import json
            response = await self._llm.generate_async(prompt, max_tokens=256, temperature=0.1)
            data = json.loads(response.strip())
            scores = data.get("relevance_scores", [])

            # NDCG@K 计算
            ndcg = self._ndcg_at_k(scores, k=len(scores))
            return RAGASResult(
                metric="context_precision",
                score=ndcg,
                passed=ndcg >= self._thresholds["context_precision"],
                threshold=self._thresholds["context_precision"],
                details=f"NDCG@{len(contexts)} = {ndcg:.3f}",
            )
        except Exception as e:
            logger.warning(f"Context Precision 评估失败: {e}")
            return self._error_result("context_precision")

    async def _eval_context_recall(
        self,
        ground_truth: str,
        contexts: list[str],
    ) -> RAGASResult:
        """Context Recall: 检索上下文覆盖了多少必要信息"""
        prompt = f"""评估检索到的上下文覆盖了参考答案中的多少信息。

参考答案: {ground_truth}

检索到的上下文:
{chr(10).join(f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts))}

评估原则:
- 如果检索上下文完整覆盖了参考答案的所有关键信息，recall = 1.0
- 如果只覆盖了部分信息，recall 按比例降低
- 如果完全没有覆盖，recall = 0.0

请以 JSON 格式输出:
{{"score": 0.0-1.0, "reasoning": "判断理由"}}
"""
        try:
            import json
            response = await self._llm.generate_async(prompt, max_tokens=256, temperature=0.1)
            data = json.loads(response.strip())
            score = float(data.get("score", 0.5))
            return RAGASResult(
                metric="context_recall",
                score=score,
                passed=score >= self._thresholds["context_recall"],
                threshold=self._thresholds["context_recall"],
                details=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Context Recall 评估失败: {e}")
            return self._error_result("context_recall")

    async def _eval_answer_correctness(
        self,
        answer: str,
        ground_truth: str,
    ) -> RAGASResult:
        """Answer Correctness: 与参考答案的一致性"""
        prompt = f"""评估生成答案与参考答案的一致性。

参考答案: {ground_truth}
生成答案: {answer}

评估原则:
- 如果生成答案与参考答案高度一致，correctness = 1.0
- 如果存在部分差异，correctness 按比例降低
- 如果完全不一致，correctness = 0.0

请以 JSON 格式输出:
{{"score": 0.0-1.0, "reasoning": "判断理由"}}
"""
        try:
            import json
            response = await self._llm.generate_async(prompt, max_tokens=256, temperature=0.1)
            data = json.loads(response.strip())
            score = float(data.get("score", 0.5))
            return RAGASResult(
                metric="answer_correctness",
                score=score,
                passed=score >= self._thresholds["answer_correctness"],
                threshold=self._thresholds["answer_correctness"],
                details=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Answer Correctness 评估失败: {e}")
            return self._error_result("answer_correctness")

    @staticmethod
    def _ndcg_at_k(scores: list[int], k: int | None = None) -> float:
        """计算 NDCG@K"""
        if not scores:
            return 0.0
        k = k or len(scores)
        scores = scores[:k]

        # DCG
        dcg = sum((2**s - 1) / (i + 1) for i, s in enumerate(scores))

        # IDCG（理想情况下所有相关都在前面）
        ideal_scores = sorted(scores, reverse=True)
        idcg = sum((2**s - 1) / (i + 1) for i, s in enumerate(ideal_scores))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _error_result(self, metric: str) -> RAGASResult:
        return RAGASResult(
            metric=metric,
            score=0.0,
            passed=False,
            threshold=self._thresholds.get(metric, 0.5),
            details="评估执行失败",
        )

    def _dummy_report(self) -> EvaluationReport:
        return EvaluationReport(
            overall_pass=False,
            results=[],
            average_score=0.0,
            weakest_metric="",
        )
