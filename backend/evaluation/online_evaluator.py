"""
online_evaluator.py — Online Evaluation (实时评估)
================================================================================
技术决策记录:
- Online Evaluation 是生产环境的质量监控核心。
- 实时评估每次回答，采样入库，持续监控。
- 采样策略: 高延迟/低置信度 → 100% 评估；高置信度+快速 → 10% 采样。

业务价值:
- 实时监控 RAG 系统质量
- 识别检索/生成退化
- 为优化提供数据依据
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """
    评估样本

    字段说明:
    - query / answer / contexts: 原始数据
    - ragas_result: RAGAS 评估结果
    - latency_ms: 响应延迟
    - timestamp: 评估时间
    - sampled: 是否为采样评估
    - tags: 标签（用于分组分析）
    """
    query: str
    answer: str
    contexts: list[dict]
    ragas_result: dict | None = None
    latency_ms: float = 0.0
    timestamp: str = ""
    sampled: bool = False
    confidence: float = 0.0
    tags: dict = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """
    质量指标

    字段说明:
    - period: 统计周期
    - total_requests: 总请求数
    - sampled_requests: 采样评估数
    - avg_latency_ms: 平均延迟
    - avg_faithfulness: 平均忠实度
    - avg_relevancy: 平均相关性
    - avg_precision: 平均精度
    - avg_recall: 平均召回
    - pass_rate: 通过率
    - weakest_metric: 最弱指标
    """
    period: str
    total_requests: int = 0
    sampled_requests: int = 0
    avg_latency_ms: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    pass_rate: float = 0.0
    weakest_metric: str = ""
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class OnlineEvaluator:
    """
    在线评估系统

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 请求进来 → 记录 latency                                │
    │  2. should_sample() → 判断是否采样评估                   │
    │  3. evaluate_and_store() → 执行评估并存入缓冲区          │
    │  4. 定期 flush → 批量写入存储                            │
    │  5. get_quality_dashboard() → 生成质量仪表板            │
    └─────────────────────────────────────────────────────────────┘

    采样策略:
    - 高延迟 (>3s) → 100% 评估（延迟异常，需排查）
    - 低置信度 (confidence < 0.6) → 100% 评估（低质量，需关注）
    - 高延迟 + 低置信度 → 100% 评估
    - 高置信度 + 快速 → 10% 采样（持续监控）
    """

    def __init__(
        self,
        evaluator=None,
        latency_threshold_ms: float = 3000,
        confidence_threshold: float = 0.6,
        sample_rate: float = 0.1,
        buffer_size: int = 100,
    ):
        """
        Args:
            evaluator: RAGASEvaluator 实例
            latency_threshold_ms: 延迟阈值，超过则 100% 采样
            confidence_threshold: 置信度阈值，低于则 100% 采样
            sample_rate: 高质量请求的采样率
            buffer_size: 样本缓冲区大小
        """
        self._evaluator = evaluator
        self._latency_threshold = latency_threshold_ms
        self._confidence_threshold = confidence_threshold
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size

        self._sample_buffer: list[EvaluationSample] = []
        self._all_samples: list[EvaluationSample] = []
        self._total_requests = 0
        self._recent_latencies: list[float] = []

    def should_sample(
        self,
        latency_ms: float,
        confidence: float,
    ) -> bool:
        """
        智能采样决策

        Args:
            latency_ms: 请求延迟
            confidence: 置信度

        Returns:
            bool: 是否应该采样评估
        """
        if latency_ms > self._latency_threshold:
            self._total_requests += 1
            return True

        if confidence < self._confidence_threshold:
            self._total_requests += 1
            return True

        if confidence > 0.85 and latency_ms < 1000:
            return False

        # 计数: 每次调用都递增, 用于采样率决策
        self._total_requests += 1
        return self._total_requests % int(1 / self._sample_rate) == 0

    async def evaluate_and_store(
        self,
        query: str,
        answer: str,
        contexts: list[dict],
        latency_ms: float,
        confidence: float = 0.0,
        ground_truth: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """
        评估并存储样本

        Args:
            query: 用户查询
            answer: 生成答案
            contexts: 检索上下文
            latency_ms: 响应延迟
            confidence: 置信度
            ground_truth: 参考答案（可选）
            metadata: 额外元数据

        Returns:
            评估结果（仅采样时返回）
        """
        self._total_requests += 1
        self._recent_latencies.append(latency_ms)
        if len(self._recent_latencies) > 1000:
            self._recent_latencies = self._recent_latencies[-1000:]

        sampled = self.should_sample(latency_ms, confidence)

        if not sampled and ground_truth is None:
            return None

        sample = EvaluationSample(
            query=query,
            answer=answer,
            contexts=contexts,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            sampled=sampled,
            confidence=confidence,
            tags=metadata or {},
        )

        if self._evaluator is not None:
            try:
                report = await self._evaluator.evaluate(
                    question=query,
                    answer=answer,
                    retrieved_contexts=[c.get("text", "") for c in contexts],
                    ground_truth=ground_truth,
                )
                sample.ragas_result = {
                    "overall_pass": report.overall_pass,
                    "average_score": report.average_score,
                    "weakest_metric": report.weakest_metric,
                    "metrics": {
                        r.metric: {"score": r.score, "passed": r.passed}
                        for r in report.results
                    },
                }
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        self._sample_buffer.append(sample)
        self._all_samples.append(sample)

        if len(self._sample_buffer) >= self._buffer_size:
            self._flush_buffer()

        if sampled:
            return sample.ragas_result
        return None

    def _flush_buffer(self):
        """将缓冲区数据写入存储"""
        if not self._sample_buffer:
            return

        logger.info(f"Flushing {len(self._sample_buffer)} evaluation samples")

        try:
            self._persist_samples(self._sample_buffer)
        except Exception as e:
            logger.error(f"Failed to persist samples: {e}")

        self._sample_buffer.clear()

    def _persist_samples(self, samples: list[EvaluationSample]):
        """
        写入存储（P2.3: 用 SQLite 持久化，跨进程可见）。
        覆盖之前空实现 (online_evaluator.py:245)。
        """
        if not samples:
            return
        try:
            from backend.evaluation.eval_store import get_eval_store

            store = get_eval_store()
            run_id = f"online_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            store.save_run(
                run_id=run_id,
                started_at=datetime.utcnow().isoformat(),
                ended_at=datetime.utcnow().isoformat(),
                total_cases=len(samples),
                passed_cases=sum(1 for s in samples if s.ragas_result and s.ragas_result.get("overall_pass", False)),
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_context_precision=0.0,
                avg_context_recall=0.0,
                avg_answer_correctness=0.0,
                weakest_metric="",
                metadata={"source": "online_evaluator"},
            )
            sample_dicts = []
            for s in samples:
                m = (s.ragas_result or {}).get("metrics", {})
                sample_dicts.append({
                    "sample_id": s.sample_id,
                    "timestamp": s.timestamp,
                    "query": s.query,
                    "answer": s.answer,
                    "faithfulness": m.get("faithfulness", {}).get("score", 0.0),
                    "answer_relevancy": m.get("answer_relevancy", {}).get("score", 0.0),
                    "context_precision": m.get("context_precision", {}).get("score", 0.0),
                    "context_recall": m.get("context_recall", {}).get("score", 0.0),
                    "answer_correctness": m.get("answer_correctness", {}).get("score", 0.0),
                    "overall_pass": (s.ragas_result or {}).get("overall_pass", False),
                    "latency_ms": s.latency_ms,
                })
            store.save_samples(run_id, sample_dicts)
            logger.info(f"在线评估已持久化: run_id={run_id}, samples={len(sample_dicts)}")
        except Exception as e:
            logger.warning(f"_persist_samples 失败（已忽略）: {e}")

    def get_quality_dashboard(
        self,
        period: Literal["1h", "24h", "7d", "30d"] = "24h",
    ) -> QualityMetrics:
        """
        生成质量仪表板数据

        Args:
            period: 统计周期

        Returns:
            QualityMetrics: 质量指标汇总
        """
        period_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }

        delta = period_map.get(period, timedelta(hours=24))
        cutoff = datetime.utcnow() - delta

        recent_samples = [
            s for s in self._all_samples
            if datetime.fromisoformat(s.timestamp.replace("Z", "+00:00")) > cutoff
        ]

        if not recent_samples:
            return QualityMetrics(period=period, total_requests=self._total_requests)

        total = len(recent_samples)
        sampled = sum(1 for s in recent_samples if s.sampled)

        avg_latency = sum(s.latency_ms for s in recent_samples) / total

        latencies = sorted(s.latency_ms for s in recent_samples)
        p50 = latencies[int(len(latencies) * 0.5)]
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 20 else latencies[-1]
        p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 100 else latencies[-1]

        eval_samples = [s for s in recent_samples if s.ragas_result]
        passed = sum(1 for s in eval_samples if s.ragas_result.get("overall_pass", False))

        faithfulness_scores = [
            s.ragas_result["metrics"].get("faithfulness", {}).get("score", 0)
            for s in eval_samples
            if "faithfulness" in s.ragas_result.get("metrics", {})
        ]
        relevancy_scores = [
            s.ragas_result["metrics"].get("answer_relevancy", {}).get("score", 0)
            for s in eval_samples
            if "answer_relevancy" in s.ragas_result.get("metrics", {})
        ]
        precision_scores = [
            s.ragas_result["metrics"].get("context_precision", {}).get("score", 0)
            for s in eval_samples
            if "context_precision" in s.ragas_result.get("metrics", {})
        ]
        recall_scores = [
            s.ragas_result["metrics"].get("context_recall", {}).get("score", 0)
            for s in eval_samples
            if "context_recall" in s.ragas_result.get("metrics", {})
        ]

        metric_avgs = {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            "relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
            "precision": sum(precision_scores) / len(precision_scores) if precision_scores else 0,
            "recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
        }

        weakest = min(metric_avgs.items(), key=lambda x: x[1], default=("faithfulness", 0))

        return QualityMetrics(
            period=period,
            total_requests=total,
            sampled_requests=sampled,
            avg_latency_ms=avg_latency,
            avg_faithfulness=metric_avgs["faithfulness"],
            avg_relevancy=metric_avgs["relevancy"],
            avg_precision=metric_avgs["precision"],
            avg_recall=metric_avgs["recall"],
            pass_rate=passed / len(eval_samples) if eval_samples else 0,
            weakest_metric=weakest[0],
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
        )

    def get_trending(self, metric: str, days: int = 7) -> list[dict]:
        """获取指标趋势数据"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [
            s for s in self._all_samples
            if datetime.fromisoformat(s.timestamp.replace("Z", "+00:00")) > cutoff
            and s.ragas_result
        ]

        if not recent:
            return []

        from collections import defaultdict

        by_day = defaultdict(list)
        for s in recent:
            dt = datetime.fromisoformat(s.timestamp.replace("Z", "+00:00"))
            day = dt.strftime("%Y-%m-%d")
            score = s.ragas_result.get("metrics", {}).get(metric, {}).get("score", None)
            if score is not None:
                by_day[day].append(score)

        return [
            {"date": day, "avg_score": sum(scores) / len(scores), "count": len(scores)}
            for day, scores in sorted(by_day.items())
        ]
