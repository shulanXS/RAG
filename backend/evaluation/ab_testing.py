"""
ab_testing.py — A/B Testing Framework (A/B 测试框架)
================================================================================
技术决策记录:
- A/B Testing 是生产环境优化的核心工具。
- 支持不同配置、不同模型、不同检索策略的对比实验。
- 实验维度: chunk_size、reranker、self-reflection、LLM 选择等。

支持实验:
- 不同 chunk_size 配置对比
- 不同 reranker 对比
- 有/无 self-reflection 对比
- DeepSeek vs Claude 生成质量对比
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """
    实验变体

    字段说明:
    - id: 变体 ID (e.g., "control", "treatment_a")
    - name: 变体名称
    - config: 变体配置参数
    - traffic_ratio: 流量比例 (0.0-1.0)
    - description: 变体描述
    """
    id: str
    name: str
    config: dict
    traffic_ratio: float = 0.5
    description: str = ""


@dataclass
class Experiment:
    """
    A/B 实验

    字段说明:
    - id / name: 实验标识
    - variants: 实验变体列表
    - metric: 评估指标
    - status: 实验状态
    - start_time / end_time: 实验时间范围
    """
    id: str
    name: str
    variants: list[Variant]
    metric: str
    status: Literal["draft", "running", "paused", "completed"] = "draft"
    start_time: str = ""
    end_time: str = ""
    description: str = ""
    min_sample_size: int = 100


@dataclass
class ExperimentOutcome:
    """
    实验结果记录

    字段说明:
    - experiment_id / variant_id: 关联实验和变体
    - user_id: 用户标识（用于去重）
    - metric_value: 指标值
    - metadata: 额外元数据
    - timestamp: 记录时间
    """
    experiment_id: str
    variant_id: str
    user_id: str
    metric_value: float
    metadata: dict = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class ExperimentResult:
    """
    实验结果

    字段说明:
    - experiment_id: 实验 ID
    - variant_results: 各变体的统计结果
    - winner: 胜出变体
    - statistical_significance: 统计显著性
    - recommendation: 建议
    """
    experiment_id: str
    variant_results: dict[str, dict]
    winner: str | None = None
    p_value: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    recommendation: str = ""
    sample_size: int = 0


class ABTestManager:
    """
    A/B 测试管理器

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. create_experiment() → 创建实验                          │
    │  2. assign_variant() → 用户流量分配                        │
    │  3. record_outcome() → 记录实验结果                        │
    │  4. analyze() → 统计分析 + 显著性检验                      │
    │  5. conclude() → 得出结论，发布推荐                        │
    └─────────────────────────────────────────────────────────────┘

    设计要点:
    - 确定性流量分配（基于 user_id hash，确保同一用户始终分到同一变体）
    - 支持多指标评估
    - 统计显著性计算
    """

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}
        self._outcomes: dict[str, list[ExperimentOutcome]] = {}
        self._assignments: dict[str, dict[str, str]] = {}
        self._online_evaluator = None

    def create_experiment(
        self,
        name: str,
        variants: list[Variant],
        metric: str,
        description: str = "",
        min_sample_size: int = 100,
    ) -> Experiment:
        """
        创建 A/B 实验

        Args:
            name: 实验名称
            variants: 变体列表
            metric: 评估指标 (faithfulness|lateness|...)
            description: 实验描述
            min_sample_size: 最小样本量

        Returns:
            Experiment: 创建的实验
        """
        experiment_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:8]

        experiment = Experiment(
            id=experiment_id,
            name=name,
            variants=variants,
            metric=metric,
            status="draft",
            description=description,
            min_sample_size=min_sample_size,
        )

        self._experiments[experiment_id] = experiment
        self._outcomes[experiment_id] = []
        self._assignments[experiment_id] = {}

        logger.info(f"Created experiment: {name} ({experiment_id})")
        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """启动实验"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False

        if exp.status == "running":
            return False

        exp.status = "running"
        exp.start_time = datetime.utcnow().isoformat()
        logger.info(f"Started experiment: {experiment_id}")
        return True

    def assign_variant(
        self,
        user_id: str,
        experiment_id: str,
    ) -> str | None:
        """
        为用户分配实验变体（确定性分配）

        Args:
            user_id: 用户 ID
            experiment_id: 实验 ID

        Returns:
            分配的变体 ID，未找到则返回 None
        """
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != "running":
            return None

        if user_id in self._assignments[experiment_id]:
            return self._assignments[experiment_id][user_id]

        hash_input = f"{experiment_id}_{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0

        cumulative = 0.0
        for variant in exp.variants:
            cumulative += variant.traffic_ratio
            if normalized < cumulative:
                self._assignments[experiment_id][user_id] = variant.id
                logger.debug(f"Assigned user {user_id} to variant {variant.id}")
                return variant.id

        if exp.variants:
            self._assignments[experiment_id][user_id] = exp.variants[0].id
            return exp.variants[0].id

        return None

    def record_outcome(
        self,
        user_id: str,
        experiment_id: str,
        variant_id: str,
        metric_value: float,
        metadata: dict | None = None,
    ) -> bool:
        """
        记录实验结果

        Args:
            user_id: 用户 ID
            experiment_id: 实验 ID
            variant_id: 变体 ID
            metric_value: 指标值
            metadata: 额外元数据

        Returns:
            是否记录成功
        """
        if experiment_id not in self._experiments:
            return False

        outcome = ExperimentOutcome(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            metric_value=metric_value,
            metadata=metadata or {},
            timestamp=datetime.utcnow().isoformat(),
        )

        self._outcomes[experiment_id].append(outcome)
        return True

    def analyze(self, experiment_id: str) -> ExperimentResult | None:
        """
        分析实验结果

        Args:
            experiment_id: 实验 ID

        Returns:
            ExperimentResult: 分析结果
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        outcomes = self._outcomes.get(experiment_id, [])
        variant_results: dict[str, dict] = {}

        for variant in exp.variants:
            variant_outcomes = [o for o in outcomes if o.variant_id == variant.id]
            if not variant_outcomes:
                variant_results[variant.id] = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }
                continue

            values = [o.metric_value for o in variant_outcomes]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5

            variant_results[variant.id] = {
                "count": len(values),
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "values": values,
            }

        winner_id = None
        if variant_results:
            winner_id = max(
                variant_results.items(),
                key=lambda x: x[1]["mean"],
                default=(None, {"mean": 0}),
            )[0]

        p_value = self._calculate_p_value(variant_results)

        recommendation = ""
        if winner_id:
            winner_data = variant_results[winner_id]
            if winner_data["count"] >= exp.min_sample_size:
                recommendation = f"变体 '{winner_id}' 表现最佳 (mean={winner_data['mean']:.4f})，建议推广"

        return ExperimentResult(
            experiment_id=experiment_id,
            variant_results=variant_results,
            winner=winner_id,
            p_value=p_value,
            recommendation=recommendation,
            sample_size=len(outcomes),
        )

    def _calculate_p_value(self, variant_results: dict[str, dict]) -> float:
        """简单计算 p-value（z-test）"""
        if len(variant_results) < 2:
            return 1.0

        variants = list(variant_results.values())
        a = variants[0]
        b = variants[1] if len(variants) > 1 else variants[0]

        if a["count"] < 2 or b["count"] < 2:
            return 1.0

        pooled_std = (
            ((a["count"] - 1) * a["std"] ** 2 + (b["count"] - 1) * b["std"] ** 2) /
            (a["count"] + b["count"] - 2)
        ) ** 0.5

        if pooled_std == 0:
            return 1.0

        z_score = abs(a["mean"] - b["mean"]) / (pooled_std * (1 / a["count"] + 1 / b["count"]) ** 0.5)

        import math
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z_score / 2 ** 0.5)))

        return min(1.0, max(0.0, p_value))

    def conclude_experiment(self, experiment_id: str) -> bool:
        """结束实验"""
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != "running":
            return False

        exp.status = "completed"
        exp.end_time = datetime.utcnow().isoformat()

        result = self.analyze(experiment_id)
        if result:
            logger.info(
                f"Experiment {experiment_id} concluded. "
                f"Winner: {result.winner}, p-value: {result.p_value:.4f}"
            )

        return True

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """获取实验信息"""
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[Experiment]:
        """列出所有实验"""
        return list(self._experiments.values())


# 内置实验工厂方法
class ExperimentFactory:
    """常用实验的工厂方法"""

    @staticmethod
    def create_chunk_size_experiment() -> list[Variant]:
        """不同 chunk_size 配置对比"""
        return [
            Variant(
                id="chunk_256",
                name="Chunk Size 256",
                config={"chunk_size": 256},
                traffic_ratio=0.33,
                description="小 chunk，精确但上下文不足",
            ),
            Variant(
                id="chunk_512",
                name="Chunk Size 512",
                config={"chunk_size": 512},
                traffic_ratio=0.34,
                description="中等 chunk，平衡",
            ),
            Variant(
                id="chunk_1024",
                name="Chunk Size 1024",
                config={"chunk_size": 1024},
                traffic_ratio=0.33,
                description="大 chunk，上下文丰富但可能稀释",
            ),
        ]

    @staticmethod
    def create_llm_comparison_experiment() -> list[Variant]:
        """DeepSeek vs Claude 生成质量对比"""
        return [
            Variant(
                id="deepseek",
                name="DeepSeek",
                config={"llm_provider": "deepseek", "model": "deepseek-chat"},
                traffic_ratio=0.5,
                description="DeepSeek (低成本)",
            ),
            Variant(
                id="claude",
                name="Claude",
                config={"llm_provider": "anthropic", "model": "claude-3-7-sonnet-20250620"},
                traffic_ratio=0.5,
                description="Claude (高质量)",
            ),
        ]

    @staticmethod
    def create_self_reflection_experiment() -> list[Variant]:
        """有/无 self-reflection 对比"""
        return [
            Variant(
                id="no_reflection",
                name="Without Self-Reflection",
                config={"enable_self_reflection": False},
                traffic_ratio=0.5,
                description="直接生成答案",
            ),
            Variant(
                id="with_reflection",
                name="With Self-Reflection",
                config={"enable_self_reflection": True},
                traffic_ratio=0.5,
                description="生成后进行自我反思",
            ),
        ]
