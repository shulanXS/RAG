"""
fusion.py — 多路检索结果融合
================================================================================
技术决策记录:
- RRF vs 加权平均: RRF (Reciprocal Rank Fusion) 是 2026 年工业界的事实标准。
  核心优势: 无需 score 归一化，对不同量纲（BM25 vs cosine similarity）天然鲁棒。
  加权平均需要将不同算法的得分归一化到同一量纲，实际操作中容易出现偏差。
- 为什么 k=60: 这是学术和工业界共同验证的最优值。
  k 值越大，各路算法的权重越均衡；k 值越小，排名靠前的结果权重越高。
  k=60 在「头部结果权重」和「各路均衡」之间取得最佳平衡。
- 场景化权重: 对于精确匹配查询（包含合同号/SKU），BM25 权重应更高；
  对于语义理解查询，dense 权重应更高。当前实现使用固定权重，
  进阶方案: 根据查询类型动态调整权重。

业务难点:
- 排名冲突处理: 当两路算法给出完全不同的排名时，RRF 通过排名倒数平滑处理。
- 相关性信号冗余: 两路算法可能都检索到同一结果，RRF 天然处理这种情况
  （同一文档在两路中的排名叠加）。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """
    融合后的检索结果

    字段说明:
    - chunk_id / doc_id: 来源标识
    - fused_score: RRF 融合得分
    - rank: 最终排名
    - sources: 来自哪些检索路（如 ["bm25", "dense"]）
    - individual_scores: 各路原始得分
    """
    chunk_id: str
    doc_id: str
    fused_score: float
    rank: int
    text: str = ""
    section_path: str = ""
    sources: list[str] = field(default_factory=list)
    individual_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class FusionStrategy(ABC):
    """检索结果融合策略抽象基类"""

    @abstractmethod
    def fuse(
        self,
        result_sets: dict[str, list],
    ) -> list[FusionResult]:
        """融合多路检索结果"""
        ...


class RRFFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF) — 2026 年工业界默认方案

    算法原理:
    score_RRF(d) = Σ 1/(k + rank_i(d))

    其中:
    - d: 目标文档
    - k: 融合参数（默认 60）
    - rank_i(d): 文档 d 在第 i 路检索中的排名

    技术决策:
    - k=60 是经过大量实验验证的最优值（Van Gysel et al., 2011 及大量工业实践）
    - k 越小，头部结果权重越高；k 越大，各路越均衡
    - 当 rank_i(d) 相同时（即同一结果在不同路的排名相同），
      该结果的 RRF 得分最高（这是我们想要的）
    """

    def __init__(self, k: int = 60):
        self._k = k

    def fuse(
        self,
        result_sets: dict[str, list],
    ) -> list[FusionResult]:
        """
        执行 RRF 融合。

        Args:
            result_sets: 形如 {"bm25": [BM25Result, ...], "dense": [VectorResult, ...]}
                        各路结果列表必须已按各自得分降序排列

        Returns:
            按 RRF 得分降序排列的融合结果
        """
        # 计算每个 chunk 在各路的排名
        doc_scores: dict[str, dict] = {}

        for source_name, results in result_sets.items():
            for rank, result in enumerate(results, 1):
                chunk_id = result.chunk_id
                if chunk_id not in doc_scores:
                    doc_scores[chunk_id] = {
                        "doc_id": result.doc_id,
                        "text": getattr(result, "text", ""),
                        "section_path": getattr(result, "section_path", ""),
                        "metadata": getattr(result, "metadata", {}),
                        "rrf_contribution": 0.0,
                        "sources": [],
                        "individual_scores": {},
                    }

                # RRF 贡献 = 1 / (k + rank)
                rrf_contrib = 1.0 / (self._k + rank)
                doc_scores[chunk_id]["rrf_contribution"] += rrf_contrib
                doc_scores[chunk_id]["sources"].append(source_name)
                doc_scores[chunk_id]["individual_scores"][source_name] = getattr(
                    result, "score", 1.0 / (self._k + rank)
                )

        # 按 RRF 得分降序排序
        ranked = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["rrf_contribution"],
            reverse=True,
        )

        fusion_results: list[FusionResult] = []
        for rank, (chunk_id, data) in enumerate(ranked, 1):
            fusion_results.append(FusionResult(
                chunk_id=chunk_id,
                doc_id=data["doc_id"],
                fused_score=data["rrf_contribution"],
                rank=rank,
                text=data["text"],
                section_path=data["section_path"],
                sources=data["sources"],
                individual_scores=data["individual_scores"],
                metadata=data["metadata"],
            ))

        logger.debug(f"RRF 融合完成: {len(result_sets)} 路检索 → {len(fusion_results)} 个唯一文档")
        return fusion_results


class WeightedFusion(FusionStrategy):
    """
    加权得分融合 — 备选方案

    技术决策:
    - 适用于两路算法得分分布已知且稳定的情况
    - 需要手动归一化（Min-Max 或 Z-Score）
    - 实测效果不如 RRF 稳定（当一路算法得分分布突变时，权重失效）
    - 保留作为 RRF 的备选方案
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        normalize: bool = True,
    ):
        """
        Args:
            weights: 各路权重，如 {"bm25": 0.5, "dense": 0.5}
            normalize: 是否对各路得分做 Min-Max 归一化
        """
        self._weights = weights or {"default": 1.0}
        self._normalize = normalize

    def fuse(
        self,
        result_sets: dict[str, list],
    ) -> list[FusionResult]:
        if self._normalize:
            result_sets = self._min_max_normalize(result_sets)

        doc_scores: dict[str, dict] = {}
        for source_name, results in result_sets.items():
            weight = self._weights.get(source_name, 1.0)
            for result in results:
                chunk_id = result.chunk_id
                if chunk_id not in doc_scores:
                    doc_scores[chunk_id] = {
                        "doc_id": result.doc_id,
                        "text": getattr(result, "text", ""),
                        "section_path": getattr(result, "section_path", ""),
                        "metadata": getattr(result, "metadata", {}),
                        "weighted_score": 0.0,
                        "sources": [],
                        "individual_scores": {},
                    }
                score = getattr(result, "score", 0.0)
                doc_scores[chunk_id]["weighted_score"] += score * weight
                doc_scores[chunk_id]["sources"].append(source_name)
                doc_scores[chunk_id]["individual_scores"][source_name] = score

        ranked = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True,
        )

        fusion_results: list[FusionResult] = []
        for rank, (chunk_id, data) in enumerate(ranked, 1):
            fusion_results.append(FusionResult(
                chunk_id=chunk_id,
                doc_id=data["doc_id"],
                fused_score=data["weighted_score"],
                rank=rank,
                text=data["text"],
                section_path=data["section_path"],
                sources=data["sources"],
                individual_scores=data["individual_scores"],
                metadata=data["metadata"],
            ))
        return fusion_results

    @staticmethod
    def _min_max_normalize(
        result_sets: dict[str, list],
    ) -> dict[str, list]:
        """对各路结果做 Min-Max 归一化"""
        normalized: dict[str, list] = {}
        for source_name, results in result_sets.items():
            if not results:
                normalized[source_name] = results
                continue

            scores = [getattr(r, "score", 0.0) for r in results]
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s

            if span < 1e-9:
                # 所有得分相同，无需归一化
                normalized[source_name] = results
                continue

            for result in results:
                result.score = (getattr(result, "score", 0.0) - min_s) / span

            normalized[source_name] = results
        return normalized
