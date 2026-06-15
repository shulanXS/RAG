"""
fusion.py — 多路检索结果融合
================================================================================
技术决策记录:
- RRF (Reciprocal Rank Fusion) 是 2026 年工业界的事实标准。
  核心优势: 无需 score 归一化，对不同量纲（BM25 vs cosine similarity）天然鲁棒。
- 为什么 k=60: 这是学术和工业界共同验证的最优值。
  k 值越大，各路算法的权重越均衡；k 值越小，排名靠前的结果权重越高。
  k=60 在「头部结果权重」和「各路均衡」之间取得最佳平衡。

业务难点:
- 排名冲突处理: 当两路算法给出完全不同的排名时，RRF 通过排名倒数平滑处理。
- 相关性信号冗余: 两路算法可能都检索到同一结果，RRF 天然处理这种情况
  （同一文档在两路中的排名叠加）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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


# 默认的 complexity → k 映射
# - simple:   k=30 (小 k → 看重头部; simple 查询通常 BM25 命中率高)
# - moderate: k=60 (默认)
# - complex:  k=90 (大 k → 各路均衡; complex 偏 dense 语义)
# - beyond_kb:k=60 (不区分, 不查知识库)
DEFAULT_K_BY_COMPLEXITY: dict[str, int] = {
    "simple": 30,
    "moderate": 60,
    "complex": 90,
    "beyond_kb": 60,
}


class DynamicRRFFusion:
    """
    按 query complexity 动态选 k 的 RRF 融合器。

    启发式:
    - 简单查询 (pronoun / 短词) → BM25 头部结果更相关 → 小 k 强调头部
    - 复杂查询 (multi-hop / analytical) → 需 dense 提权重 → 大 k 让 RRF 均衡融合
    - moderate / beyond_kb → 默认 k=60

    算法: score_RRF(d) = Σ 1/(k + rank_i(d))

    Args:
        k_default: 无 complexity 信号时的 fallback k (默认 60)
        k_by_complexity: complexity → k 覆盖映射, None 用默认
        enabled: config 开关, False 时退回 k_default
    """

    def __init__(
        self,
        k_default: int = 60,
        k_by_complexity: dict[str, int] | None = None,
        enabled: bool = True,
    ):
        self._k_default = k_default
        self._k_by_complexity = k_by_complexity or DEFAULT_K_BY_COMPLEXITY
        self._enabled = enabled

    def k_for_complexity(self, complexity: str | None) -> int:
        """根据 complexity 选 k (供上层 span attributes 记录)"""
        if not self._enabled or not complexity:
            return self._k_default
        return self._k_by_complexity.get(complexity, self._k_default)

    def fuse(
        self,
        result_sets: dict[str, list],
        complexity: str | None = None,
    ) -> list[FusionResult]:
        """
        执行动态 k 的 RRF 融合。

        Args:
            result_sets: 形如 {"bm25": [...], "dense": [...]}
            complexity: 路由出来的 query complexity (simple/moderate/complex/beyond_kb)

        Returns:
            按 RRF 得分降序的 FusionResult 列表
        """
        k = self.k_for_complexity(complexity)
        return self._rrf_fuse(result_sets, k)

    @staticmethod
    def _rrf_fuse(result_sets: dict[str, list], k: int) -> list[FusionResult]:
        """
        RRF 核心算法（内联，避免 RRFFusion 中间类）

        score_RRF(d) = Σ 1/(k + rank_i(d))
        """
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

                rrf_contrib = 1.0 / (k + rank)
                doc_scores[chunk_id]["rrf_contribution"] += rrf_contrib
                doc_scores[chunk_id]["sources"].append(source_name)
                doc_scores[chunk_id]["individual_scores"][source_name] = getattr(
                    result, "score", 1.0 / (k + rank)
                )

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
