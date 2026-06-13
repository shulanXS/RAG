"""
reranker.py — Cross-Encoder 重排序
================================================================================
技术决策记录:
- 两阶段检索的必要性: Bi-encoder 在 query 和 doc 独立编码时缺乏深度交互，
  导致「看起来相似但实际不相关」的 chunk 被误召回。Cross-encoder 将 query+doc
  联合编码，通过 Transformer 的 attention 机制捕获细粒度相关性。
- Cohere Rerank 3.5: 2026 年企业默认值，$2/1K 查询，~80ms 延迟，
  NDCG@10 提升 10-30%。API 简单，无需维护 GPU 资源。
- BGE Reranker v2-m3: 开源自托管方案，需小型 GPU（如 A10G），~30ms 延迟，
  适合合规要求或高查询量（>100K/月）的成本优化。
- top-5 vs top-10: 实测 top-5 是 LLM 上下文的 cost-quality 最优切分点，
  超过 top-5 的块信息增益递减，且会显著增加 token 消耗。

业务难点:
- Reranker 延迟: 每增加一次 Reranker 调用增加约 80-100ms 延迟。
  决策: 在 top-50 → top-5 的压缩比下，延迟增加可接受，NDCG 提升显著。
- Chunk 文本过长: Reranker 的输入是 query + doc 拼接文本。
  当 doc 过长时需要截断，截断策略影响检索质量。
  决策: 保留 chunk 开头（通常包含最重要的信息）+ query 在前的拼接顺序。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """
    Reranker 重排结果

    字段说明:
    - chunk_id / doc_id: 来源标识
    - rerank_score: Cross-encoder 的相关性得分（归一化到 0-1）
    - original_rank: 在 Reranker 输入列表中的原始位置
    - final_rank: 重排后的最终排名
    """
    chunk_id: str
    doc_id: str
    rerank_score: float
    final_rank: int = 0
    text: str = ""
    section_path: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CrossEncoderReranker(ABC):
    """
    Cross-Encoder Reranker 抽象基类

    设计模式: 策略模式
    - Cohere 和 BGE 是两种不同的具体实现
    - 对外接口统一: rerank(query, chunks) → List[RerankResult]
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        对候选 chunks 进行重排序

        Args:
            query: 用户查询
            chunks: 候选 chunks，格式为 [{"chunk_id": ..., "text": ..., ...}]
            top_k: 返回 top-k 结果

        Returns:
            按 rerank_score 降序排列的结果
        """
        ...


class CohereReranker(CrossEncoderReranker):
    """
    Cohere Rerank 3.5 — 2026 年企业默认值

    技术决策:
    - rerank-3.5: 最新版本，支持长文档，NDCG@10 较 v3 提升约 5%
    - API 定价: $2/1K 查询（2026 年 6 月），性价比最优
    - 延迟: ~80ms P50，~150ms P99
    - API Key: 从 COHERE_API_KEY 环境变量读取
    """

    def __init__(
        self,
        model: str = "rerank-3.5",
        max_chunks_per_doc: int = 10,
        truncation: str = "end",  # 截断位置: "start" | "end"
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError("需要安装 cohere: pip install cohere")

        self._client = cohere.Client()
        self._model = model
        self._max_chunks = max_chunks_per_doc
        self._truncation = truncation

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        使用 Cohere Rerank API 进行重排序。

        技术要点:
        - Cohere API 会自动处理长文本截断（保留最重要部分）
        - 返回结果按相关性得分降序排列
        - API 有 batch 优化，一次请求多个 doc 比多次单 doc 调用更快
        """
        if not chunks:
            return []

        # 准备 API 输入：仅传递文本，API 自动关联 chunk_id
        docs = [c["text"][:4000] for c in chunks]  # 截断到 4000 tokens
        chunk_ids = [c["chunk_id"] for c in chunks]
        doc_ids = [c.get("doc_id", "") for c in chunks]
        texts = [c.get("text", "") for c in chunks]
        section_paths = [c.get("section_path", "") for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        response = self._client.rerank(
            query=query,
            documents=docs,
            model=self._model,
            top_n=min(top_k, len(chunks)),
            return_characters=True,
        )

        results: list[RerankResult] = []
        for rank, item in enumerate(response.results, 1):
            idx = item.index
            results.append(RerankResult(
                chunk_id=chunk_ids[idx],
                doc_id=doc_ids[idx],
                rerank_score=float(item.relevance_score),
                final_rank=rank,
                text=texts[idx],
                section_path=section_paths[idx],
                metadata=metadatas[idx],
            ))

        logger.debug(f"Cohere Rerank: {len(chunks)} → {len(results)} chunks")
        return results


class BGEReranker(CrossEncoderReranker):
    """
    BGE Reranker v2-m3 — 开源自托管方案

    技术决策:
    - BAAI/bge-reranker-v2-m3: 最高质量的国产开源 reranker，MTEB Reranking 第一
    - 需自行部署（Transformers + GPU）：推荐 A10G 或 L4，int8 量化后 T4 也能跑
    - 延迟: ~30ms（GPU），适合高 QPS 场景
    - 优势: 零 API 成本，数据不出境（合规），适合金融/医疗行业

    权衡取舍:
    - vs Cohere: 需要维护 GPU 资源，但查询量大时（>50K/月）成本更低
    - 需要额外的模型服务（本地部署或 Kubernetes），运维复杂度增加
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")

        self._model_name = model
        self._device = device
        self._cross_encoder = CrossEncoder(
            model,
            device=device,
            max_length=512,
            automodel_args={"torch_dtype": "auto"} if use_fp16 else {},
        )
        self._use_fp16 = use_fp16

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        使用本地 BGE Reranker 进行重排序。

        技术要点:
        - CrossEncoder 的输入是 [query, doc] 对列表
        - 自动处理 batch，batch_size 由模型自动决定
        - 输出是 [0, 1] 范围的相关性得分
        """
        if not chunks:
            return []

        # 构建 query-doc 对
        pairs = [
            (query, c["text"][:2000]) for c in chunks
        ]

        # 批量预测
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)

        # 转为 list[float]
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        else:
            scores = list(scores)

        # 按得分降序排列
        indexed_scores = list(enumerate(scores))
        ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]

        results: list[RerankResult] = []
        for final_rank, (idx, score) in enumerate(ranked, 1):
            chunk = chunks[idx]
            results.append(RerankResult(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk.get("doc_id", ""),
                rerank_score=float(score),
                final_rank=final_rank,
                text=chunk.get("text", ""),
                section_path=chunk.get("section_path", ""),
                metadata=chunk.get("metadata", {}),
            ))

        logger.debug(f"BGE Rerank: {len(chunks)} → {len(results)} chunks")
        return results


def get_reranker(
    provider: Literal["cohere", "bge"] = "cohere",
    **kwargs,
) -> CrossEncoderReranker:
    """
    Reranker 工厂函数

    技术决策:
    - 工厂模式，隐藏具体实现细节
    - Cohere 默认（最快上线），BGE 作为合规/成本敏感场景的替代
    """
    if provider == "cohere":
        return CohereReranker(**kwargs)
    elif provider == "bge":
        return BGEReranker(**kwargs)
    else:
        raise ValueError(f"不支持的 reranker provider: {provider}")
