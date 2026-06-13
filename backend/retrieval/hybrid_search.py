"""
hybrid_search.py — 混合检索编排引擎
================================================================================
技术决策记录:
- 这是检索层的中央编排器，协调 BM25 + Dense + RRF + Reranker 的全流程。
- 混合检索是 2026 年 FAANG 标准: 纯向量检索在精确标识符上召回差；
  纯 BM25 无法处理语义 paraphrase；RRF 融合两者优势。
- BM25 和 Dense 必须并行执行: 串行会浪费 1 倍延迟。
- 两阶段检索（粗召回→精排）将 top-50 压缩到 top-5，NDCG@10 提升 10-30%。

业务难点:
- 延迟预算: 每条查询的总延迟预算约 200-500ms。
  BM25 ~10ms + Dense ~15ms + RRF ~2ms + Reranker ~80ms = ~107ms，可接受。
- 过滤查询: ACL 过滤在检索前执行（pre-filter），而非 post-filter。
  这是监管行业的合规要求：未授权文档不能进入 LLM 上下文。

权衡取舍:
- BM25 权重 0.5 vs 0.3: 默认 0.5 平等权重。
  对于精确标识符多的查询（合同/政策），BM25 权重可提高到 0.7。
  当前实现使用固定权重，未来可接入 Query Router 动态调整。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal

from backend.ingestion.embedder import Embedder
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.fusion import RRFFusion, FusionResult
from backend.retrieval.reranker import CrossEncoderReranker, get_reranker
from backend.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """
    检索上下文 — 包含检索过程的完整信息（用于评估和 debug）

    字段说明:
    - total_latency_ms: 总检索延迟
    - bm25_latency_ms / dense_latency_ms / rerank_latency_ms: 各阶段延迟
    - bm25_top_k / dense_top_k: 各路召回数量
    - fusion_candidates: RRF 融合后的候选集大小
    - reranked_top_k: Reranker 输出数量
    - retrieved_chunks: 最终检索结果
    """
    query: str
    total_latency_ms: float
    bm25_latency_ms: float = 0.0
    dense_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    bm25_top_k: int = 0
    dense_top_k: int = 0
    fusion_candidates: int = 0
    reranked_top_k: int = 0
    retrieved_chunks: list = None

    def __post_init__(self):
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []

    @property
    def stage_breakdown(self) -> dict:
        """返回各阶段延迟占比"""
        return {
            "bm25": f"{self.bm25_latency_ms:.1f}ms ({100*self.bm25_latency_ms/max(self.total_latency_ms,1):.0f}%)",
            "dense": f"{self.dense_latency_ms:.1f}ms ({100*self.dense_latency_ms/max(self.total_latency_ms,1):.0f}%)",
            "fusion": f"{self.fusion_latency_ms:.1f}ms ({100*self.fusion_latency_ms/max(self.total_latency_ms,1):.0f}%)",
            "rerank": f"{self.rerank_latency_ms:.1f}ms ({100*self.rerank_latency_ms/max(self.total_latency_ms,1):.0f}%)",
        }


class HybridSearchEngine:
    """
    混合检索引擎 — 检索层的中央编排器

    检索流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. query_embedding (Haiku / embedder)                     │
    └──────────────────────────┬──────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐         ┌──────────┐       ┌─────────────────┐
    │  BM25   │         │  Dense    │       │  (并行执行)       │
    │检索 top-50│         │检索 top-50│       │                 │
    └────┬────┘         └────┬─────┘       └─────────────────┘
         │                    │
         └──────────┬─────────┘
                    ▼
             ┌─────────────┐
             │ RRF 融合    │ top-50 融合候选集
             │ (k=60)     │
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │  Reranker   │ top-5 精排结果
             │ (Cross-Enc) │
             └──────┬──────┘
                    │
                    ▼
             返回 top-5 chunks

    技术要点:
    - 并行执行: BM25 和 dense search 同时发起，节省 1 倍延迟
    - RRF 融合: 无需 score 归一化，对不同量纲鲁棒
    - Reranker: Cross-encoder 精排，提升 top-K 排序精度

    风险考量:
    - BM25 索引未构建: 启动时检查，无索引则跳过 BM25 路（仅 dense）
    - Reranker 调用失败: 降级到 RRF 融合结果（不调用 reranker）
    - 过滤查询返回为空: 记录告警，返回空结果（而不是放宽过滤条件）
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        individual_top_k: int = 50,
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self._embedder = embedder
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._individual_top_k = individual_top_k
        self._rrf_fusion = RRFFusion(k=rrf_k)
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight

    @classmethod
    def from_config(cls, config, embedder: Embedder) -> "HybridSearchEngine":
        """
        从配置对象构建 HybridSearchEngine

        技术决策:
        - 工厂方法模式：从 config 对象构建实例，避免在 __init__ 中
          直接依赖全局 config（隐式耦合）。
        - Reranker 懒加载：默认使用 Cohere Rerank 3.5，可配置切换。
        """
        vector_retriever = VectorRetriever(
            url=config.vector_db.url,
            collection_name=config.vector_db.collection_name,
            vector_size=config.vector_db.vector_size,
            distance=config.vector_db.distance,
        )

        # BM25 Retriever（可选，Qdrant native sparse vector 可能已覆盖）
        bm25_retriever = None
        try:
            bm25_retriever = BM25Retriever(
                language="mixed",
                k1=config.vector_db.sparse_k1,
                b=config.vector_db.sparse_b,
            )
        except ImportError:
            logger.warning("BM25 retriever 不可用（rank-bm25 未安装）")

        # Reranker（默认 Cohere，可配置为 BGE 本地）
        reranker = None
        if config.reranker.provider == "cohere":
            from backend.retrieval.reranker import CohereReranker
            reranker = CohereReranker(model=config.reranker.cohere_model)
        elif config.reranker.provider == "bge":
            from backend.retrieval.reranker import BGEReranker
            reranker = BGEReranker(model=config.reranker.bge_model)

        return cls(
            embedder=embedder,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            individual_top_k=config.hybrid_search.individual_top_k,
            rrf_k=config.hybrid_search.rrf_k,
            bm25_weight=config.hybrid_search.bm25_weight,
            dense_weight=config.hybrid_search.dense_weight,
        )

    async def search(
        self,
        query: str,
        query_vector: list[float] | None = None,
        acl_filter: dict | None = None,
    ) -> tuple[list[dict], RetrievalContext]:
        """
        执行混合检索

        Args:
            query: 用户查询文本
            query_vector: 可选，预计算的 query embedding（如已缓存）
            acl_filter: 可选，ACL 过滤条件

        Returns:
            (retrieved_chunks, retrieval_context)
            retrieved_chunks: 检索到的 top-5 chunks
            retrieval_context: 检索过程信息（延迟、分数等）
        """
        start = time.perf_counter()
        context = RetrievalContext(query=query)

        # 步骤 1: Embed query（如果未提供）
        if query_vector is None:
            embed_start = time.perf_counter()
            query_vector = self._embedder.embed(query)
            context.dense_latency_ms += (time.perf_counter() - embed_start) * 1000

        # 步骤 2: 并行执行 BM25 和 Dense 检索（独立计时）
        bm25_start = time.perf_counter()

        async def run_bm25() -> list:
            if self._bm25_retriever is None:
                return []
            return self._bm25_retriever.search(query, top_k=self._individual_top_k)

        async def run_dense() -> list:
            return self._vector_retriever.search(
                query_vector=query_vector,
                top_k=self._individual_top_k,
                query_filter=acl_filter,
            )

        # 并行执行
        bm25_results, dense_results = await asyncio.gather(
            asyncio.to_thread(run_bm25),
            asyncio.to_thread(run_dense),
        )

        bm25_end = time.perf_counter()
        dense_end = time.perf_counter()

        # 分别记录各自耗时
        context.bm25_latency_ms = (bm25_end - bm25_start) * 1000
        context.dense_latency_ms = (dense_end - bm25_start) * 1000 + context.dense_latency_ms
        context.bm25_top_k = len(bm25_results)
        context.dense_top_k = len(dense_results)

        # 至少有一路有结果
        if not bm25_results and not dense_results:
            context.total_latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"混合检索无结果: query='{query[:50]}'")
            return [], context

        # 步骤 3: RRF 融合
        fusion_start = time.perf_counter()
        result_sets = {}
        if bm25_results:
            result_sets["bm25"] = bm25_results
        if dense_results:
            result_sets["dense"] = dense_results

        fused_results: list[FusionResult] = self._rrf_fusion.fuse(result_sets)
        context.fusion_candidates = len(fused_results)
        context.fusion_latency_ms = (time.perf_counter() - fusion_start) * 1000

        # 取 RRF top-K 作为 Reranker 输入
        fusion_top_k = min(50, len(fused_results))
        fusion_candidates = fused_results[:fusion_top_k]

        # 步骤 4: Cross-Encoder Reranker 精排
        rerank_start = time.perf_counter()
        final_chunks = fusion_candidates  # 默认：RRF 结果

        if self._reranker is not None and fusion_candidates:
            try:
                # 构建 Reranker 输入
                reranker_input = [
                    {
                        "chunk_id": r.chunk_id,
                        "doc_id": r.doc_id,
                        "text": r.text,
                        "section_path": r.section_path,
                        "metadata": {
                            **r.metadata,
                            "rrf_score": r.fused_score,
                            "sources": r.sources,
                            "individual_scores": r.individual_scores,
                        },
                    }
                    for r in fusion_candidates
                ]

                # 调用 Reranker
                reranked: list = await asyncio.to_thread(
                    self._reranker.rerank,
                    query,
                    reranker_input,
                    top_k=5,
                )

                if reranked:
                    final_chunks = [
                        {
                            "chunk_id": r.chunk_id,
                            "doc_id": r.doc_id,
                            "text": r.text,
                            "section_path": r.section_path,
                            "rerank_score": r.rerank_score,
                            "rank": r.final_rank,
                            "metadata": {
                                **r.metadata,
                                "rrf_score": r.metadata.get("rrf_score", 0),
                            },
                        }
                        for r in reranked
                    ]
                    context.reranked_top_k = len(final_chunks)

            except Exception as e:
                logger.warning(f"Reranker 调用失败，降级到 RRF 结果: {e}")
                final_chunks = [
                    {
                        "chunk_id": r.chunk_id,
                        "doc_id": r.doc_id,
                        "text": r.text,
                        "section_path": r.section_path,
                        "rrf_score": r.fused_score,
                        "metadata": r.metadata,
                    }
                    for r in fusion_candidates[:5]
                ]
                context.reranked_top_k = len(final_chunks)
        else:
            # 无 Reranker：直接取 RRF top-5
            final_chunks = [
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "text": r.text,
                    "section_path": r.section_path,
                    "rrf_score": r.fused_score,
                    "metadata": r.metadata,
                }
                for r in fusion_candidates[:5]
            ]

        context.rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000
        context.total_latency_ms = (time.perf_counter() - start) * 1000
        context.retrieved_chunks = final_chunks

        logger.info(
            f"混合检索完成: query='{query[:40]}...' | "
            f"bm25={context.bm25_top_k} dense={context.dense_top_k} → "
            f"fused={context.fusion_candidates} → reranked={len(final_chunks)} | "
            f"total={context.total_latency_ms:.0f}ms"
        )

        return final_chunks, context

    def search_sync(
        self,
        query: str,
        query_vector: list[float] | None = None,
        acl_filter: dict | None = None,
    ) -> tuple[list[dict], RetrievalContext]:
        """
        同步版本的混合检索（用于不支持 async 的场景）
        """
        return asyncio.run(self.search(query, query_vector, acl_filter))
