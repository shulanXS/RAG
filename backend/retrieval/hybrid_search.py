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
from typing import TYPE_CHECKING, Literal

from backend.ingestion.embedder import Embedder
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.fusion import RRFFusion, DynamicRRFFusion, DEFAULT_K_BY_COMPLEXITY, FusionResult
from backend.retrieval.reranker import CrossEncoderReranker, get_reranker
from backend.retrieval.vector_retriever import VectorRetriever

if TYPE_CHECKING:
    from backend.agentic.query_signals import QuerySignals
    from backend.security.tenant import TenantContext

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
    total_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    dense_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    bm25_top_k: int = 0
    dense_top_k: int = 0
    fusion_candidates: int = 0
    reranked_top_k: int = 0
    retrieved_chunks: list = None
    # P2-B5: DynamicRRFFusion 实际选用的 k (随 complexity 变化)
    fusion_k_used: int = 0
    # P2-B7: 路由信号 (P2-B6 analyzer 的输出), 用于 OTel attribute
    query_signals: dict | None = None

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
        bm25_mode: str = "qdrant_sparse",
        fusion: DynamicRRFFusion | None = None,
        dynamic_k_enabled: bool = True,
    ):
        self._embedder = embedder
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._individual_top_k = individual_top_k
        # P2-B5: 默认用 DynamicRRFFusion, 通过 complexity 选 k。
        # 显式传 fusion 可覆盖 (P2 不引入 ML, fusion 实例还是 DynamicRRFFusion)。
        self._fusion = fusion or DynamicRRFFusion(
            k_default=rrf_k,
            k_by_complexity=DEFAULT_K_BY_COMPLEXITY,
            enabled=dynamic_k_enabled,
        )
        # 保留旧字段以备外部代码访问 (P1 旧版本属性), 但实际用 self._fusion
        self._rrf_fusion = self._fusion
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._bm25_mode = bm25_mode

    @classmethod
    def from_config(cls, config, embedder: Embedder) -> "HybridSearchEngine":
        """
        从配置对象构建 HybridSearchEngine

        技术决策:
        - 工厂方法模式：从 config 对象构建实例，避免在 __init__ 中
          直接依赖全局 config（隐式耦合）。
        - Reranker 懒加载：默认使用 Cohere Rerank 3.5，可配置切换。
        - BM25 路径根据 config.vector_db.bm25_mode 决定：
            * "qdrant_sparse": 服务端融合，无需外部 BM25Retriever
            * "external":  保留 rank_bm25 + 应用层 RRF 融合
        """
        vector_retriever = VectorRetriever(
            url=config.vector_db.url,
            collection_name=config.vector_db.collection_name,
            vector_size=config.vector_db.vector_size,
            distance=config.vector_db.distance,
        )

        # BM25 Retriever：仅在 external 模式下需要
        bm25_retriever = None
        if config.vector_db.bm25_mode == "external":
            try:
                bm25_retriever = BM25Retriever(
                    language="mixed",
                    k1=config.vector_db.sparse_k1,
                    b=config.vector_db.sparse_b,
                )
            except ImportError:
                logger.warning("BM25 retriever 不可用（rank-bm25 未安装）")
        else:
            logger.info("BM25 mode=qdrant_sparse: 使用 Qdrant 原生 sparse vector，跳过外部 BM25Retriever")

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
            bm25_mode=config.vector_db.bm25_mode,
            dynamic_k_enabled=config.hybrid_search.dynamic_k_enabled,
            fusion=DynamicRRFFusion(
                k_default=config.hybrid_search.rrf_k,
                k_by_complexity=config.hybrid_search.k_by_complexity,
                enabled=config.hybrid_search.dynamic_k_enabled,
            ),
        )

    async def search(
        self,
        query: str,
        query_vector: list[float] | None = None,
        acl_filter: dict | None = None,
        tenant: "TenantContext | str | None" = None,
        complexity: str | None = None,
        signals: "QuerySignals | None" = None,
    ) -> tuple[list[dict], RetrievalContext]:
        """
        执行混合检索

        Args:
            query: 用户查询文本
            query_vector: 可选，预计算的 query embedding（如已缓存）
            acl_filter: 可选，ACL 过滤条件
            tenant: 可选，租户上下文（TenantContext 或 str tenant_id）。
                    传入后会自动注入 tenant_id 过滤条件，
                    与 acl_filter 通过 AND 组合，确保跨租户数据不可见。
            complexity: P2-B5: 可选，路由出来的 query complexity
                       (simple/moderate/complex/beyond_kb)。
                       传入后 DynamicRRFFusion 按此选 k。
                       None 时用 k_default (60)。
            signals: P2-B7: 可选，QuerySignals (pronoun/entity/length/quote 等)。
                    仅写入 RetrievalContext, 供上层 OTel attributes / debug。

        Returns:
            (retrieved_chunks, retrieval_context)
            retrieved_chunks: 检索到的 top-5 chunks
            retrieval_context: 检索过程信息（延迟、分数等）

        技术决策:
        - 根据 bm25_mode 走两条不同路径：
            * "qdrant_sparse": Qdrant 服务端融合 dense + sparse，
              单一结果流，避免 RRF 融合时的 chunk_id 重复
            * "external":      保留 rank_bm25 + 应用层 RRF 融合（高级调参场景）
        - 无论哪种模式，最终都做 chunk_id 去重，避免 RRF 权重叠加。
        - P2-B5: complexity 透传到 DynamicRRFFusion.fuse(..., complexity=...)
        - P2-B7: signals 写入 context.query_signals, 上层 (OTel span) 读
        """
        start = time.perf_counter()
        context = RetrievalContext(query=query)
        # P2-B7: 把 signals 早写到 context, 即使中途异常也保留
        if signals is not None:
            context.query_signals = signals.to_dict()
        start = time.perf_counter()
        context = RetrievalContext(query=query)

        # 多租户隔离：合并 acl_filter + tenant filter（P3-13）
        from backend.security.tenant import build_tenant_filter

        tenant_filter = build_tenant_filter(tenant, acl_filter) if tenant is not None else None
        effective_acl = acl_filter  # 透传给可能的额外逻辑

        # 步骤 1: Embed query（如果未提供）
        if query_vector is None:
            embed_start = time.perf_counter()
            query_vector = self._embedder.embed(query)
            context.dense_latency_ms += (time.perf_counter() - embed_start) * 1000

        bm25_results: list = []
        dense_results: list = []
        bm25_ms = 0.0
        dense_ms = 0.0

        if self._bm25_mode == "qdrant_sparse":
            # 路径 A: Qdrant 服务端 hybrid search（推荐）
            dense_start = time.perf_counter()
            fused = self._vector_retriever.hybrid_search(
                query_vector=query_vector,
                sparse_query=self._build_sparse_query(query),
                top_k=self._individual_top_k,
                query_filter=tenant_filter,  # 用合并后的 filter
            )
            dense_ms = (time.perf_counter() - dense_start) * 1000
            dense_results = [
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "score": r.score,
                    "rank": r.rank,
                    "text": r.text,
                    "section_path": r.section_path,
                    "metadata": r.metadata,
                }
                for r in fused
            ]
            context.bm25_top_k = 0  # BM25 已融合到 dense 结果中
            context.dense_top_k = len(dense_results)

            # Qdrant 服务端融合已完成，跳过应用层 RRF
            result_sets: dict[str, list] = {"fused": dense_results}
            fused_results = self._normalize_fused(result_sets)
        else:
            # 路径 B: 外部 rank_bm25 + dense + 应用层 RRF
            async def run_bm25() -> tuple[list, float]:
                t0 = time.perf_counter()
                if self._bm25_retriever is None:
                    return [], 0.0
                result = self._bm25_retriever.search(query, top_k=self._individual_top_k)
                return result, (time.perf_counter() - t0) * 1000

            async def run_dense() -> tuple[list, float]:
                t0 = time.perf_counter()
                result = await self._vector_retriever.search(
                    query_vector=query_vector,
                    top_k=self._individual_top_k,
                    query_filter=tenant_filter,  # 用合并后的 filter
                )
                return result, (time.perf_counter() - t0) * 1000

            (bm25_results, bm25_ms), (dense_results, dense_ms) = await asyncio.gather(
                asyncio.to_thread(run_bm25),
                asyncio.to_thread(run_dense),
            )

            context.bm25_latency_ms = bm25_ms
            context.dense_latency_ms = dense_ms
            context.bm25_top_k = len(bm25_results)
            context.dense_top_k = len(dense_results)

            if not bm25_results and not dense_results:
                context.total_latency_ms = (time.perf_counter() - start) * 1000
                logger.warning(f"混合检索无结果: query='{query[:50]}'")
                return [], context

            result_sets = {}
            if bm25_results:
                result_sets["bm25"] = bm25_results
            if dense_results:
                result_sets["dense"] = dense_results
            fused_results = self._normalize_fused(result_sets)

        # 步骤 3: RRF 融合（qdrant_sparse 模式下 fused_results 已经是单流）
        fusion_start = time.perf_counter()
        if self._bm25_mode != "qdrant_sparse":
            # P2-B5: 把 complexity 透传到 DynamicRRFFusion
            fused_results: list[FusionResult] = self._fusion.fuse(
                result_sets, complexity=complexity
            )
        # 记录实际用的 k, 供 OTel span (B6) 与 metric 打点
        k_used = self._fusion.k_for_complexity(complexity)
        context.fusion_candidates = len(fused_results)
        context.fusion_latency_ms = (time.perf_counter() - fusion_start) * 1000
        # P2-B5: 暴露 k_used 到 RetrievalContext, 上层 B6 span 写 attributes 用
        context.fusion_k_used = k_used

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
                # P2-B3: 区分瞬时 / 永久错误, 不同路径降级
                from backend.retrieval.reranker import (
                    is_transient_error,
                    is_permanent_error,
                )
                if is_transient_error(e):
                    logger.warning(
                        f"Reranker 瞬时错误 (rate limit / timeout): {e}; "
                        f"降级到 RRF top-5"
                    )
                elif is_permanent_error(e):
                    logger.error(
                        f"Reranker 永久错误 (auth / bad input): {e}; "
                        f"降级到 RRF top-5 (不重试)"
                    )
                else:
                    logger.warning(f"Reranker 未知错误: {e}; 降级到 RRF top-5")
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

    def _build_sparse_query(self, query: str) -> dict | "models.Document":
        """
        从 query 构造 Qdrant sparse vector 查询体。

        Qdrant 1.10+ 推荐方案：传 `models.Document(text=..., model="Qdrant/bm25")`，
        由 Qdrant 服务端按 BM25 (IDF + TF 归一化) 算 indices/values。

        这样能保持和写入端 (index_chunks_with_sparse) 完全一致的 BM25 公式，
        避免应用层手算 TF 与服务端 IDF 公式不一致造成的排序偏差。
        """
        try:
            from qdrant_client.http import models
            return models.Document(text=query, model="Qdrant/bm25")
        except Exception:
            # 极旧版 Qdrant 兼容：手工 tokenize + TF
            import collections
            tokens = [t for t in query.split() if t]
            if not tokens:
                return {"indices": [], "values": []}
            counter = collections.Counter(tokens)
            indices = sorted(counter.keys())
            values = [float(counter[i]) for i in indices]
            return {"indices": indices, "values": values}

    def _normalize_fused(self, result_sets: dict[str, list]) -> list[FusionResult]:
        """
        将多路检索结果转换为 FusionResult 列表（RRF 融合前的中间格式）。

        同时按 chunk_id 去重，避免同一 chunk 在多路中出现造成 RRF 权重叠加。
        """
        seen_chunk_ids: set[str] = set()
        out: list[FusionResult] = []
        for source, items in result_sets.items():
            for item in items:
                chunk_id = item.get("chunk_id", "")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                out.append(
                    FusionResult(
                        chunk_id=chunk_id,
                        doc_id=item.get("doc_id", ""),
                        text=item.get("text", ""),
                        section_path=item.get("section_path", ""),
                        metadata=item.get("metadata", {}),
                        fused_score=float(item.get("score", 0.0)),
                        rank=item.get("rank", 0),
                        individual_scores={source: float(item.get("score", 0.0))},
                        sources=[source],
                    )
                )
        return out

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
