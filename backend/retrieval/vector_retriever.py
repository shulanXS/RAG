"""
vector_retriever.py — 稠密向量检索 (Qdrant HNSW)
================================================================================
技术决策记录:
- HNSW vs IVF: HNSW 是 2026 年生产环境绝对主流的向量索引算法。
  分层可导航小世界图（Hierarchical Navigable Small World），
  通过分层结构实现 O(log N) 的近似最近邻搜索，精度和速度兼得。
- 为什么用 Qdrant 而非 Chroma: Chroma 是纯内存索引，
  无法处理 >100 万向量（内存爆炸）。Qdrant 的 mmap 技术允许向量
  数据存储在磁盘上，通过操作系统的页缓存实现智能缓存。
- with_vectors=True vs False: 检索时不需要返回原始向量
  （节省带宽），只需要 payload（chunk 文本和 metadata）。

业务难点:
- HNSW 参数调优: ef_construct（构建时）和 ef（查询时）需要平衡。
  ef↑ = 精度↑ = 速度↓，实测 128 是良好的默认值。
- 过滤查询性能: payload filter 与向量搜索的联合执行是 Qdrant 的强项，
  但过滤条件过多会显著降低性能。解决方案：ACL filter 尽量简单。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """
    向量检索结果

    字段说明:
    - chunk_id / doc_id: 来源标识
    - score: 余弦相似度（0-1，Qdrant 的 normalize=True 时）
    - rank: 在向量结果中的排名
    - text / section_path: 用于展示和 Reranker 输入
    """
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    text: str
    section_path: str
    metadata: dict


class VectorRetriever:
    """
    Qdrant 稠密向量检索器

    技术要点:
    - HNSW 索引，支持 ef 参数动态调整
    - Payload filter 预过滤（ACL 过滤在检索前执行，而非后过滤）
    - Scroll API 用于批量获取 chunks 文本（构建 BM25 索引时需要）
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "enterprise_rag",
        vector_size: int = 1024,
        distance: str = "Cosine",
    ):
        self._client = QdrantClient(url=url, prefer_grpc=True)
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance

    def search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        query_filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """
        执行向量相似度检索。

        Args:
            query_vector: 查询的 embedding 向量
            top_k: 返回 top-k 结果
            query_filter: Qdrant filter 表达式（ACL 过滤等）
            score_threshold: 最小相似度阈值

        Returns:
            按余弦相似度降序排列的检索结果
        """
        # 构建 Qdrant filter
        qdrant_filter = None
        if query_filter:
            qdrant_filter = self._build_filter(query_filter)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,  # 不返回原始向量，节省带宽
            params=models.SearchParams(hnsw_ef=128),
        )

        vector_results = []
        for rank, hit in enumerate(results, 1):
            payload = hit.payload or {}
            vector_results.append(VectorSearchResult(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                score=float(hit.score),
                rank=rank,
                text=payload.get("text", ""),
                section_path=payload.get("section_path", ""),
                metadata={
                    "chunk_index": payload.get("chunk_index", 0),
                    "token_count": payload.get("token_count", 0),
                    "section_path": payload.get("section_path", ""),
                },
            ))

        return vector_results

    def hybrid_search(
        self,
        query_vector: list[float],
        sparse_query: "object",  # Qdrant sparse input: dict 或 models.Document
        top_k: int = 50,
        query_filter: dict | None = None,
    ) -> list[VectorSearchResult]:
        """
        Qdrant 原生 Hybrid Search — 在数据库层面融合 dense 和 sparse 结果。

        技术决策:
        - 2026 推荐：使用 Qdrant 1.10+ 的 Query API + Rrf 融合，
          替代 search_batch + FusionType.RRF（后者在 1.10 之后被弃用）。
        - 服务端融合，减少应用层代码和网络传输。
        - 优势: 单次往返即获得 RRF 融合结果，NDCG@10 通常优于应用层 RRF
          （因为 Qdrant 内核对排序分数有更好校准）。

        权衡取舍:
        - 优势: 简化代码，服务端 RRF 公式与写入端 BM25 公式天然一致
        - 劣势: 灵活性降低（无法对 BM25 和 dense 应用不同权重）
        """
        from qdrant_client.http import models

        qdrant_filter = None
        if query_filter:
            qdrant_filter = self._build_filter(query_filter)

        # 2026 推荐路径: Query API with RRF
        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=top_k,
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using="sparse",
                        limit=top_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
            )
            hits = results.points
        except (TypeError, AttributeError):
            # 极老版本 Qdrant 兼容：使用 search + Python 层 RRF
            logger.warning("Qdrant 版本不支持 RRF Query API，使用 dense-only 降级")
            return self.search(query_vector, top_k, query_filter)

        # 解析融合结果
        fused_results = []
        for rank, hit in enumerate(hits, 1):
            payload = hit.payload or {}
            fused_results.append(VectorSearchResult(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                score=float(hit.score),
                rank=rank,
                text=payload.get("text", ""),
                section_path=payload.get("section_path", ""),
                metadata={
                    "chunk_index": payload.get("chunk_index", 0),
                    "token_count": payload.get("token_count", 0),
                },
            ))
        return fused_results

    def scroll_all(self, limit: int = 1000) -> list[dict]:
        """
        获取 Collection 中所有 chunks 的 metadata。

        用于: (1) 构建 BM25 索引 (2) 全量统计

        技术要点:
        - Scroll API 是 Qdrant 的游标式批量读取接口
        - 适用于需要遍历全量数据的场景
        """
        all_records = []
        offset = None

        while True:
            result, offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in result:
                all_records.append({
                    "chunk_id": point.payload.get("chunk_id", ""),
                    "doc_id": point.payload.get("doc_id", ""),
                    "text": point.payload.get("text", ""),
                    "metadata": point.payload,
                })

            if offset is None:
                break

        return all_records

    @staticmethod
    def _build_filter(filter_dict: dict) -> models.Filter:
        """将字典格式的 filter 转换为 Qdrant Filter 对象"""
        must_clauses = []
        for key, value in filter_dict.items():
            if isinstance(value, list):
                must_clauses.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                must_clauses.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        if not must_clauses:
            return None
        return models.Filter(must=must_clauses)
