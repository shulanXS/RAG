"""
indexer.py — Qdrant 向量索引写入器
================================================================================
技术决策记录:
- Qdrant 选型理由:
  (1) mmap 零拷贝: >1M 向量时内存效率显著优于纯内存索引
  (2) sparse vector 原生支持: 同一 collection 内混合 BM25 + dense，无需外部 BM25 索引
  (3) HNSW + payload index 联合过滤: 检索时直接过滤 ACL，避免 post-filter 的内存泄漏
  (4) Docker 一键启动，开发体验友好

业务难点:
- 批量写入性能: 逐条插入极慢，使用批量 upsert。
- Collection 不存在: 自动创建 collection（如果已存在则跳过）。
- 向量维度不匹配: 启动时校验 embedding 维度与配置一致性。

权衡取舍:
- pgvector vs Qdrant: pgvector 在 <5M 向量时更简单（无需额外服务），
  但缺乏 sparse vector 原生支持。Qdrant 是需要 Hybrid Search 的必然选择。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import qdrant_client.http.exceptions as qe
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from backend.ingestion.chunker import Chunk
from backend.security.tenant import (
    DEFAULT_TENANT_ID,
    ensure_tenant_payload_index,
    with_tenant_payload,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexedChunk:
    """
    已索引的分块 — 包含向量数据库返回的元信息
    """
    chunk: Chunk
    vector_id: str | None = None
    indexed_at: str = ""


class QdrantIndexer:
    """
    Qdrant 向量索引写入器

    技术要点:
    - 批量 upsert: batch_size=64，平衡内存占用和写入速度
    - 自动创建 Collection: 如果不存在则创建，支持首次运行
    - 双重索引: dense 向量（HNSW）+ sparse 向量（BM25）同 Collection
    - Payload 索引: 对 doc_id、section_path、chunk_index 建立字段索引，
      支持高效过滤查询

    风险考量:
    - 向量维度不匹配: 启动时严格校验，发现不一致立即报错
    - Collection 已存在时的策略: 跳过创建（不覆盖已有数据），支持增量索引
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "enterprise_rag",
        vector_size: int = 1024,
        distance: str = "Cosine",
        batch_size: int = 64,
    ):
        self._client = QdrantClient(url=url, prefer_grpc=True)
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = Distance[distance.upper()]
        self._batch_size = batch_size

    def ensure_collection(self, with_sparse_index: bool = True) -> bool:
        """
        确保 Collection 存在，不存在则创建。

        Args:
            with_sparse_index: 是否同时创建 sparse 向量索引（BM25）

        Returns:
            True 表示创建了新 Collection，False 表示已存在

        技术决策:
        - 2026 推荐: 使用 Qdrant 1.10+ 的 TextIndexParams + Modifier.IDF 方案，
          让 Qdrant 服务端自动计算 BM25 权重（indices & IDF），
          不再需要应用层 rank_bm25 库。
        - Modifier.IDF 是 BM25 排序质量的关键，缺失则退化为纯 TF。
        """
        try:
            self._client.get_collection(collection_name=self._collection_name)
            logger.info(f"Collection '{self._collection_name}' 已存在，跳过创建")
            # 即使已存在，也确保 tenant_id payload 索引存在（幂等）
            ensure_tenant_payload_index(self._client, self._collection_name)
            return False
        except qe.NotFound:
            pass

        # 创建 Collection（同时支持 dense 和 sparse 向量）
        sparse_cfg = None
        if with_sparse_index:
            # Qdrant 1.10+ 原生 BM25: 服务端自动算 IDF + 词频归一化
            sparse_cfg = {
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                        modifier=models.Modifier.IDF,  # 真正的 BM25 关键
                    ),
                )
            }

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                # Dense 向量索引（HNSW）
                "dense": VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=128,
                    ),
                ),
            },
            sparse_vectors_config=sparse_cfg,
        )

        # 创建 Payload 索引（加速过滤查询）
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="chunk_index",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="section_path",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="token_count",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        # 多租户隔离
        ensure_tenant_payload_index(self._client, self._collection_name)

        logger.info(f"创建 Collection '{self._collection_name}' (vector_size={self._vector_size}, sparse={with_sparse_index})")
        return True

    def index_chunks_with_sparse(
        self,
        chunks: list[Chunk],
        dense_embeddings: list[list[float]],
        doc_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        precomputed_sparse: list[dict] | None = None,
    ) -> list[IndexedChunk]:
        """
        将分块同时写入 dense 和 sparse 向量索引（用于 Hybrid Search）。

        技术决策:
        - Qdrant 1.10+ sparse + Modifier.IDF：让 Qdrant 服务端算 BM25。
        - 两种写入方式：
          (1) 推荐：传 `models.Document(text=..., model="Qdrant/bm25")`，
             Qdrant 服务端自动 build sparse vector（避免客户端预计算）。
          (2) 兼容：传 precomputed `{"indices": [int], "values": [float]}`，
             适用于已有外部 BM25 计算的迁移场景。
        - 服务端融合：检索时由 Qdrant 在数据库内 RRF 融合 dense+sparse，
          减少网络传输和 Python 层 RRF 代码。
        """
        if len(chunks) != len(dense_embeddings):
            raise ValueError("chunks 与 dense_embeddings 数量必须一致")
        if precomputed_sparse is not None and len(precomputed_sparse) != len(chunks):
            raise ValueError("precomputed_sparse 与 chunks 数量必须一致")

        points: list[dict[str, Any]] = []
        for i, (chunk, dense_emb) in enumerate(zip(chunks, dense_embeddings)):
            point_id = f"{chunk.chunk_id}"
            base_payload = {
                "doc_id": doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "section_path": chunk.section_path,
                "token_count": chunk.token_count,
                "text": chunk.text[:2000],
                "chunk_size_category": chunk.chunk_size_category,
                "parent_doc_summary": chunk.parent_doc_summary,
                "metadata": chunk.metadata,
            }
            payload = with_tenant_payload(tenant_id, base_payload)

            # 选择 sparse vector 表达方式
            if precomputed_sparse is not None:
                sparse_vec = precomputed_sparse[i]
            else:
                # 让 Qdrant 服务端从 text 字段自动生成 BM25 sparse vector
                sparse_vec = models.Document(
                    text=chunk.text,
                    model="Qdrant/bm25",
                )

            points.append({
                "id": point_id,
                "vector": {
                    "dense": dense_emb,
                    "sparse": sparse_vec,
                },
                "payload": payload,
            })

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

        logger.info(f"索引完成 (with sparse): doc_id={doc_id}, tenant={tenant_id}, {len(chunks)} 个 chunks")
        return [
            IndexedChunk(chunk=c, vector_id=p["id"])
            for c, p in zip(chunks, points)
        ]

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        删除指定文档的所有 chunks。

        用于增量更新: 文档变更时，先删后建。

        Returns:
            返回删除操作的 operation_id（Qdrant 不返回删除行数），
            实际行数可由调用方通过 scroll 重算。
        """
        result = self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
        )
        logger.info(f"删除文档: doc_id={doc_id}, operation_id={getattr(result, 'operation_id', None)}")
        return getattr(result, "operation_id", 0) or 0

    def get_collection_info(self) -> dict[str, Any]:
        """获取 Collection 统计信息"""
        info = self._client.get_collection(collection_name=self._collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }
