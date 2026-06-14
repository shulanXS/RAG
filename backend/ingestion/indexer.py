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

import qdrant_client as qc
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
        # 多租户隔离（P3-13）
        ensure_tenant_payload_index(self._client, self._collection_name)

        logger.info(f"创建 Collection '{self._collection_name}' (vector_size={self._vector_size}, sparse={with_sparse_index})")
        return True

    def index_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        doc_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> list[IndexedChunk]:
        """
        将分块及其 embedding 批量写入向量数据库。

        Args:
            chunks: 分块列表
            embeddings: 对应的 embedding 向量列表
            doc_id: 父文档 ID（用于过滤）
            tenant_id: 多租户隔离 ID（P3-13），默认 "default"

        Returns:
            IndexedChunk 列表（包含 Qdrant 分配的 vector_id）
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) 和 embeddings ({len(embeddings)}) 数量不匹配")

        points: list[dict[str, Any]] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = f"{chunk.chunk_id}"
            base_payload = {
                "doc_id": doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "section_path": chunk.section_path,
                "token_count": chunk.token_count,
                "text": chunk.text[:2000],  # 截断存储，节省空间
                "chunk_size_category": chunk.chunk_size_category,
                "parent_doc_summary": chunk.parent_doc_summary,
                "metadata": chunk.metadata,
            }
            payload = with_tenant_payload(tenant_id, base_payload)
            points.append({
                "id": point_id,
                "vector": {"dense": embedding},
                "payload": payload,
            })

        # 批量写入
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,  # 等待写入确认
        )

        logger.info(f"索引完成: doc_id={doc_id}, tenant={tenant_id}, {len(chunks)} 个 chunks")
        return [
            IndexedChunk(chunk=c, vector_id=p["id"])
            for c, p in zip(chunks, points)
        ]

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

    def upsert_chunk(
        self,
        chunk: Chunk,
        embedding: list[float],
        doc_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> str:
        """
        Upsert 单个 chunk。

        Args:
            chunk: 分块对象
            embedding: 对应的 embedding 向量
            doc_id: 父文档 ID
            tenant_id: 多租户 ID

        Returns:
            chunk_id
        """
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
        self._client.upsert(
            collection_name=self._collection_name,
            points=[{
                "id": point_id,
                "vector": {"dense": embedding},
                "payload": payload,
            }],
            wait=True,
        )
        logger.debug(f"Upsert chunk: chunk_id={chunk.chunk_id}, doc_id={doc_id}, tenant={tenant_id}")
        return chunk.chunk_id

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        删除单个 chunk。

        Args:
            chunk_id: 要删除的 chunk ID

        Returns:
            True 表示删除成功，False 表示 chunk 不存在
        """
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points=[chunk_id],
            )
            logger.debug(f"删除 chunk: chunk_id={chunk_id}")
            return True
        except Exception as e:
            logger.warning(f"删除 chunk 失败: chunk_id={chunk_id}, error={e}")
            return False

    def get_chunk(self, chunk_id: str) -> dict | None:
        """
        获取单个 chunk 的元信息和文本。

        Args:
            chunk_id: chunk ID

        Returns:
            包含 chunk 信息的字典，或 None（如果不存在）
        """
        results = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[chunk_id],
        )
        if not results:
            return None

        record = results[0]
        return {
            "chunk_id": record.payload.get("chunk_id"),
            "doc_id": record.payload.get("doc_id"),
            "chunk_index": record.payload.get("chunk_index"),
            "section_path": record.payload.get("section_path"),
            "token_count": record.payload.get("token_count"),
            "text": record.payload.get("text"),
            "chunk_size_category": record.payload.get("chunk_size_category"),
            "parent_doc_summary": record.payload.get("parent_doc_summary"),
            "metadata": record.payload.get("metadata"),
        }

    def get_doc_chunks(self, doc_id: str) -> list[dict]:
        """
        获取指定文档的所有 chunks。

        Args:
            doc_id: 文档 ID

        Returns:
            该文档的所有 chunk 信息列表
        """
        results, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=10000,
        )

        chunks = []
        for record in results:
            chunks.append({
                "chunk_id": record.payload.get("chunk_id"),
                "doc_id": record.payload.get("doc_id"),
                "chunk_index": record.payload.get("chunk_index"),
                "section_path": record.payload.get("section_path"),
                "token_count": record.payload.get("token_count"),
                "text": record.payload.get("text"),
                "chunk_size_category": record.payload.get("chunk_size_category"),
                "parent_doc_summary": record.payload.get("parent_doc_summary"),
                "metadata": record.payload.get("metadata"),
            })

        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks

    def get_chunk_count(self) -> int:
        """
        获取 Collection 中的总 chunk 数量。

        Returns:
            chunk 总数
        """
        info = self._client.get_collection(collection_name=self._collection_name)
        return info.points_count or 0

    # =========================================================================
    # Embedding 版本管理
    # =========================================================================

    def compute_embedding_version(
        self,
        model_name: str,
        model_version: str | None = None,
    ) -> str:
        """
        计算当前 embedding 配置的版本标识。

        版本标识基于模型名称 + 可选版本/时间戳哈希，
        用于检测模型变更后触发重新 embedding。

        Args:
            model_name: embedding 模型名称
            model_version: 可选，模型版本号

        Returns:
            版本标识字符串（8位哈希）
        """
        import hashlib
        import time

        version_str = f"{model_name}:{model_version or int(time.time())}"
        return hashlib.md5(version_str.encode()).hexdigest()[:8]

    def get_embedding_version(self) -> str | None:
        """
        获取当前 collection 的 embedding 版本。

        Returns:
            版本字符串，如果未设置则返回 None
        """
        try:
            info = self._client.get_collection(collection_name=self._collection_name)
            return info.params.get("embedding_version", None)
        except Exception:
            return None

    def detect_embedding_version_drift(
        self,
        expected_version: str,
    ) -> int:
        """
        检测有多少 chunk 使用了旧版本 embedding（与 expected_version 不一致）。

        用于在模型更新后评估重 embedding 的规模。

        算法:
        - 统计 total = 全量 chunk 数
        - 统计 matched = payload 中 embedding_version == expected_version 的 chunk 数
        - 返回 (total - matched) 即「需要重 embedding 的 chunk 数」

        Args:
            expected_version: 期望的 embedding 版本

        Returns:
            旧版本（需要重 embedding）的 chunk 数量
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        try:
            total = self.get_chunk_count()
            if total == 0:
                return 0

            # 1) 统计匹配 expected_version 的 chunk 数（scroll 必须传 limit，但分页累加）
            matched = 0
            offset = None
            page_size = 1000
            while True:
                results, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="embedding_version",
                                match=MatchValue(value=expected_version),
                            )
                        ]
                    ),
                    limit=page_size,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                matched += len(results)
                if next_offset is None:
                    break
                offset = next_offset

            return max(0, total - matched)
        except Exception as e:
            logger.warning(f"detect_embedding_version_drift 异常: {e}")
            return 0

    def reindex_legacy_chunks(
        self,
        expected_version: str,
        embedder,  # Embedder instance
    ) -> dict:
        """
        重新索引使用旧版本 embedding 的 chunk。

        检测到 embedding 版本漂移后，批量重新 embedding 并更新索引。

        Args:
            expected_version: 期望的新版本
            embedder: Embedder 实例（用于生成新 embedding）

        Returns:
            重新索引统计信息
        """
        from qdrant_client.http.models import Filter

        logger.info(f"开始重 embedding：expected_version={expected_version}")

        results, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=None,  # 全量扫描
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        legacy_count = 0
        updated_count = 0

        for point in results:
            payload = point.payload
            current_version = payload.get("embedding_version", "")

            if current_version == expected_version:
                continue

            legacy_count += 1

            # 重新 embedding
            new_embedding = embedder.embed([payload.get("text", "")])[0]

            # 更新
            self._client.upsert(
                collection_name=self._collection_name,
                points=[{
                    "id": point.id,
                    "vector": {"dense": new_embedding},
                    "payload": {
                        **payload,
                        "embedding_version": expected_version,
                    },
                }],
            )
            updated_count += 1

        logger.info(f"重 embedding 完成：{updated_count}/{legacy_count} 个 chunk 已更新")
        return {
            "total_scanned": len(results),
            "legacy_count": legacy_count,
            "updated_count": updated_count,
        }
