"""
document_registry.py — 文档元数据注册表 (P1-A1/A3/A4)

技术决策:
- 用 Qdrant payload-only collection 持久化文档元数据，取代之前的内存 dict
  (模块级 _indexed_documents，重启即丢)
- payload schema 含 file_id/tenant_id/filename/size/status/chunks/started_at/
  finished_at/elapsed_ms/error/content_hash/created_by
- 复合 payload 索引: (tenant_id + status), (tenant_id + content_hash)
  支持按租户快速过滤 + 内容去重查询

业务:
- 任何 backend 重启 / 滚动发布不影响文档状态可见性
- 上传时去重: 同一 (tenant_id, content_hash, status='indexed') 视为同文档
- 删除时同步回收 Qdrant chunks (callers 自行调用 indexer.delete_by_doc_id)
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import qdrant_client.http.exceptions as qe
from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.domain.tenant import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)

REGISTRY_COLLECTION = "document_registry"


@dataclass
class DocumentRecord:
    """单个文档的完整 audit record。"""

    file_id: str
    tenant_id: str = DEFAULT_TENANT_ID
    filename: str = ""
    size: int = 0
    status: str = "queued"  # queued | processing | indexed | failed
    chunks: int = 0
    started_at: int = 0
    finished_at: int = 0
    elapsed_ms: int = 0
    error: str | None = None
    content_hash: str = ""
    created_by: str = ""

    def to_payload(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_payload(cls, payload: dict) -> "DocumentRecord":
        return cls(
            file_id=payload.get("file_id", ""),
            tenant_id=payload.get("tenant_id", DEFAULT_TENANT_ID),
            filename=payload.get("filename", ""),
            size=int(payload.get("size", 0)),
            status=payload.get("status", "unknown"),
            chunks=int(payload.get("chunks", 0)),
            started_at=int(payload.get("started_at", 0)),
            finished_at=int(payload.get("finished_at", 0)),
            elapsed_ms=int(payload.get("elapsed_ms", 0)),
            error=payload.get("error"),
            content_hash=payload.get("content_hash", ""),
            created_by=payload.get("created_by", ""),
        )


class DocumentRegistry:
    """
    Qdrant payload-only collection 上的文档注册表。

    设计原则:
    - 同步操作: Qdrant Python SDK 是同步的，本类不引入 async 包装
      (上层使用方通常在线程池 / 异步任务里调用)
    - 单点 Qdrant 客户端: 实例化时建立一次，复用连接
    - 幂等: ensure_collection 多次调用安全
    """

    def __init__(self, url: str = "http://localhost:6333"):
        self._client = QdrantClient(url=url, prefer_grpc=True)
        self._collection = REGISTRY_COLLECTION
        self._lock = threading.RLock()
        self._ensured = False

    def ensure_collection(self) -> None:
        """幂等创建 registry collection。多次调用安全。"""
        with self._lock:
            if self._ensured:
                return
            try:
                self._client.get_collection(collection_name=self._collection)
                self._ensured = True
                return
            except qe.NotFound:
                pass

            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={},  # payload-only，无 dense/sparse
            )
            # 复合索引: tenant_id + status (按租户过滤某状态)
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="status",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            # 内容哈希索引 (去重查询)
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="content_hash",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="started_at",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
            logger.info(f"创建 DocumentRegistry collection '{self._collection}'")
            self._ensured = True

    def upsert(self, record: DocumentRecord) -> None:
        """插入或更新一条文档记录。"""
        self.ensure_collection()
        payload = record.to_payload()
        # Qdrant point id 用 file_id 的稳定 hash（UUID 不安全字符）
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record.file_id))
        self._client.upsert(
            collection_name=self._collection,
            points=[{
                "id": point_id,
                "vector": {},
                "payload": payload,
            }],
            wait=True,
        )

    def update_status(
        self,
        file_id: str,
        status: str,
        **fields: Any,
    ) -> None:
        """仅更新状态 + 给定字段，其他字段保持不变。"""
        existing = self.get(file_id)
        if existing is None:
            logger.warning(f"update_status 找不到 file_id={file_id}")
            return
        existing.status = status
        for k, v in fields.items():
            if hasattr(existing, k):
                setattr(existing, k, v)
        self.upsert(existing)

    def get(self, file_id: str) -> DocumentRecord | None:
        """获取单条记录。"""
        self.ensure_collection()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_id))
        try:
            results = self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id],
            )
        except Exception as e:
            logger.warning(f"DocumentRegistry.get 异常: {e}")
            return None
        if not results:
            return None
        return DocumentRecord.from_payload(results[0].payload)

    def delete(self, file_id: str) -> bool:
        """删除一条记录。返回是否真的删除了。"""
        self.ensure_collection()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_id))
        try:
            self._client.delete(
                collection_name=self._collection,
                points=[point_id],
            )
            return True
        except Exception as e:
            logger.warning(f"DocumentRegistry.delete 异常: {e}")
            return False

    def list_by_tenant(
        self,
        tenant_id: str = DEFAULT_TENANT_ID,
        status: str | None = None,
        limit: int = 100,
    ) -> list[DocumentRecord]:
        """按租户列出文档（可选 status 过滤）。"""
        self.ensure_collection()
        must = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            )
        ]
        if status:
            must.append(
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value=status),
                )
            )
        results, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(must=must),
            limit=limit,
        )
        return [DocumentRecord.from_payload(r.payload) for r in results]

    def find_by_content_hash(
        self,
        content_hash: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        status: str | None = "indexed",
    ) -> DocumentRecord | None:
        """
        按 (tenant_id, content_hash) 查找已索引过的同内容文档。
        status 默认 'indexed' — 'failed' 的旧记录不视为有效去重目标。
        """
        self.ensure_collection()
        must = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="content_hash",
                match=models.MatchValue(value=content_hash),
            ),
        ]
        if status:
            must.append(
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value=status),
                )
            )
        results, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(must=must),
            limit=1,
        )
        if not results:
            return None
        return DocumentRecord.from_payload(results[0].payload)


# -----------------------------------------------------------------------------
# 模块级单例 (类似 trace_store 的 pattern)
# -----------------------------------------------------------------------------
_instance: DocumentRegistry | None = None
_instance_lock = threading.Lock()


def get_document_registry() -> DocumentRegistry:
    """获取 DocumentRegistry 单例。"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                from backend.config import get_config
                cfg = get_config()
                url = cfg.vector_db.url if cfg else "http://localhost:6333"
                _instance = DocumentRegistry(url=url)
    return _instance


def reset_document_registry_for_test() -> None:
    """仅供测试：清除单例。"""
    global _instance
    with _instance_lock:
        _instance = None
