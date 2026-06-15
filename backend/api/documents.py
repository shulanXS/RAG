"""
documents.py — 文档管理 API 路由
================================================================================
技术决策记录:
|- 上传后通过 Arq 任务队列 (P1-A2) 触发异步索引，任务持久化到 Redis Stream
  (替换之前的 FastAPI BackgroundTasks，绑定 worker 进程会丢任务)
|- 文档元数据持久化到 Qdrant DocumentRegistry (P1-A1)
  不再使用内存 dict，重启/滚动发布后状态可追溯
|- 状态查询通过 GET /documents/{id}/status 返回完整 audit record (P1-A3)
  含 filename / size / tenant_id / content_hash / error
|- 上传时做内容哈希去重 (P1-A4): 同一 (tenant_id, content_hash) 已索引则复用
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

from backend.ingestion.document_registry import (
    DocumentRecord,
    get_document_registry,
)
from backend.security.auth import require_current_user
from backend.security.tenant import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# 文档存储目录（可配置）
DOCS_DIR = Path(__file__).parent.parent.parent / "data" / "uploaded_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


class DocumentInfo(BaseModel):
    """P1-A3: 完整 audit record — 用户上传后能查到自己上传了什么"""
    id: str
    filename: str = ""
    size: int = 0
    tenant_id: str = DEFAULT_TENANT_ID
    status: str = "ready"  # queued | processing | indexed | failed
    content_hash: str = ""
    created_by: str = ""
    uploaded_at: int = 0
    chunks: int | None = None
    elapsed_ms: int | None = None
    started_at: int | None = None
    finished_at: int | None = None
    error: str | None = None
    deduplicated: bool = False  # P1-A4: 是否命中内容去重


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = 0


class DocumentUploadResponse(BaseModel):
    success: bool
    document: DocumentInfo | None = None
    error: str | None = None
    indexing: str = "queued"  # queued | processing | indexed | failed | deduplicated


# 文件扩展名白名单统一从 ingestion.document_parser 的 registry 派生
# （单一来源，添加新格式时只改一处）
from backend.ingestion.document_parser import _EXTENSION_PARSERS

ALLOWED_EXTENSIONS = frozenset(_EXTENSION_PARSERS.keys())
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def _validate_file(file: UploadFile) -> None:
    """验证文件类型"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}。支持的类型: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


def _record_to_info(rec: DocumentRecord, deduplicated: bool = False) -> DocumentInfo:
    """P1-A1+A3: DocumentRecord → API 响应 DTO"""
    return DocumentInfo(
        id=rec.file_id,
        filename=rec.filename,
        size=rec.size,
        tenant_id=rec.tenant_id,
        status=rec.status,
        content_hash=rec.content_hash,
        created_by=rec.created_by,
        uploaded_at=rec.started_at or rec.finished_at,
        chunks=rec.chunks or None,
        elapsed_ms=rec.elapsed_ms or None,
        started_at=rec.started_at or None,
        finished_at=rec.finished_at or None,
        error=rec.error,
        deduplicated=deduplicated,
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    token_payload: dict = Depends(require_current_user),
) -> DocumentUploadResponse:
    """
    上传文档到系统。

    P1-A1+A2+A3+A4:
    - 上传时立刻写 DocumentRegistry (status=queued, 含完整 audit record)
    - 内容哈希去重: 命中已索引文档则不重复处理
    - 通过 Arq 任务队列 (Redis Stream) 触发持久化索引
    - worker 重启 / 滚动发布不丢任务
    """
    try:
        _validate_file(file)

        content = await file.read()
        size = len(content)

        if size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件大小超过限制 ({MAX_FILE_SIZE // 1024 // 1024}MB)",
            )

        # 解析 tenant_id（多租户隔离）
        tenant_id = (
            token_payload.get("tenant_id")
            if isinstance(token_payload, dict)
            else DEFAULT_TENANT_ID
        ) or DEFAULT_TENANT_ID
        created_by = (
            token_payload.get("sub", "") if isinstance(token_payload, dict) else ""
        )

        # ---- P1-A4: 内容哈希去重 ----
        content_hash = hashlib.sha256(content).hexdigest()
        registry = get_document_registry()
        existing = registry.find_by_content_hash(
            content_hash=content_hash, tenant_id=tenant_id
        )
        if existing is not None:
            logger.info(
                f"[upload] 去重命中: tenant={tenant_id} hash={content_hash[:12]} "
                f"-> 复用 file_id={existing.file_id}"
            )
            return DocumentUploadResponse(
                success=True,
                document=_record_to_info(existing, deduplicated=True),
                indexing="deduplicated",
            )

        # ---- P1-A3: 完整 audit record ----
        file_id = str(__import__("uuid").uuid4())
        ext = Path(file.filename).suffix.lower()
        stored_path = DOCS_DIR / f"{file_id}{ext}"
        stored_path.write_bytes(content)

        record = DocumentRecord(
            file_id=file_id,
            tenant_id=tenant_id,
            filename=file.filename or "",
            size=size,
            status="queued",
            content_hash=content_hash,
            created_by=created_by,
            started_at=int(time.time()),
        )
        registry.upsert(record)
        logger.info(
            f"[upload] queued: file_id={file_id} ({file.filename}, {size} bytes, "
            f"hash={content_hash[:12]})"
        )

        # ---- P1-A2: Arq 任务队列 (持久化) ----
        try:
            from backend.workers.arq_pool import enqueue_index_task
            await enqueue_index_task(
                file_path=stored_path,
                file_id=file_id,
                tenant_id=tenant_id,
                strategy="recursive",
            )
        except Exception as e:
            # 入队失败时回退到直接执行 (开发环境无 Redis)
            logger.warning(f"[upload] Arq 入队失败: {e}; fallback 到直接执行")
            from backend.ingestion.pipeline import run_index_pipeline
            await run_index_pipeline(
                file_path=stored_path,
                file_id=file_id,
                tenant_id=tenant_id,
                strategy="recursive",
            )

        return DocumentUploadResponse(
            success=True,
            document=_record_to_info(record),
            indexing="queued",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Document upload failed: {e}")
        return DocumentUploadResponse(success=False, error=str(e))


@router.get("/{doc_id}/status", response_model=DocumentInfo)
async def get_indexing_status(
    doc_id: str,
    token_payload: dict = Depends(require_current_user),
) -> DocumentInfo:
    """P1-A3: 获取文档完整 audit record (含 filename/size/tenant_id 等)"""
    rec = get_document_registry().get(doc_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="文档不存在或尚未开始处理")
    return _record_to_info(rec)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    status: Annotated[str | None, Query()] = None,
    token_payload: dict = Depends(require_current_user),
) -> DocumentListResponse:
    """P1-A1: 列出当前租户的文档（按 tenant 过滤，支持 status 过滤）"""
    tenant_id = (
        token_payload.get("tenant_id", DEFAULT_TENANT_ID)
        if isinstance(token_payload, dict)
        else DEFAULT_TENANT_ID
    )
    records = get_document_registry().list_by_tenant(
        tenant_id=tenant_id, status=status, limit=skip + limit
    )
    total = len(records)
    paginated = records[skip : skip + limit]
    return DocumentListResponse(
        documents=[_record_to_info(r) for r in paginated],
        total=total,
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    token_payload: dict = Depends(require_current_user),
) -> DocumentInfo:
    """P1-A3: 获取单条文档完整 audit record"""
    rec = get_document_registry().get(doc_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="文档不存在或尚未开始处理")
    return _record_to_info(rec)


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    token_payload: dict = Depends(require_current_user),
) -> dict:
    """删除指定文档（磁盘 + Qdrant chunks + DocumentRegistry）"""
    rec = get_document_registry().get(doc_id)
    tenant_id = (
        token_payload.get("tenant_id", DEFAULT_TENANT_ID)
        if isinstance(token_payload, dict)
        else DEFAULT_TENANT_ID
    )

    # 1. 删除磁盘文件
    if rec is not None:
        for ext in ALLOWED_EXTENSIONS:
            candidate = DOCS_DIR / f"{doc_id}{ext}"
            if candidate.exists():
                candidate.unlink()
                break
    else:
        # registry 找不到时仍尝试按已知 ext 模式找
        for ext in ALLOWED_EXTENSIONS:
            candidate = DOCS_DIR / f"{doc_id}{ext}"
            if candidate.exists():
                candidate.unlink()
                break

    # 2. 从 Qdrant 删除该 doc_id 的所有 chunks
    try:
        from backend.config import get_config
        from backend.ingestion import QdrantIndexer

        cfg = get_config()
        indexer = QdrantIndexer(
            url=cfg.vector_db.url,
            collection_name=cfg.vector_db.collection_name,
            vector_size=cfg.vector_db.vector_size,
            distance=cfg.vector_db.distance,
        )
        indexer.delete_by_doc_id(doc_id)
    except Exception as e:
        logger.warning(f"Qdrant 删除失败（可能索引为空）: {e}")

    # 3. 从 DocumentRegistry 删除
    get_document_registry().delete(doc_id)

    return {"success": True, "message": f"文档 {doc_id} 已删除"}
