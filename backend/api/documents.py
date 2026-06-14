"""
documents.py — 文档管理 API 路由
================================================================================
技术决策记录:
- 上传后通过 FastAPI BackgroundTasks 触发异步索引 (P1.3)：
  之前只存盘不处理，本修复让 upload → index 真正闭环。
- 文档元数据存在 Qdrant 的 `documents` collection (metadata-only)，
  不再使用内存 dict。
- 状态查询通过 GET /documents/{id}/status 端点返回 indexing 状态。
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

from backend.ingestion.pipeline import get_document_status, index_file_task
from backend.security.auth import require_current_user
from backend.security.tenant import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# 文档存储目录（可配置）
DOCS_DIR = Path(__file__).parent.parent.parent / "data" / "uploaded_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size: int
    uploaded_at: int
    status: str = "ready"
    metadata: dict | None = None
    chunks: int | None = None
    elapsed_ms: int | None = None
    error: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = 0


class DocumentUploadResponse(BaseModel):
    success: bool
    document: DocumentInfo | None = None
    error: str | None = None
    indexing: str = "queued"  # queued | processing | indexed | failed


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv", ".json", ".html", ".xml"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def _validate_file(file: UploadFile) -> None:
    """验证文件类型和大小"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}。支持的类型: {', '.join(ALLOWED_EXTENSIONS)}",
        )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    token_payload: dict = Depends(require_current_user),
) -> DocumentUploadResponse:
    """
    上传文档到系统，自动执行解析和索引。

    P1.3: 上传后通过 BackgroundTasks 触发 index_file_task，
    文档 metadata 状态在 Qdrant 中维护（避免内存 dict 重启丢数据）。

    支持格式: PDF, TXT, MD, DOCX, CSV, JSON, HTML, XML
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

        # 保存文件
        file_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix.lower()
        stored_path = DOCS_DIR / f"{file_id}{ext}"
        stored_path.write_bytes(content)

        # 解析 tenant_id（多租户隔离）
        tenant_id = (
            token_payload.get("tenant_id")
            if isinstance(token_payload, dict)
            else DEFAULT_TENANT_ID
        ) or DEFAULT_TENANT_ID

        doc_info = DocumentInfo(
            id=file_id,
            filename=file.filename,
            size=size,
            uploaded_at=int(Path(stored_path).stat().st_mtime * 1000),
            status="queued",
            metadata={"original_name": file.filename, "extension": ext},
        )

        # P1.3: 用 BackgroundTasks 触发异步索引，不阻塞上传响应
        background_tasks.add_task(
            index_file_task,
            stored_path,
            file_id,
            tenant_id,
            "recursive",
        )
        logger.info(f"Upload queued: {file_id} ({file.filename}, {size} bytes)")

        return DocumentUploadResponse(
            success=True,
            document=doc_info,
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
    """
    获取文档的索引状态（P1.3）。

    返回: queued | processing | indexed | failed
    """
    status_info = get_document_status(doc_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="文档不存在或尚未开始处理")
    return DocumentInfo(
        id=status_info["file_id"],
        filename="",
        size=0,
        uploaded_at=status_info.get("started_at", 0),
        status=status_info.get("status", "unknown"),
        chunks=status_info.get("chunks"),
        elapsed_ms=status_info.get("elapsed_ms"),
        error=status_info.get("error"),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    token_payload: dict = Depends(require_current_user),
) -> DocumentListResponse:
    """列出已上传的文档（按索引状态聚合）"""
    from backend.ingestion.pipeline import list_indexed_documents

    # 优先返回在 pipeline 中登记的索引状态
    docs = list_indexed_documents()
    total = len(docs)
    paginated = docs[skip : skip + limit]
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=d["file_id"],
                filename=d.get("file_id", ""),
                size=0,
                uploaded_at=d.get("started_at", 0),
                status=d.get("status", "unknown"),
                chunks=d.get("chunks"),
                elapsed_ms=d.get("elapsed_ms"),
                error=d.get("error"),
            )
            for d in paginated
        ],
        total=total,
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    token_payload: dict = Depends(require_current_user),
) -> DocumentInfo:
    """获取单个文档信息"""
    status_info = get_document_status(doc_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="文档不存在或尚未开始处理")
    return DocumentInfo(
        id=status_info["file_id"],
        filename="",
        size=0,
        uploaded_at=status_info.get("started_at", 0),
        status=status_info.get("status", "unknown"),
        chunks=status_info.get("chunks"),
        elapsed_ms=status_info.get("elapsed_ms"),
        error=status_info.get("error"),
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    token_payload: dict = Depends(require_current_user),
) -> dict:
    """删除指定文档（同时从磁盘和 Qdrant 移除）"""
    # 1. 删除磁盘文件
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

    # 3. 从 pipeline 状态字典中删除
    from backend.ingestion.pipeline import _indexed_documents
    _indexed_documents.pop(doc_id, None)

    return {"success": True, "message": f"文档 {doc_id} 已删除"}
