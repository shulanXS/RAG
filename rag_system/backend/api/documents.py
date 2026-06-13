"""
documents.py — 文档管理 API 路由
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# 文档存储目录（可配置）
DOCS_DIR = Path(__file__).parent.parent.parent / "data" / "uploaded_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 内存中的文档索引（生产环境应替换为数据库）
_documents_index: dict[str, dict] = {}


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size: int
    uploaded_at: int
    status: str = "ready"
    metadata: dict | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = 0


class DocumentUploadResponse(BaseModel):
    success: bool
    document: DocumentInfo | None = None
    error: str | None = None


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
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    上传文档到系统，自动执行解析和索引。

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

        doc_info = DocumentInfo(
            id=file_id,
            filename=file.filename,
            size=size,
            uploaded_at=int(Path(stored_path).stat().st_mtime * 1000),
            status="ready",
            metadata={"original_name": file.filename, "extension": ext},
        )

        _documents_index[file_id] = doc_info.model_dump()

        # TODO: 触发异步解析和索引流程
        # background_tasks.add_task(index_document, stored_path, file_id)

        return DocumentUploadResponse(success=True, document=doc_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Document upload failed: {e}")
        return DocumentUploadResponse(success=False, error=str(e))


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DocumentListResponse:
    """列出所有已上传的文档"""
    docs = list(_documents_index.values())
    total = len(docs)
    paginated = docs[skip : skip + limit]

    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in paginated],
        total=total,
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str) -> DocumentInfo:
    """获取单个文档信息"""
    if doc_id not in _documents_index:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentInfo(**_documents_index[doc_id])


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """删除指定文档"""
    if doc_id not in _documents_index:
        raise HTTPException(status_code=404, detail="文档不存在")

    doc_info = _documents_index[doc_id]
    ext = doc_info.get("metadata", {}).get("extension", "")
    stored_path = DOCS_DIR / f"{doc_id}{ext}"

    if stored_path.exists():
        stored_path.unlink()

    del _documents_index[doc_id]

    return {"success": True, "message": f"文档 {doc_id} 已删除"}
