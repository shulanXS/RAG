"""
pipeline.py — 文档索引管道（用于 API 上传后的后台异步处理）
================================================================================
技术决策记录:
- 复用 scripts/ingest.py 的核心流程，但用 BackgroundTasks 触发，不阻塞 API 响应。
- 把 ingest 的核心 5 步封装成单一函数 `index_file_task`，便于单元测试。
- Contextual Retrieval 与 scripts/ingest.py 保持一致：每个文档先 LLM 生成摘要，
  再 prepend 到 chunk 前 embed。
- Sparse vector 写入遵循 P1.1：使用 index_chunks_with_sparse(Qdrant native BM25)。
- 错误处理：解析失败时不抛异常，只记日志并把 document status 标为 'failed'。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 模块级状态：indexed_documents 记录每个上传文档的处理状态
# -----------------------------------------------------------------------------
_indexed_documents: dict[str, dict] = {}


def get_document_status(file_id: str) -> dict | None:
    """获取指定文件 ID 的索引状态（用于 API 查询）"""
    return _indexed_documents.get(file_id)


def list_indexed_documents() -> list[dict]:
    """列出所有索引过的文档状态"""
    return list(_indexed_documents.values())


def reset_index_state() -> None:
    """清空索引状态（仅供测试）"""
    _indexed_documents.clear()


# -----------------------------------------------------------------------------
# 后台任务
# -----------------------------------------------------------------------------


async def index_file_task(
    file_path: Path | str,
    file_id: str,
    tenant_id: str = "default",
    strategy: str = "recursive",
) -> dict:
    """
    异步执行文档解析与索引。

    P1.3 修复: 之前 backend/api/documents.py:106-107 有 TODO 注释，
    上传文档后仅存盘但不处理。改用 FastAPI BackgroundTasks.add_task 调用本函数。

    流程:
    1. DocumentParserFactory.parse_directory([file_path])  →  ParsedDocument
    2. Embedder.generate_doc_summary()                       →  doc_summary
    3. get_chunker(strategy).split(...)                       →  list[Chunk]
    4. Embedder.embed_chunks_with_context(...)               →  list[vector]
    5. QdrantIndexer.index_chunks_with_sparse(...)           →  IndexedChunk[]

    Args:
        file_path: 已保存到磁盘的文件路径
        file_id: 文档 ID（与文件名一致）
        tenant_id: 多租户 ID
        strategy: 分块策略 (recursive | hierarchical | semantic)

    Returns:
        dict: 处理统计 {chunks_indexed, elapsed_ms, status}
    """
    import time

    t0 = time.perf_counter()
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    _indexed_documents[file_id] = {
        "file_id": file_id,
        "status": "processing",
        "started_at": int(time.time()),
    }

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")

        from backend.config import get_config
        from backend.ingestion import (
            DocumentParserFactory,
            Embedder,
            QdrantIndexer,
            get_chunker,
        )
        from backend.generation.llm_client import LLMClient

        config = get_config()

        # ---- 步骤 1: 文档解析 ----
        parser = DocumentParserFactory()
        documents = parser.parse_directory(file_path.parent)
        # 只保留与本文件相关的（filename 匹配）
        target_filename = file_path.name
        documents = [d for d in documents if d.metadata.get("source_path", "").endswith(target_filename)]
        # parse_directory 可能根据 suffix 自动过滤；如果没找到，就直接 parse 单个文件
        if not documents:
            documents = parser.parse_file(file_path)

        if not documents:
            raise ValueError(f"no parsable content in {file_path}")

        # ---- 步骤 2: Contextual LLM (用于生成文档摘要) ----
        contextual_llm = LLMClient(
            generator_provider=config.llm.generator.provider,
            generator_model=config.llm.generator.model,
            router_provider=config.llm.router.provider,
            router_model=config.llm.router.model,
        )

        embedder = Embedder(
            backend=config.embedding.backend,
            contextual_llm_client=contextual_llm.generator_client,
            contextual_prefix_tokens=config.embedding.contextual_prefix_tokens,
        )

        # ---- 步骤 3: 分块 + 文档摘要 ----
        chunker = get_chunker(
            strategy=strategy,
            config={
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
                "min_chunk_size": config.chunking.min_chunk_size,
                "heading_levels": config.chunking.heading_levels,
            },
        )

        all_chunks: list = []
        doc_summaries: dict[str, str] = {}
        for doc in documents:
            doc_summary = await embedder.generate_doc_summary(doc.content, doc.doc_id)
            doc_summaries[doc.doc_id] = doc_summary
            chunks = chunker.split(
                text_units=doc.text_units,
                doc_id=doc.doc_id,
                metadata={
                    "headings": doc.metadata.get("headings", []),
                    "doc_summary": doc_summary,
                },
            )
            valid_chunks = [c for c in chunks if c.token_count >= config.chunking.min_chunk_size]
            all_chunks.extend(valid_chunks)

        if not all_chunks:
            raise ValueError("no valid chunks after splitting")

        # ---- 步骤 4: Embedding (Contextual) ----
        # 按 doc_id 分组共享 doc_summary
        chunks_by_doc: dict[str, list] = {}
        for c in all_chunks:
            chunks_by_doc.setdefault(c.doc_id, []).append(c)

        all_embeddings: list[list[float]] = []
        ordered_chunks: list = []
        for doc_id, doc_chunks in chunks_by_doc.items():
            doc_summary = doc_summaries.get(doc_id, "")
            embs = embedder.embed_chunks_with_context(doc_chunks, doc_summary)
            all_embeddings.extend(embs)
            ordered_chunks.extend(doc_chunks)

        # ---- 步骤 5: Qdrant 索引（sparse + dense，P1.1） ----
        indexer = QdrantIndexer(
            url=config.vector_db.url,
            collection_name=config.vector_db.collection_name,
            vector_size=config.vector_db.vector_size,
            distance=config.vector_db.distance,
            batch_size=config.vector_db.batch_size,
        )
        indexer.ensure_collection(with_sparse_index=True)
        # 一次 upsert 整个文档的所有 chunks（跨多个 doc_id 的话可以循环）
        for doc_id, doc_chunks in chunks_by_doc.items():
            doc_embs = all_embeddings[
                sum(len(v) for k, v in list(chunks_by_doc.items())[:list(chunks_by_doc).index(doc_id)]) :
                sum(len(v) for k, v in list(chunks_by_doc.items())[: list(chunks_by_doc).index(doc_id) + 1])
            ]
            indexer.index_chunks_with_sparse(
                chunks=doc_chunks,
                dense_embeddings=doc_embs,
                doc_id=doc_id,
                tenant_id=tenant_id,
            )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        _indexed_documents[file_id] = {
            "file_id": file_id,
            "status": "indexed",
            "chunks": sum(len(v) for v in chunks_by_doc.values()),
            "elapsed_ms": elapsed_ms,
            "finished_at": int(time.time()),
        }
        logger.info(
            f"[index_file_task] {file_id} indexed "
            f"({sum(len(v) for v in chunks_by_doc.values())} chunks, {elapsed_ms}ms)"
        )
        return _indexed_documents[file_id]

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception(f"[index_file_task] {file_id} failed: {e}")
        _indexed_documents[file_id] = {
            "file_id": file_id,
            "status": "failed",
            "error": str(e)[:500],
            "elapsed_ms": elapsed_ms,
            "finished_at": int(time.time()),
        }
        return _indexed_documents[file_id]
