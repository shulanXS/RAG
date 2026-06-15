"""
pipeline.py — 文档索引管道（用于 API 上传后的后台异步处理）
================================================================================
技术决策记录:
|- 复用 scripts/ingest.py 的核心流程
|- 核心逻辑封装在 `run_index_pipeline`，便于 Arq worker 复用 (P1-A2)
|- Document metadata 现在持久化到 Qdrant DocumentRegistry (P1-A1)
  不再依赖模块级内存 dict
|- Contextual Retrieval: 每个文档先 LLM 生成摘要，再 prepend 到 chunk 前 embed
|- Sparse vector 写入遵循 P1.1：使用 Qdrant native BM25
|- 错误处理：解析失败时不抛异常，只记日志并把 document status 标为 'failed'
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 核心索引流程（被 API upload 与 Arq worker 共用）
# -----------------------------------------------------------------------------


async def run_index_pipeline(
    file_path: Path | str,
    file_id: str,
    tenant_id: str = "default",
    strategy: str = "recursive",
) -> dict:
    """
    同步执行文档解析与索引 (async 函数以兼容 FastAPI 异步上下文)。

    P1-A1: Document metadata 通过 DocumentRegistry 持久化。
    P1-A2: 仍可被 BackgroundTasks 调用，但推荐改用 Arq worker 调用。

    流程:
    1. DocumentParserFactory.parse_file(file_path) → ParsedDocument
    2. Embedder.generate_doc_summary() → doc_summary
    3. get_chunker(strategy).split(...) → list[Chunk]
    4. Embedder.embed_chunks_with_context(...) → list[vector]
    5. QdrantIndexer.index_chunks_with_sparse(...) → indexed
    6. DocumentRegistry.update_status(file_id, 'indexed', chunks=...)

    Args:
        file_path: 已保存到磁盘的文件路径
        file_id: 文档 ID（与文件名一致）
        tenant_id: 多租户 ID
        strategy: 分块策略 (recursive | hierarchical | semantic)

    Returns:
        dict: 处理统计 {file_id, chunks_indexed, elapsed_ms, status, error}
    """
    from backend.config import get_config
    from backend.domain.generation.llm_client import LLMClient
    from backend.domain.ingestion import (
        DocumentParserFactory,
        Embedder,
        QdrantIndexer,
        get_chunker,
    )
    from backend.domain.ingestion.document_registry import (
        DocumentRecord,
        get_document_registry,
    )

    t0 = time.perf_counter()
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    registry = get_document_registry()
    registry.update_status(file_id, "processing", started_at=int(time.time()))

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")

        config = get_config()

        # ---- 步骤 1: 文档解析 ----
        parser = DocumentParserFactory()
        documents = parser.parse_file(file_path)
        if not documents:
            raise ValueError(f"no parsable content in {file_path}")

        # ---- 步骤 2: Contextual LLM (用于生成文档摘要) ----
        contextual_llm = LLMClient(
            generator_provider=config.llm.generator.provider,
            generator_model=config.llm.generator.model,
            router_provider=config.llm.router.provider,
            router_model=config.llm.router.model,
            generator_api_key=config.llm.deepseek.api_key or None,
            generator_base_url=config.llm.deepseek.base_url,
            router_api_key=config.llm.deepseek.api_key or None,
            router_base_url=config.llm.deepseek.base_url,
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
            },
        )

        all_chunks = []
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
            valid_chunks = [
                c for c in chunks
                if c.token_count >= config.chunking.min_chunk_size
            ]
            all_chunks.extend(valid_chunks)

        if not all_chunks:
            raise ValueError("no valid chunks after splitting")

        # ---- 步骤 4: Embedding (Contextual) ----
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

        # 一次 upsert 整个文档的所有 chunks
        doc_ids = list(chunks_by_doc.keys())
        cursor = 0
        for doc_id in doc_ids:
            doc_chunks = chunks_by_doc[doc_id]
            n = len(doc_chunks)
            doc_embs = all_embeddings[cursor : cursor + n]
            cursor += n
            indexer.index_chunks_with_sparse(
                chunks=doc_chunks,
                dense_embeddings=doc_embs,
                doc_id=doc_id,
                tenant_id=tenant_id,
            )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        total_chunks = sum(len(v) for v in chunks_by_doc.values())
        registry.update_status(
            file_id,
            "indexed",
            chunks=total_chunks,
            elapsed_ms=elapsed_ms,
            finished_at=int(time.time()),
            error=None,
        )
        logger.info(
            f"[run_index_pipeline] {file_id} indexed "
            f"({total_chunks} chunks, {elapsed_ms}ms)"
        )
        return {
            "file_id": file_id,
            "chunks_indexed": total_chunks,
            "elapsed_ms": elapsed_ms,
            "status": "indexed",
        }

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception(f"[run_index_pipeline] {file_id} failed: {e}")
        registry.update_status(
            file_id,
            "failed",
            elapsed_ms=elapsed_ms,
            finished_at=int(time.time()),
            error=str(e)[:500],
        )
        return {
            "file_id": file_id,
            "chunks_indexed": 0,
            "elapsed_ms": elapsed_ms,
            "status": "failed",
            "error": str(e)[:500],
        }


# -----------------------------------------------------------------------------
# 旧的 `index_file_task` / `get_document_status` / `list_indexed_documents` /
# `reset_index_state` 入口已在 Phase1-1.10 删除（调用方已全部迁移到
# `run_index_pipeline` + `DocumentRegistry`）。
# -----------------------------------------------------------------------------
