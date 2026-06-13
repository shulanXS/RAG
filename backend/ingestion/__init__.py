"""
Ingestion 模块 — 文档解析与索引管道
================================================================================
技术决策记录:
- 分层架构: parser(解析) → chunker(分块) → contextual_embed(上下文增强)
  → embedder(向量化) → indexer(入库)。每层职责单一，可独立测试和替换。
- 多格式支持: PDF/DOCX/MD/HTML 是企业文档的绝对主流格式。
  不追求 100+ 格式的全面覆盖（那是 Unstructured 的职责），专注于这 4 种。
- 类型标注: 每个处理步骤都有明确的输入/输出类型，mypy 可做静态检查。
"""

from backend.ingestion.document_parser import (
    DocumentParser,
    DocumentParserFactory,
    ParsedDocument,
    SimHashDeduplicator,
    get_global_deduplicator,
)
from backend.ingestion.chunker import ChunkResult, HierarchicalChunker, RecursiveChunker, SemanticChunker, get_chunker
from backend.ingestion.embedder import Embedder
from backend.ingestion.indexer import QdrantIndexer

__all__ = [
    "DocumentParser",
    "DocumentParserFactory",
    "ParsedDocument",
    "SimHashDeduplicator",
    "get_global_deduplicator",
    "ChunkResult",
    "HierarchicalChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "get_chunker",
    "Embedder",
    "QdrantIndexer",
]
