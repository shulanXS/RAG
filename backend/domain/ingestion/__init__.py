"""文档解析与索引管道 — parser → chunker → embedder → indexer"""
from backend.domain.ingestion.document_parser import (
    DocumentParser, DocumentParserFactory, ParsedDocument,
    SimHashDeduplicator, get_global_deduplicator,
)
from backend.domain.ingestion.chunker import (
    ChunkResult, RecursiveChunker, get_chunker,
)
from backend.domain.ingestion.embedder import Embedder
from backend.domain.ingestion.indexer import QdrantIndexer
