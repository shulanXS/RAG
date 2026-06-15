"""文档解析与索引管道 — parser → chunker → embedder → indexer"""
from backend.ingestion.document_parser import (
    DocumentParser, DocumentParserFactory, ParsedDocument,
    SimHashDeduplicator, get_global_deduplicator,
)
from backend.ingestion.chunker import (
    ChunkResult, RecursiveChunker, get_chunker,
)
from backend.ingestion.embedder import Embedder
from backend.ingestion.indexer import QdrantIndexer
