"""
test_chunking.py — 分块策略单元测试
================================================================================
P1-3: HierarchicalChunker / SemanticChunker 已删除（仅保留 RecursiveChunker）。
"""

import pytest
from backend.ingestion.chunker import (
    Chunk,
    ChunkResult,
    RecursiveChunker,
    count_tokens,
)


class TestRecursiveChunker:
    """Recursive 分块策略测试"""

    def test_basic_split(self):
        """基本拆分测试"""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=30)

        text = "这是第一段内容。" * 20 + "\n\n" + "这是第二段内容。" * 20
        chunks = chunker.split(
            text_units=text.split("\n\n"),
            doc_id="test_doc",
            metadata={},
        )

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.token_count >= 30 for c in chunks)

    def test_min_chunk_size_filter(self):
        """最小块大小过滤测试"""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=50)

        # 碎片文本
        text_units = ["短", "更短", "极短文本"]
        chunks = chunker.split(text_units, "test", {})

        # 碎片文本可能被合并或丢弃
        for c in chunks:
            assert c.token_count >= 30  # min_chunk_size - overlap 允许范围内

    def test_token_counting(self):
        """Token 计数测试"""
        text = "这是一个测试句子。" * 5
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # token 数应小于字符数

    def test_chunk_metadata(self):
        """Chunk metadata 填充测试"""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=30)
        text_units = ["测试内容" * 50]
        chunks = chunker.split(text_units, "doc_123", {"section_path": "第一章/第一节"})

        if chunks:
            c = chunks[0]
            assert c.doc_id == "doc_123"
            assert c.section_path == "第一章/第一节"
            assert c.chunk_index >= 0


class TestTokenCounting:
    """Token 计数测试"""

    def test_english_text(self):
        """英文文本 token 计数"""
        text = "This is a test sentence. " * 10
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < len(text) / 2  # 英文约 4 char/token

    def test_chinese_text(self):
        """中文文本 token 计数"""
        text = "这是一个测试句子。" * 10
        tokens = count_tokens(text)
        assert tokens > 0
        # 中文约 2 char/token
        assert tokens > len(text) / 3

    def test_mixed_text(self):
        """中英混合文本 token 计数"""
        text = "这是中文。" * 5 + "This is English. " * 5
        tokens = count_tokens(text)
        assert tokens > 0

    def test_empty_text(self):
        """空文本处理"""
        assert count_tokens("") == 0
        assert count_tokens("   ") == 0
