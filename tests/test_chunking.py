"""
test_chunking.py — 分块策略单元测试
================================================================================
"""

import pytest
from backend.ingestion.chunker import (
    Chunk,
    ChunkResult,
    RecursiveChunker,
    HierarchicalChunker,
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


class TestHierarchicalChunker:
    """层级分块策略测试"""

    def test_heading_extraction(self):
        """标题层级识别测试"""
        chunker = HierarchicalChunker(chunk_size=200, min_chunk_size=20, heading_levels=[1, 2])

        # text_units 是已按 \\n\\n 切分的段; 标题必须在段内
        # 段落要够长 (>20 tokens) 才会被产出, 否则会合并到前一个或被丢弃
        text_units = [
            "# 第一章\n这是第一章的内容。本章将详细介绍 RAG 系统的第一个核心组件即文档解析模块的工作原理、关键算法选择以及在生产环境中常见的优化技巧和性能瓶颈分析。",
            "## 第一节\n这是第一节的详细内容。本节会深入讨论 RAG 系统的第二个核心组件即混合检索模块的 BM25 算法与向量检索算法的融合策略与权重调优。",
            "# 第二章\n这是第二章的内容。本章聚焦 RAG 系统的第三个核心组件即 LLM 生成模块的 prompt 工程、结构化输出约束、引用标注机制和置信度评估方法。",
        ]
        chunks = chunker.split(text_units, "test_hier", {"headings": []})

        # 应该有至少 2 个 chunks
        assert len(chunks) >= 2

        # 检查 section_path 是否被正确填充
        paths = [c.section_path for c in chunks]
        assert any("第一章" in p for p in paths)


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
