"""
test_generation.py — Generation 模块测试
"""

import pytest
from backend.generation.grounded_generator import (
    GroundedGenerator,
    GroundedResult,
    GroundedClaim,
)
from backend.generation.citation_generator import (
    SentenceLevelCitationExtractor,
    CitationGenerationResult,
    SentencedAnswer,
    SentenceSource,
)


class TestGroundedGenerator:
    """Grounded Generation 测试"""

    def test_fallback_generate_without_llm(self):
        """无 LLM client 时降级生成"""
        generator = GroundedGenerator(llm_client=None)

        contexts = [
            {"chunk_id": "c1", "text": "这是上下文内容", "doc_id": "d1"},
            {"chunk_id": "c2", "text": "这是另一段上下文", "doc_id": "d1"},
        ]

        result = generator._fallback_generate("测试查询", contexts)

        assert isinstance(result, GroundedResult)
        assert result.grounding_score < 1.0
        assert "测试查询" in result.answer

    def test_empty_contexts(self):
        """空上下文测试"""
        generator = GroundedGenerator(llm_client=None)

        result = generator._fallback_generate("查询", [])

        assert result.grounding_score == 0.0
        assert "无法找到" in result.answer

    def test_format_contexts(self):
        """上下文格式化测试"""
        generator = GroundedGenerator(llm_client=None)

        contexts = [
            {"chunk_id": "c1", "text": "第一段文本内容", "doc_id": "d1"},
            {"chunk_id": "c2", "text": "第二段文本内容", "doc_id": "d1"},
        ]

        formatted = generator._format_contexts(contexts)

        assert "[chunk_c1]" in formatted
        assert "[chunk_c2]" in formatted
        assert "第一段文本内容" in formatted

    def test_context_truncation(self):
        """上下文截断测试"""
        generator = GroundedGenerator(llm_client=None, max_context_tokens=50)

        long_text = "这是一段很长的文本内容。" * 50
        contexts = [{"chunk_id": "c1", "text": long_text, "doc_id": "d1"}]

        formatted = generator._format_contexts(contexts)

        assert len(formatted) <= generator._max_context_tokens * 4 + 100


class TestSentenceLevelCitationExtractor:
    """Sentence-level Citation 测试"""

    def test_sentence_split(self):
        """句子切分测试"""
        extractor = SentenceLevelCitationExtractor(llm_client=None)

        text = "这是第一句。这是第二句！这是第三句？"
        sentences = extractor._split_sentences(text)

        assert len(sentences) >= 3
        assert sentences[0] == "这是第一句"
        assert sentences[1] == "这是第二句"

    def test_create_sentenced_answer(self):
        """SentencedAnswer 创建测试"""
        extractor = SentenceLevelCitationExtractor(llm_client=None)

        sentences = ["这是第一句", "但是", "这是一段有意义的内容"]
        result = extractor._create_sentenced_answer(sentences)

        assert len(result) == 3
        assert result[1].is_citable is False  # 连接词

    def test_keyword_match(self):
        """关键词匹配测试"""
        extractor = SentenceLevelCitationExtractor(llm_client=None)

        sentence = "第一段文本内容包含关键信息"
        contexts = [
            {"chunk_id": "c1", "text": "第一段文本内容包含关键信息", "doc_id": "d1"},
            {"chunk_id": "c2", "text": "完全不相关的内容", "doc_id": "d1"},
        ]

        sources = extractor._keyword_match(sentence, contexts)

        assert len(sources) == 1
        assert sources[0].chunk_id == "c1"

    def test_empty_result(self):
        """空结果测试"""
        extractor = SentenceLevelCitationExtractor(llm_client=None)

        result = extractor.extract("", [])

        assert isinstance(result, CitationGenerationResult)
        assert result.answer == ""
        assert len(result.sentenced_answer) == 0

    def test_to_json(self):
        """JSON 序列化测试"""
        extractor = SentenceLevelCitationExtractor(llm_client=None)

        result = CitationGenerationResult(
            answer="这是答案",
            sentenced_answer=[
                SentencedAnswer(
                    text="这是句子",
                    sources=[SentenceSource(chunk_id="c1", doc_id="d1", quote="引用的内容")],
                    is_citable=True,
                    is_verified=True,
                )
            ],
            citations=[{"chunk_id": "c1", "doc_id": "d1", "quote": "引用的内容"}],
            citation_map={"c1": {"chunk_id": "c1", "doc_id": "d1", "quote": "引用的内容", "appears_in": []}},
            verification_results={"c1": True},
        )

        json_output = extractor.to_json(result)

        assert json_output["answer"] == "这是答案"
        assert len(json_output["sentences"]) == 1
        assert len(json_output["citations"]) == 1


class TestGroundedClaim:
    """GroundedClaim 数据结构测试"""

    def test_claim_creation(self):
        claim = GroundedClaim(
            text="这是声明内容",
            source_chunk_id="c1",
            source_doc_id="d1",
            quote="引用的原文",
            is_verifiable=True,
        )

        assert claim.text == "这是声明内容"
        assert claim.source_chunk_id == "c1"
        assert claim.is_verifiable is True


class TestSentencedAnswer:
    """SentencedAnswer 数据结构测试"""

    def test_sentenced_answer_creation(self):
        answer = SentencedAnswer(
            text="这是句子内容",
            sources=[SentenceSource(chunk_id="c1", doc_id="d1", quote="引用")],
            is_citable=True,
            is_verified=False,
        )

        assert answer.text == "这是句子内容"
        assert len(answer.sources) == 1
        assert answer.is_verified is False
