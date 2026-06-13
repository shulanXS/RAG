"""
generation 模块 — 生成层
================================================================================
技术决策记录:
- 分层 LLM 策略: Router 用 Haiku (快速+便宜)，Generator 用 Sonnet (质量)。
- Structured Output: JSON Schema 约束 LLM 输出，避免解析错误和幻觉引用。
- Prompt Template: 引用标注格式、置信度评估、信息缺口识别。
"""

from backend.generation.llm_client import LLMClient
from backend.generation.prompt_builder import PromptBuilder
from backend.generation.structured_output import StructuredOutputGenerator, RAGStructuredOutput
from backend.generation.streaming import LLMStreamer
from backend.generation.grounded_generator import GroundedGenerator, GroundedResult, GroundedClaim
from backend.generation.citation_generator import (
    SentenceLevelCitationExtractor,
    CitationGenerationResult,
    SentencedAnswer,
    SentenceSource,
)

__all__ = [
    "LLMClient",
    "PromptBuilder",
    "StructuredOutputGenerator",
    "RAGStructuredOutput",
    "LLMStreamer",
    "GroundedGenerator",
    "GroundedResult",
    "GroundedClaim",
    "SentenceLevelCitationExtractor",
    "CitationGenerationResult",
    "SentencedAnswer",
    "SentenceSource",
]
