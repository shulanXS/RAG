"""
generation 模块 — 生成层
================================================================================
技术决策记录:
- 分层 LLM 策略: Router 用 Haiku (快速+便宜)，Generator 用 Sonnet (质量)。
- Structured Output: 直接在各 SDK 调用中传 JSON Schema（Anthropic/OpenAI 均原生支持），
  统一在 LLMClient 内部处理，无需单独的 StructuredOutputGenerator 类。
- 移除项 (P0): structured_output.py — LLMClient 已内置 schema 透传，单独类徒增抽象。
"""

from backend.generation.llm_client import LLMClient
from backend.generation.prompt_builder import PromptBuilder
from backend.generation.citation_verifier import CitationVerifier

__all__ = [
    "LLMClient",
    "PromptBuilder",
    "CitationVerifier",
]
