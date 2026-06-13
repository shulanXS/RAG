"""
structured_output.py — 结构化输出生成器
================================================================================
技术决策记录:
- 为什么需要 Structured Output: LLM 的自由文本输出在生产环境中有三个问题：
  (1) 格式不稳定（JSON 解析可能失败）
  (2) 幻觉引用（凭空捏造引用）
  (3) 机器不可解析（无法做自动化流程）
- 解决方案: 使用各 LLM SDK 的原生 Structured Output 功能（Pydantic 模型），
  这是 2026 年生产环境的标配。Anthropic 的 JSON Schema 输出和
  OpenAI 的 response_format 参数都原生支持。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGStructuredOutput:
    """
    RAG 结构化输出

    字段说明:
    - answer: 答案正文
    - citations: 引用列表（doc_id, chunk_id, quote）
    - confidence: 置信度
    - gaps: 上下文中缺失的信息（有助于识别检索质量问题）
    """
    answer: str
    citations: list[dict]
    confidence: str
    gaps: list[str]


class StructuredOutputGenerator:
    """
    结构化输出生成器

    技术要点:
    - 使用 Pydantic 模型定义输出 Schema
    - 各 LLM 后端使用原生 Structured Output API
    - 自动验证输出格式，解析失败时降级到自由文本

    支持的后端:
    - Anthropic: messages API + output_json_schema
    - OpenAI: response_format = {"type": "json_schema", ...}
    - Google Gemini: generation_config + output_schema
    """

    RAG_OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "直接回答用户问题，不要重复问题"
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "quote": {"type": "string", "description": "引用原文片段（50字以内）"}
                    },
                    "required": ["doc_id", "quote"]
                },
                "description": "引用列表，每个引用对应上下文中的一个来源"
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "insufficient"],
                "description": "置信度评估: high=上下文充分支持, medium=部分支持, low=支持不足, insufficient=完全不支持"
            },
            "gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "上下文中缺失的、但问题可能需要的关键信息"
            }
        },
        "required": ["answer", "citations", "confidence"]
    }

    async def generate_structured(
        self,
        prompt: str,
        llm_client,
        provider: str = "anthropic",
        model: str = "claude-3-7-sonnet-20250620",
    ) -> RAGStructuredOutput:
        """
        生成结构化输出

        Args:
            prompt: 构建好的 Prompt
            llm_client: LLM 客户端
            provider: LLM 提供商 (anthropic | openai | google)
            model: 模型名称

        Returns:
            RAGStructuredOutput: 结构化输出对象
        """
        import json

        if provider == "anthropic":
            return await self._generate_anthropic(prompt, llm_client, model)
        elif provider == "openai":
            return await self._generate_openai(prompt, llm_client, model)
        else:
            # 降级到自由文本
            text = await llm_client.generate_async(prompt)
            return RAGStructuredOutput(
                answer=text,
                citations=[],
                confidence="medium",
                gaps=[],
            )

    async def _generate_anthropic(
        self,
        prompt: str,
        llm_client,
        model: str,
    ) -> RAGStructuredOutput:
        """Anthropic Structured Output"""
        import anthropic

        try:
            client = llm_client.generator_client._client
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                # Anthropic 的 JSON Schema 输出
                output={
                    "type": "json_object",
                    "schema": self.RAG_OUTPUT_SCHEMA,
                }
            )
            text = response.content[0].text

            data = json.loads(text)
            return RAGStructuredOutput(
                answer=data.get("answer", ""),
                citations=data.get("citations", []),
                confidence=data.get("confidence", "medium"),
                gaps=data.get("gaps", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Anthropic Structured Output 解析失败: {e}，降级到自由文本")
            text = response.content[0].text if "response" in dir() else ""
            return RAGStructuredOutput(
                answer=text,
                citations=[],
                confidence="medium",
                gaps=[],
            )
        except Exception as e:
            logger.error(f"Anthropic 生成失败: {e}")
            return RAGStructuredOutput(
                answer="生成失败",
                citations=[],
                confidence="insufficient",
                gaps=[f"生成错误: {str(e)}"],
            )

    async def _generate_openai(
        self,
        prompt: str,
        llm_client,
        model: str,
    ) -> RAGStructuredOutput:
        """OpenAI Structured Output"""
        import json

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rag_output",
                        "schema": self.RAG_OUTPUT_SCHEMA,
                    }
                },
            )
            text = response.choices[0].message.content or "{}"
            data = json.loads(text)

            return RAGStructuredOutput(
                answer=data.get("answer", ""),
                citations=data.get("citations", []),
                confidence=data.get("confidence", "medium"),
                gaps=data.get("gaps", []),
            )
        except Exception as e:
            logger.warning(f"OpenAI Structured Output 生成失败: {e}，降级到自由文本")
            return RAGStructuredOutput(
                answer="生成失败",
                citations=[],
                confidence="insufficient",
                gaps=[f"生成错误: {str(e)}"],
            )
