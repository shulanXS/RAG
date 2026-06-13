"""
grounded_generator.py — Grounded Generation (基于上下文的严格生成)
================================================================================
技术决策记录:
- Grounded Generation 是 FAANG RAG 减少幻觉的核心机制。
- 核心思想: 强制 LLM 只使用检索到的上下文中的信息生成答案，
  明确禁止使用外部知识（虽然 LLM 实际上无法完全遵守，但通过 prompt 设计可以显著减少）。
- 与 Self-Reflection 的关系: Grounded 是生成时的约束，Self-Reflection 是生成后的验证。

Prompt 设计要点:
1. 明确禁止 LLM 使用外部知识
2. 每个 claim 必须标注来源 [来源: chunk_id]
3. 如果上下文不足，明确说 "我无法从提供的文档中找到..."
4. groundness_score: 衡量上下文对答案的支撑程度

技术方案:
- 使用 DeepSeek 作为首选 LLM（成本极低）
- 生成结果包含 grounded_claims（每个 claim 的来源链）
- knowledge_gaps: 无法回答的部分
- grounding_score: 0-1 的支撑度分数
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class GroundedClaim:
    """
    有来源支撑的声明

    字段说明:
    - text: 声明文本
    - source_chunk_id: 来源 chunk ID
    - source_doc_id: 来源文档 ID
    - quote: 原文引用
    - is_verifiable: 是否可验证（上下文中有支撑）
    """
    text: str
    source_chunk_id: str = ""
    source_doc_id: str = ""
    quote: str = ""
    is_verifiable: bool = True


@dataclass
class GroundedResult:
    """
    Grounded Generation 结果

    字段说明:
    - answer: 带来源标注的答案
    - grounded_claims: 每个 claim 的来源链
    - knowledge_gaps: 无法回答的部分
    - grounding_score: 上下文支撑度 (0-1)
    - missing_aspects: 缺失的方面
    """
    answer: str
    grounded_claims: list[GroundedClaim] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    grounding_score: float = 0.0
    missing_aspects: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)


class GroundedGenerator:
    """
    基于上下文的严格生成器

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 构建 Grounded Prompt                                     │
    │     - 注入检索上下文                                          │
    │     - 明确禁止外部知识                                        │
    │     - 要求句级引用标注                                       │
    │  2. LLM 生成                                                │
    │  3. 解析答案 + 提取 claim 来源                               │
    │  4. 计算 grounding_score                                     │
    │  5. 识别 knowledge_gaps                                      │
    └─────────────────────────────────────────────────────────────┘

    设计要点:
    - 使用 DeepSeek 作为首选 LLM
    - 支持 strict_mode: 严格模式（完全禁止外部知识）vs 宽松模式（允许常识）
    - 生成结果可直接用于前端渲染（带引用的答案）
    """

    GROUNDING_PROMPT_TEMPLATE = """你是一个严格基于提供的上下文回答问题的助手。

【核心规则】
1. 只使用上下文中的信息，不要引入外部知识或你的训练数据中的信息
2. 每个关键陈述必须标注来源: [来源: chunk_xxx]
3. 如果某个方面上下文无法回答，明确说明 "在提供的文档中无法找到关于...的信息"
4. 保持简洁、直接、专业

【上下文】
{context}

【用户问题】
{query}

【要求】
1. 回答必须直接基于上述上下文，不要添加未在上下文中出现的信息
2. 对于数据/数字/日期，必须确保在上下文中存在
3. 如果上下文中有矛盾的信息，指出这一点
4. 在回答末尾，如果有无法回答的方面，列出 "知识缺口: ..."

请以 JSON 格式输出:
{{
  "answer": "完整答案（包含来源标注）",
  "knowledge_gaps": ["无法回答的方面1", "无法回答的方面2"],
  "grounding_score": 0.0-1.0,
  "citations": [
    {{"chunk_id": "xxx", "quote": "引用的原文"}},
    ...
  ]
}}"""

    def __init__(
        self,
        llm_client=None,
        strict_mode: bool = True,
        max_context_tokens: int = 8000,
    ):
        """
        Args:
            llm_client: DeepSeek LLM client
            strict_mode: 严格模式（完全禁止外部知识）
            max_context_tokens: 最大上下文 token 数
        """
        self._llm = llm_client
        self._strict_mode = strict_mode
        self._max_context_tokens = max_context_tokens

    async def generate(
        self,
        query: str,
        contexts: list[dict],
        strict_mode: bool | None = None,
    ) -> GroundedResult:
        """
        基于上下文的严格生成

        Args:
            query: 用户查询
            contexts: 检索到的上下文列表
            strict_mode: 覆盖默认的严格模式设置

        Returns:
            GroundedResult: 包含答案、claim 来源、知识缺口、支撑分数
        """
        if not contexts:
            return GroundedResult(
                answer="在提供的文档中无法找到相关信息。",
                grounding_score=0.0,
                knowledge_gaps=["无相关文档"],
            )

        if self._llm is None:
            return self._fallback_generate(query, contexts)

        use_strict = strict_mode if strict_mode is not None else self._strict_mode

        try:
            return await self._llm_generate(query, contexts, use_strict)
        except Exception as e:
            logger.warning(f"Grounded generation failed: {e}")
            return self._fallback_generate(query, contexts)

    async def _llm_generate(
        self,
        query: str,
        contexts: list[dict],
        strict_mode: bool,
    ) -> GroundedResult:
        """使用 LLM 进行 grounded 生成"""
        import json

        context_text = self._format_contexts(contexts)
        prompt = self.GROUNDING_PROMPT_TEMPLATE.format(
            context=context_text,
            query=query,
        )

        if strict_mode:
            prompt += "\n\n【重要】严格模式: 不要使用任何超出上下文的知识。"

        response = await self._llm.generate(
            prompt,
            max_tokens=1024,
            temperature=0.2,
        )

        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse grounded generation JSON: {e}")
            return GroundedResult(
                answer=response,
                grounding_score=0.5,
            )

        citations = []
        for c in data.get("citations", []):
            if isinstance(c, dict):
                citations.append({
                    "chunk_id": c.get("chunk_id", ""),
                    "doc_id": c.get("doc_id", ""),
                    "quote": c.get("quote", ""),
                })

        grounded_claims = []
        for c in citations:
            grounded_claims.append(GroundedClaim(
                text="",
                source_chunk_id=c.get("chunk_id", ""),
                source_doc_id=c.get("doc_id", ""),
                quote=c.get("quote", ""),
            ))

        return GroundedResult(
            answer=data.get("answer", ""),
            grounded_claims=grounded_claims,
            knowledge_gaps=data.get("knowledge_gaps", []),
            grounding_score=float(data.get("grounding_score", 0.5)),
            citations=citations,
        )

    def _fallback_generate(
        self,
        query: str,
        contexts: list[dict],
    ) -> GroundedResult:
        """降级方案：简单拼接上下文"""
        context_text = self._format_contexts(contexts)

        return GroundedResult(
            answer=f"基于检索到的文档回答 '{query}':\n\n{context_text[:500]}...",
            grounding_score=0.3,
            knowledge_gaps=["LLM 不可用，降级处理"],
        )

    def _format_contexts(self, contexts: list[dict]) -> str:
        """格式化上下文文本"""
        if not contexts:
            return "（无相关文档）"

        parts = []
        for i, ctx in enumerate(contexts[:10], 1):
            if isinstance(ctx, dict):
                chunk_id = ctx.get("chunk_id", f"chunk_{i}")
                text = ctx.get("text", "")
                section = ctx.get("section_path", "")
                parts.append(f"[chunk_{chunk_id}] {section}\n{text[:400]}")
            else:
                parts.append(str(ctx)[:400])

        total = "\n\n---\n\n".join(parts)

        if len(total) > self._max_context_tokens * 4:
            total = total[: self._max_context_tokens * 4] + "\n\n(...)"

        return total

    async def generate_with_verification(
        self,
        query: str,
        contexts: list[dict],
    ) -> tuple[GroundedResult, GroundedResult]:
        """
        生成 + 验证双流程

        第一轮: 生成答案
        第二轮: 验证答案中的每个声明是否被上下文支撑

        Returns:
            (generation_result, verification_result)
        """
        generation = await self.generate(query, contexts)

        verification = await self._verify_claims(query, contexts, generation)

        combined_score = (
            generation.grounding_score * 0.6 +
            verification.grounding_score * 0.4
        )

        if verification.grounding_score < 0.5:
            generation.grounding_score = combined_score
            generation.knowledge_gaps.extend(verification.knowledge_gaps)

        return generation, verification

    async def _verify_claims(
        self,
        query: str,
        contexts: list[dict],
        generation: GroundedResult,
    ) -> GroundedResult:
        """验证生成的声明是否被上下文支撑"""
        import json

        if not generation.grounded_claims and not generation.citations:
            return GroundedResult(grounding_score=1.0)

        context_text = self._format_contexts(contexts)

        prompt = f"""你是一个答案验证专家。请验证以下答案中的每个声明是否被上下文支撑。

用户问题: {query}

上下文:
{context_text}

答案:
{generation.answer}

请检查:
1. 每个关键声明是否有上下文支撑？
2. 是否存在无法证实的声明（幻觉）？
3. 上下文是否充分覆盖了问题？

请以 JSON 格式输出:
{{
  "grounding_score": 0.0-1.0,
  "verified_claims": ["可证实的声明1", ...],
  "unverifiable_claims": ["无法证实的声明1", ...],
  "knowledge_gaps": ["知识缺口1", ...]
}}"""

        try:
            response = await self._llm.generate(
                prompt,
                max_tokens=512,
                temperature=0.1,
            )

            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                data = json.loads(text)

            return GroundedResult(
                grounding_score=float(data.get("grounding_score", 0.5)),
                knowledge_gaps=data.get("knowledge_gaps", []),
            )
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return GroundedResult(grounding_score=0.5)
