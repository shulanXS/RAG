"""
hyde.py — HyDE (Hypothetical Document Embeddings)
================================================================================
技术决策记录:
- HyDE 的核心洞察: 用户查询（问句）和文档（陈述句）的语义分布存在差异。
  问句用「如何」「为什么」开头，陈述句用名词开头，embedding 空间中的
  位置不同。通过生成假设性答案（陈述句风格），可以让 embedding 更接近文档分布。
- 适用场景: 语义模糊的查询。「关于最近X的质量问题」→ 生成「X质量问题表现为...」
- 不适用场景: 精确关键词查询（「合同编号 A-2024-001」），HyDE 会引入幻觉干扰。
- 决策: HyDE 作为可选策略，由 QueryComplexityRouter 决定何时启用。
  复杂度低的简单查询跳过 HyDE（节省 LLM 调用 + 避免幻觉）。
- 增强: 新增多假设生成、假设置信度评估、与 Query Expansion 集成。

风险考量:
- 幻觉传播: 假设性答案中的虚假信息虽然不进入最终答案，
  但会影响 embedding，从而影响检索结果。
  缓解: 假设性答案仅用于 embedding，不作为参考上下文。
- 成本: 每条查询额外一次 LLM 调用（轻量生成，Haiku 即可）。
- DeepSeek: 使用 DeepSeek 生成假设性答案，成本极低。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class HyDEHypothesis:
    """
    假设性文档假设

    字段说明:
    - text: 假设性文档内容
    - confidence: 假设的置信度 (0-1)，用于后续融合
    - approach: 生成方式 (deductive/inductive/reverse)
    """
    text: str
    confidence: float = 0.5
    approach: Literal["deductive", "inductive", "reverse"] = "deductive"


@dataclass
class HyDEResult:
    """
    HyDE 增强结果

    字段说明:
    - original_query: 原始查询
    - hypotheses: 生成的假设性文档列表
    - combined_text: 所有假设的融合文本（用于 embedding）
    - hyde_score: HyDE 增强的有效性分数
    """
    original_query: str
    hypotheses: list[HyDEHypothesis] = field(default_factory=list)
    combined_text: str = ""
    hyde_score: float = 0.0


class HyDEQueryEnhancer:
    """
    HyDE 查询增强器

    用法:
        enhancer = HyDEQueryEnhancer(llm_client=deepseek_client)
        result = await enhancer.enhance("如何解决X的质量问题？")
        # result.hypotheses = [HyDEHypothesis("X的质量问题通常表现为..."), ...]
        # result.combined_text = 融合后的检索文本
    """

    def __init__(
        self,
        llm_client=None,
        max_hypotheses: int = 3,
        confidence_threshold: float = 0.3,
    ):
        """
        Args:
            llm_client: DeepSeek LLM，用于生成假设性答案
            max_hypotheses: 最大假设数量
            confidence_threshold: 假设置信度阈值，低于此值的假设被过滤
        """
        self._llm = llm_client
        self._max_hypotheses = max_hypotheses
        self._confidence_threshold = confidence_threshold

    async def enhance(self, query: str) -> str:
        """
        将查询增强为假设性文档文本（保持向后兼容）

        Args:
            query: 原始用户查询

        Returns:
            假设性文档文本（用于 embedding）
        """
        result = await self.generate(query)
        return result.combined_text or query

    async def generate(self, query: str) -> HyDEResult:
        """
        生成多个假设性文档

        Args:
            query: 原始用户查询

        Returns:
            HyDEResult: 包含多个假设及融合文本
        """
        if not query or not query.strip():
            return HyDEResult(original_query=query)

        if self._llm is None:
            return HyDEResult(original_query=query, combined_text=query)

        try:
            hypotheses = await self._generate_hypotheses(query)
            if not hypotheses:
                return HyDEResult(original_query=query, combined_text=query)

            combined_text = self._combine_hypotheses(hypotheses)
            hyde_score = self._estimate_hyde_effectiveness(query, hypotheses)

            return HyDEResult(
                original_query=query,
                hypotheses=hypotheses,
                combined_text=combined_text,
                hyde_score=hyde_score,
            )

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return HyDEResult(original_query=query, combined_text=query)

    async def _generate_hypotheses(self, query: str) -> list[HyDEHypothesis]:
        """生成多个不同风格的假设性文档"""
        prompt = f"""你是一个文档生成器。请根据以下问题，生成 {self._max_hypotheses} 段不同风格的假设性文档内容。

要求:
1. 每段内容应该是「假设这个问题已经被解决了/某个事实已经被确认了」后的文档片段
2. 使用不同的生成策略:
   -  deductive: 从一般到特殊的演绎风格
   -  inductive: 从特殊到一般的归纳风格
   -  reverse: 从结论倒推的反向风格
3. 每段 2-4 句话，以陈述句风格撰写
4. 不需要完全准确，只需捕捉核心语义即可

问题: {query}

请以 JSON 数组格式输出，每段包含:
- "text": 假设性文档内容 (英文或中英文混合)
- "approach": 风格 (deductive/inductive/reverse)
- "confidence": 置信度 (0.0-1.0)

示例输出格式:
[
  {{"text": "The quality issues of X typically manifest as...", "approach": "deductive", "confidence": 0.7}},
  {{"text": "Based on the available data, X's quality problems include...", "approach": "inductive", "confidence": 0.6}},
  {{"text": "To resolve X quality issues, the following steps are recommended...", "approach": "reverse", "confidence": 0.5}}
]"""

        import json

        response = await self._llm.generate(
            prompt,
            max_tokens=512,
            temperature=0.7,
        )

        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [
                    HyDEHypothesis(
                        text=item.get("text", ""),
                        confidence=float(item.get("confidence", 0.5)),
                        approach=item.get("approach", "deductive"),
                    )
                    for item in data
                    if item.get("text") and float(item.get("confidence", 0)) >= self._confidence_threshold
                ]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse HyDE response: {e}")

        return [
            HyDEHypothesis(
                text=await self._generate_single_hypothesis(query, "deductive"),
                confidence=0.5,
                approach="deductive",
            )
        ]

    async def _generate_single_hypothesis(
        self,
        query: str,
        approach: str,
    ) -> str:
        """生成单个假设性文档"""
        prompt = f"""Based on the question below, generate a hypothetical document passage that would answer this question.

The passage should be written in a factual, declarative style as if it were extracted from a real document.

Question: {query}

Generate 2-4 sentences in English:"""

        try:
            response = await self._llm.generate(
                prompt,
                max_tokens=200,
                temperature=0.7,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Single hypothesis generation failed: {e}")
            return query

    def _combine_hypotheses(self, hypotheses: list[HyDEHypothesis]) -> str:
        """融合多个假设为单一检索文本"""
        if not hypotheses:
            return ""

        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

        parts = [h.text for h in sorted_hypotheses if h.text]

        if len(parts) == 1:
            return parts[0]

        combined = "\n\n".join(parts)
        return combined[:1500]

    def _estimate_hyde_effectiveness(
        self,
        query: str,
        hypotheses: list[HyDEHypothesis],
    ) -> float:
        """估算 HyDE 增强的有效性"""
        if not hypotheses:
            return 0.0

        avg_confidence = sum(h.confidence for h in hypotheses) / len(hypotheses)
        diversity = len(set(h.approach for h in hypotheses)) / 3.0

        return min(1.0, avg_confidence * 0.7 + diversity * 0.3)

    async def batch_enhance(self, queries: list[str]) -> list[str]:
        """
        批量增强查询

        Args:
            queries: 查询列表

        Returns:
            增强后的文本列表
        """
        import asyncio

        results = await asyncio.gather(
            *[self.enhance(q) for q in queries],
            return_exceptions=True,
        )

        return [
            r if isinstance(r, str) else q
            for r, q in zip(results, queries)
        ]

    async def enhance_for_hybrid(
        self,
        query: str,
        original_embedding: list[float] | None = None,
    ) -> tuple[str, list[float], HyDEResult]:
        """
        增强查询并返回用于混合检索的文本

        Args:
            query: 原始查询
            original_embedding: 原始查询的 embedding (可选，用于比较)

        Returns:
            (combined_text, hyde_embedding, hyde_result)
        """
        from backend.ingestion.embedder import Embedder

        result = await self.generate(query)
        combined = result.combined_text or query

        hyde_embedding = None
        if self._llm is not None:
            try:
                embedder = Embedder(backend="openai")
                hyde_embedding = embedder.embed(combined)
            except Exception as e:
                logger.warning(f"HyDE embedding failed: {e}")

        return combined, hyde_embedding, result
