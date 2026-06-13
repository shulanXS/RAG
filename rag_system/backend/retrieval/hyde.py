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

风险考量:
- 幻觉传播: 假设性答案中的虚假信息虽然不进入最终答案，
  但会影响 embedding，从而影响检索结果。
  缓解: 假设性答案仅用于 embedding，不作为参考上下文。
- 成本: 每条查询额外一次 LLM 调用（轻量生成，Haiku 即可）。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class HyDEQueryEnhancer:
    """
    HyDE 查询增强器

    用法:
        enhancer = HyDEQueryEnhancer(llm_client=haiku_client)
        hypothetical_doc = await enhancer.enhance("如何解决X的质量问题？")
        # hypothetical_doc ≈ "X的质量问题通常表现为...解决方法是..."
        # 用这个文本去做 embedding，而不是原始查询
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Haiku 或其他轻量 LLM，用于生成假设性答案
        """
        self._llm = llm_client

    async def enhance(self, query: str) -> str:
        """
        将查询增强为假设性文档文本。

        Args:
            query: 原始用户查询

        Returns:
            假设性文档文本（用于 embedding）
        """
        if self._llm is None:
            return query

        prompt = f"""你是一个文档生成器。请根据以下问题，生成一段假设性的文档内容。
这段内容应该是一个「假设这个问题的正确答案」，用于改进向量检索的准确性。
注意：这段内容不需要完全准确，它的作用是帮助找到相关的真实文档。

问题: {query}

请生成一段 3-5 句话的假设性答案（以陈述句风格撰写）:"""

        try:
            response = await self._llm.generate(
                prompt,
                max_tokens=200,
                temperature=0.7,
            )
            enhanced = response.strip()
            logger.debug(f"HyDE 增强: '{query}' → '{enhanced[:80]}...'")
            return enhanced
        except Exception as e:
            logger.warning(f"HyDE 增强失败: {e}，使用原始查询")
            return query

    async def batch_enhance(self, queries: list[str]) -> list[str]:
        """
        批量增强查询

        Args:
            queries: 查询列表

        Returns:
            增强后的文本列表
        """
        results = []
        for query in queries:
            enhanced = await self.enhance(query)
            results.append(enhanced)
        return results
