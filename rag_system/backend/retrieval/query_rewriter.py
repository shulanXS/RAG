"""
query_rewriter.py — 多轮对话查询改写器
================================================================================
技术决策记录:
- 为什么需要 Query Rewriting: 在多轮对话场景中，用户的追问往往是省略句
 （「那第二点呢？」），缺少主语和上下文，纯向量检索会失败。
- 实现方式: LLM 重写为完整独立问题。这是最有效但成本最高的方式。
- 备选方案:
  (1) 历史窗口拼接: 将对话历史直接拼接到当前查询前。
    缺点: 对话历史可能很长，消耗 token 且稀释语义。
  (2) 关键词抽取: 从历史中提取核心实体。
    缺点: 丢失语义关系。
- 决策: 使用 LLM 重写（质量最高），Haiku 成本可忽略（$0.8/1M tokens）。

业务难点:
- LLM 重写的质量不稳定: 有时会引入新的实体或误解原意。
  解决方案: 添加 self-check prompt，让 LLM 验证重写后的查询是否保留原意。
- 简单查询的处理: 对于单轮对话，直接跳过重写以节省成本。
  解决方案: 检测查询是否包含代词（我/你/它/这/那），有则重写。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    """
    改写后的查询结果

    字段说明:
    - rewritten: 改写后的完整问题
    - was_rewritten: 是否进行了改写（原查询即完整则返回原查询）
    - confidence: 重写置信度（0-1）
    - original: 原始查询
    """
    rewritten: str
    was_rewritten: bool
    confidence: float
    original: str


class QueryRewriter:
    """
    多轮对话查询改写器

    技术要点:
    - 代词检测: 通过正则检测中文代词（我/你/他/它/这/那/哪）和
      英文代词（it/this/that/they/we），有代词则触发重写
    - LLM 重写: Haiku 4.5 调用，生成完整独立问题
    - Self-check: 让 LLM 验证重写是否保留原意，不保留则回退到原查询

    风险考量:
    - 过度重写: 简单的事实型查询被不必要地重写，浪费 LLM 调用。
      缓解: 先检测代词，有代词才重写。
    - 重写错误: LLM 可能误解原意，生成完全不同的问题。
      缓解: 添加 self-check，置信度 < 0.7 时回退到原查询。
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: 可选，传入 LLM client 实例。
                       如果不传，跳过重写（返回原查询）。
        """
        self._llm = llm_client

    def rewrite(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> RewrittenQuery:
        """
        将多轮对话中的不完整查询改写为独立完整问题。

        Args:
            query: 当前用户查询
            conversation_history: 对话历史，格式: [{"role": "user"/"assistant", "content": "..."}]

        Returns:
            RewrittenQuery: 包含改写结果和置信度
        """
        # 步骤 1: 检测是否需要重写
        if not self._needs_rewriting(query):
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=1.0,
                original=query,
            )

        # 步骤 2: 如果没有 LLM client 或不完整查询，直接返回原查询
        if self._llm is None:
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.5,
                original=query,
            )

        # 步骤 3: 调用 LLM 重写
        return self._llm_rewrite(query, conversation_history or [])

    def _needs_rewriting(self, query: str) -> bool:
        """
        判断查询是否需要重写

        技术决策:
        - 代词检测: 中英文代词是省略句的主要标志
        - 字数检测: 少于 5 个词的查询很可能是不完整的追问
        - 问号后追加内容: 「...吗？」类型的问题通常不需要重写
        """
        # 英文代词
        en_pronouns = r"\b(i|you|he|she|it|we|they|this|that|these|those|what|which)\b"
        # 中文代词
        zh_pronouns = r"[这那它她他我你咱咱们的哪个哪个些哪个]"

        has_pronoun = bool(
            re.search(en_pronouns, query.lower()) or re.search(zh_pronouns, query)
        )

        # 短查询大概率需要重写
        is_short = len(query.split()) < 6 and len(query) < 15

        return has_pronoun or is_short

    def _llm_rewrite(
        self,
        query: str,
        history: list[dict],
    ) -> RewrittenQuery:
        """
        使用 LLM 重写查询

        提示词设计:
        - 明确角色: "你是一个查询改写专家"
        - 核心要求: 改写为完整、独立、可直接检索的问题
        - 约束: 不引入新实体，不改变原意
        - 历史上下文: 如果有对话历史，将其纳入考虑
        """
        # 构建提示词
        history_context = ""
        if history:
            history_context = "\n\n对话历史:\n" + "\n".join(
                f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
                for h in history[-3:]  # 只取最近 3 轮
            )

        prompt = f"""你是一个查询改写专家。请将不完整的多轮对话查询改写为完整、独立的问题。

{history_context}

当前查询: {query}

要求:
1. 将查询改写为一个完整、独立、可直接用于向量检索的问题
2. 如果当前查询已经是完整问题，直接返回原查询
3. 不要引入原查询中没有的信息
4. 如果原查询是完整问题，直接返回原查询，不要做任何修改

请以 JSON 格式输出，格式如下:
{{"rewritten": "改写后的完整问题（如果原问题已完整则同原问题）", "confidence": 0.0-1.0}}
"""

        try:
            response = self._llm.generate(
                prompt,
                max_tokens=256,
                temperature=0.1,
            )
            # 解析 JSON 响应
            import json
            result = json.loads(response.strip())
            rewritten = result.get("rewritten", query)
            confidence = float(result.get("confidence", 0.8))

            # 如果置信度过低，回退到原查询
            if confidence < 0.7:
                logger.debug(f"重写置信度过低 ({confidence})，回退到原查询")
                rewritten = query
                was_rewritten = False
            else:
                was_rewritten = rewritten != query

            return RewrittenQuery(
                rewritten=rewritten,
                was_rewritten=was_rewritten,
                confidence=confidence,
                original=query,
            )

        except Exception as e:
            logger.warning(f"查询重写失败: {e}，使用原查询")
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.0,
                original=query,
            )
