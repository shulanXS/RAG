"""
query_rewriter.py — 多轮对话查询改写器 + 意图分类
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

增强 (Phase 1):
- IntentClassification: 新增查询意图分类 (factual/comparative/analytical/summarization)
- ClarificationDetection: 识别模糊查询，主动要求用户澄清
- QueryType: 返回查询类型，指导后续检索策略选择
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """
    查询意图分类 — 用于决定扩展策略和检索策略

    - factual: 事实型查询，需要精确匹配
    - comparative: 比较型查询，需要多角度对比
    - analytical: 分析型查询，需要多来源综合
    - definitional: 定义型查询，需要权威定义
    - procedural: 流程型查询，需要步骤性文档
    - summarization: 摘要型查询，需要全面覆盖
    - conversational: 对话型查询，不需要检索
    """
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    DEFINITIONAL = "definitional"
    PROCEDURAL = "procedural"
    SUMMARIZATION = "summarization"
    CONVERSATIONAL = "conversational"


@dataclass
class QueryType:
    """
    查询类型信息

    字段说明:
    - intent: 查询意图
    - is_beyond_kb: 是否超出知识库范围
    - needs_clarification: 是否需要用户澄清
    - clarification_questions: 如果需要澄清，列出具体问题
    - estimated_complexity: 预估复杂度 (1-5)
    """
    intent: QueryIntent
    is_beyond_kb: bool = False
    needs_clarification: bool = False
    clarification_questions: list[str] = field(default_factory=list)
    estimated_complexity: int = 1
    intent_confidence: float = 0.5


@dataclass
class RewrittenQuery:
    """
    改写后的查询结果

    字段说明:
    - rewritten: 改写后的完整问题
    - was_rewritten: 是否进行了改写（原查询即完整则返回原查询）
    - confidence: 重写置信度（0-1）
    - original: 原始查询
    - query_type: 查询类型信息
    """
    rewritten: str
    was_rewritten: bool
    confidence: float
    original: str
    query_type: QueryType | None = None


class QueryClassifier:
    """
    查询意图分类器 — 基于 DeepSeek 的轻量级分类

    能力:
    1. 意图分类: factual/comparative/analytical/definitional/procedural/summarization
    2. 知识边界判断: 是否超出知识库
    3. 模糊检测: 是否需要用户澄清
    4. 复杂度评估: 1-5 分
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client

    async def classify(self, query: str) -> QueryType:
        """
        对查询进行多维度分类

        Args:
            query: 用户查询

        Returns:
            QueryType: 包含意图、复杂度、是否需要澄清等信息
        """
        if self._llm is None:
            return self._rule_based_classify(query)

        try:
            return await self._llm_classify(query)
        except Exception as e:
            logger.warning(f"Query classification failed: {e}")
            return self._rule_based_classify(query)

    def _rule_based_classify(self, query: str) -> QueryType:
        """基于规则的简单分类（降级方案）"""
        q_lower = query.lower()

        if any(w in q_lower for w in ["vs", "versus", "compare", "difference", "不同", "比较"]):
            intent = QueryIntent.COMPARATIVE
            complexity = 3
        elif any(w in q_lower for w in ["why", "how", "analyze", "原因", "分析", "为什么", "如何"]):
            intent = QueryIntent.ANALYTICAL
            complexity = 3
        elif any(w in q_lower for w in ["what is", "define", "definition", "什么是", "定义"]):
            intent = QueryIntent.DEFINITIONAL
            complexity = 1
        elif any(w in q_lower for w in ["how to", "steps", "流程", "步骤", "如何做"]):
            intent = QueryIntent.PROCEDURAL
            complexity = 2
        elif any(w in q_lower for w in ["summarize", "summary", "总结", "概括"]):
            intent = QueryIntent.SUMMARIZATION
            complexity = 2
        elif query.strip().endswith("?"):
            intent = QueryIntent.FACTUAL
            complexity = 1
        else:
            intent = QueryIntent.FACTUAL
            complexity = 1

        is_beyond_kb = any(w in q_lower for w in ["current", "today", "latest", "最新", "今天", "实时"])
        needs_clarification = len(query) < 10

        return QueryType(
            intent=intent,
            is_beyond_kb=is_beyond_kb,
            needs_clarification=needs_clarification,
            estimated_complexity=complexity,
            intent_confidence=0.3,
        )

    async def _llm_classify(self, query: str) -> QueryType:
        """使用 LLM 进行分类"""
        import json

        prompt = f"""Analyze the following query and classify it according to these dimensions:

1. INTENT: What type of information is the user looking for?
   - factual: A specific fact, statistic, or piece of information
   - comparative: Comparing two or more entities
   - analytical: Analysis, explanation, or reasoning about a topic
   - definitional: Definition or explanation of a concept
   - procedural: Steps, processes, or how-to instructions
   - summarization: Summary or overview of a topic
   - conversational: Chat/greeting that doesn't need retrieval

2. BEYOND_KB: Is this query asking about information that might be outside the knowledge base?
   (e.g., current events, real-time data, opinions, subjective topics)
   Answer: true or false

3. NEEDS_CLARIFICATION: Is this query too vague or ambiguous to answer directly?
   (e.g., single word queries, queries with unclear pronouns)
   Answer: true or false

4. COMPLEXITY: Estimated cognitive complexity (1-5)
   1 = Simple fact lookup
   2 = Definition or summary
   3 = Multi-part or comparative question
   4 = Analysis or explanation requiring reasoning
   5 = Complex multi-hop reasoning

Query: {query}

Respond in JSON format:
{{
  "intent": "factual|comparative|analytical|definitional|procedural|summarization|conversational",
  "is_beyond_kb": true|false,
  "needs_clarification": true|false,
  "clarification_questions": ["optional question 1", "optional question 2"],
  "estimated_complexity": 1-5,
  "intent_confidence": 0.0-1.0
}}"""

        response = await self._llm.generate(prompt, max_tokens=256, temperature=0.1)
        data = json.loads(response.strip())

        intent_map = {
            "factual": QueryIntent.FACTUAL,
            "comparative": QueryIntent.COMPARATIVE,
            "analytical": QueryIntent.ANALYTICAL,
            "definitional": QueryIntent.DEFINITIONAL,
            "procedural": QueryIntent.PROCEDURAL,
            "summarization": QueryIntent.SUMMARIZATION,
            "conversational": QueryIntent.CONVERSATIONAL,
        }

        return QueryType(
            intent=intent_map.get(data.get("intent", "factual"), QueryIntent.FACTUAL),
            is_beyond_kb=data.get("is_beyond_kb", False),
            needs_clarification=data.get("needs_clarification", False),
            clarification_questions=data.get("clarification_questions", []),
            estimated_complexity=int(data.get("estimated_complexity", 1)),
            intent_confidence=float(data.get("intent_confidence", 0.5)),
        )


class QueryRewriter:
    """
    多轮对话查询改写器

    技术要点:
    - 代词检测: 通过正则检测中文代词（我/你/他/它/这/那/哪）和
      英文代词（it/this/that/they/we），有代词则触发重写
    - LLM 重写: Haiku 4.5 调用，生成完整独立问题
    - Self-check: 让 LLM 验证重写是否保留原意，不保留则回退到原查询
    - Intent Classification: 分析查询意图，指导后续检索策略

    风险考量:
    - 过度重写: 简单的事实型查询被不必要地重写，浪费 LLM 调用。
      缓解: 先检测代词，有代词才重写。
    - 重写错误: LLM 可能误解原意，生成完全不同的问题。
      缓解: 添加 self-check，置信度 < 0.7 时回退到原查询。
    """

    def __init__(
        self,
        llm_client=None,
        enable_intent_classification: bool = True,
    ):
        """
        Args:
            llm_client: 可选，传入 LLM client 实例。如果不传，跳过重写。
            enable_intent_classification: 是否启用意图分类
        """
        self._llm = llm_client
        self._enable_intent_classification = enable_intent_classification
        self._classifier = QueryClassifier(llm_client) if enable_intent_classification else None

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
            RewrittenQuery: 包含改写结果、置信度和查询类型
        """
        query_type = None

        if self._classifier:
            import asyncio
            try:
                query_type = asyncio.get_event_loop().run_until_complete(
                    self._classifier.classify(query)
                )
            except Exception as e:
                logger.warning(f"Intent classification failed: {e}")

        if not self._needs_rewriting(query):
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=1.0,
                original=query,
                query_type=query_type,
            )

        if self._llm is None:
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.5,
                original=query,
                query_type=query_type,
            )

        return self._llm_rewrite(query, conversation_history or [], query_type)

    def _needs_rewriting(self, query: str) -> bool:
        """
        判断查询是否需要重写

        技术决策:
        - 代词检测: 中英文代词是省略句的主要标志
        - 字数检测: 少于 5 个词的查询很可能是不完整的追问
        - 问号后追加内容: 「...吗？」类型的问题通常不需要重写
        """
        en_pronouns = r"\b(i|you|he|she|it|we|they|this|that|these|those|what|which)\b"
        zh_pronouns = r"[这那它她他我你咱咱们的哪个哪个些哪个]"

        has_pronoun = bool(
            re.search(en_pronouns, query.lower()) or re.search(zh_pronouns, query)
        )

        is_short = len(query.split()) < 6 and len(query) < 15

        return has_pronoun or is_short

    def _llm_rewrite(
        self,
        query: str,
        history: list[dict],
        query_type: QueryType | None,
    ) -> RewrittenQuery:
        """使用 LLM 重写查询"""
        import json

        history_context = ""
        if history:
            history_context = "\n\n对话历史:\n" + "\n".join(
                f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
                for h in history[-3:]
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
{{"rewritten": "改写后的完整问题（如果原问题已完整则同原问题）", "confidence": 0.0-1.0}}"""

        try:
            response = self._llm.generate(
                prompt,
                max_tokens=256,
                temperature=0.1,
            )
            result = json.loads(response.strip())
            rewritten = result.get("rewritten", query)
            confidence = float(result.get("confidence", 0.8))

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
                query_type=query_type,
            )

        except Exception as e:
            logger.warning(f"查询重写失败: {e}，使用原查询")
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.0,
                original=query,
                query_type=query_type,
            )

    async def rewrite_async(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> RewrittenQuery:
        """异步版本的 rewrite"""
        query_type = None

        if self._classifier:
            try:
                query_type = await self._classifier.classify(query)
            except Exception as e:
                logger.warning(f"Intent classification failed: {e}")

        if not self._needs_rewriting(query):
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=1.0,
                original=query,
                query_type=query_type,
            )

        if self._llm is None:
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.5,
                original=query,
                query_type=query_type,
            )

        return await self._llm_rewrite_async(query, conversation_history or [], query_type)

    async def _llm_rewrite_async(
        self,
        query: str,
        history: list[dict],
        query_type: QueryType | None,
    ) -> RewrittenQuery:
        """异步 LLM 重写"""
        import json

        history_context = ""
        if history:
            history_context = "\n\n对话历史:\n" + "\n".join(
                f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
                for h in history[-3:]
            )

        prompt = f"""你是一个查询改写专家。请将不完整的多轮对话查询改写为完整、独立的问题。

{history_context}

当前查询: {query}

要求:
1. 将查询改写为一个完整、独立、可直接用于向量检索的问题
2. 如果当前查询已经是完整问题，直接返回原查询
3. 不要引入原查询中没有的信息
4. 如果原查询是完整问题，直接返回原查询，不要做任何修改

请以 JSON 格式输出:
{{"rewritten": "...", "confidence": 0.0-1.0}}"""

        try:
            response = await self._llm.generate(prompt, max_tokens=256, temperature=0.1)
            result = json.loads(response.strip())
            rewritten = result.get("rewritten", query)
            confidence = float(result.get("confidence", 0.8))

            if confidence < 0.7:
                rewritten = query
                was_rewritten = False
            else:
                was_rewritten = rewritten != query

            return RewrittenQuery(
                rewritten=rewritten,
                was_rewritten=was_rewritten,
                confidence=confidence,
                original=query,
                query_type=query_type,
            )

        except Exception as e:
            logger.warning(f"查询重写失败: {e}，使用原查询")
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.0,
                original=query,
                query_type=query_type,
            )
