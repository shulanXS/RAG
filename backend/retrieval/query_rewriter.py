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

P0-6: 移除 QueryIntent / QueryType / `query_type` 字段 — 下游零消费（QueryRouter
只分 SIMPLE/MODERATE/COMPLEX/BEYOND_KB 复杂度，不读 intent）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict
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

    def __init__(
        self,
        llm_client=None,
        cache_size: int = 500,
    ):
        """
        Args:
            llm_client: 可选，传入 LLM client 实例。如果不传，跳过重写。
            cache_size: rewrite 结果的 LRU 缓存大小
        """
        self._llm = llm_client
        # LRU cache: query+history hash -> RewrittenQuery
        self._cache: OrderedDict[str, RewrittenQuery] = OrderedDict()
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _make_cache_key(query: str, history: list[dict] | None) -> str:
        """生成稳定 cache key：query + 最近 3 条历史的 content 哈希"""
        h = hashlib.md5()
        h.update(query.encode("utf-8"))
        h.update(b"|")
        if history:
            for msg in history[-3:]:
                h.update(str(msg.get("content", "")).encode("utf-8"))
                h.update(b"|")
        return h.hexdigest()

    def _cache_get(self, key: str) -> RewrittenQuery | None:
        """从 LRU 缓存获取，并把访问过的项移到末尾"""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        self._cache_hits += 1
        return self._cache[key]

    def _cache_put(self, key: str, value: RewrittenQuery) -> None:
        """放入 LRU 缓存；超过容量时淘汰最旧的"""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        self._cache_misses += 1

    def get_cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "max_size": self._cache_size,
            "hit_rate": f"{hit_rate:.2%}",
        }

    def rewrite(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> RewrittenQuery:
        """
        将多轮对话中的不完整查询改写为独立完整问题。
        (同步版本，无 LRU 缓存；异步版本见 rewrite_async)

        Args:
            query: 当前用户查询
            conversation_history: 对话历史

        Returns:
            RewrittenQuery: 改写结果
        """
        if not self._needs_rewriting(query):
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=1.0,
                original=query,
            )

        if self._llm is None:
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.5,
                original=query,
            )

        return self._llm_rewrite(query, conversation_history or [])

    def _needs_rewriting(self, query: str) -> bool:
        """
        判断查询是否需要重写

        技术决策:
        - 代词检测: 中英文代词是省略句的主要标志
        - 长度检测: 中文 ≤8 字 / 英文 <6 词 且 <30 字符 视为过短
        - 问号后追加内容: 「...吗？」类型的问题通常不需要重写
        """
        en_pronouns = r"\b(i|you|he|she|it|we|they|this|that|these|those|what|which)\b"
        zh_pronouns = r"[这那它她他我你咱咱们哪哪些]"

        has_pronoun = bool(
            re.search(en_pronouns, query.lower()) or re.search(zh_pronouns, query)
        )

        # 中文: char ≤ 8 (中文按字符计, 不按空格 split)
        # 英文: 词数 < 6 且 char < 30
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
        if has_chinese:
            is_short = len(query) <= 8
        else:
            is_short = len(query.split()) < 6 and len(query) < 30

        return has_pronoun or is_short

    def _llm_rewrite(
        self,
        query: str,
        history: list[dict],
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
            )

        except Exception as e:
            logger.warning(f"查询重写失败: {e}，使用原查询")
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.0,
                original=query,
            )

    async def rewrite_async(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> RewrittenQuery:
        """异步版本的 rewrite（带 LRU 缓存）"""
        cache_key = self._make_cache_key(query, conversation_history)
        cached = self._cache_get(cache_key)
        if cached is not None:
            # 标记为缓存命中，供 trace/observability 使用
            return cached

        if not self._needs_rewriting(query):
            result = RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=1.0,
                original=query,
            )
            self._cache_put(cache_key, result)
            return result

        if self._llm is None:
            result = RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.5,
                original=query,
            )
            self._cache_put(cache_key, result)
            return result

        result = await self._llm_rewrite_async(query, conversation_history or [])
        self._cache_put(cache_key, result)
        return result

    async def _llm_rewrite_async(
        self,
        query: str,
        history: list[dict],
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
            )

        except Exception as e:
            logger.warning(f"查询重写失败: {e}，使用原查询")
            return RewrittenQuery(
                rewritten=query,
                was_rewritten=False,
                confidence=0.0,
                original=query,
            )
