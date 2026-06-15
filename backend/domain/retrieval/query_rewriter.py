"""
query_rewriter.py — 多轮对话查询改写器(2026 async-only)
================================================================================
P3.1 重构(plan §3.1):
- 删同步版 rewrite() / _llm_rewrite(),所有调用方走 async 路径
- async-only 是 2026 Python 后端标准,2025 后 asyncio + uvicorn 已是默认

设计不变:
- 代词检测 + LRU 缓存 500 容量
- LLM 改写 + confidence<0.7 回退原 query

业务不变:
- 多轮对话省略句("那第二点呢?")改写为完整独立问题
- 代词/短查询触发 LLM 调用,完整查询直接返回
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass

from backend.observability.metrics import create_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    """改写结果 — 唯一 4 字段,无冗余字段(plan §3.3)

    字段:
    - rewritten: 改写后的完整问题
    - was_rewritten: 是否进行了改写
    - confidence: 重写置信度(0-1)
    - original: 原始查询
    """
    rewritten: str
    was_rewritten: bool
    confidence: float
    original: str


class QueryRewriter:
    """多轮对话查询改写器(2026 async-only 唯一入口)"""

    def __init__(self, llm_client=None, cache_size: int = 500):
        self._llm = llm_client
        self._cache: OrderedDict[str, RewrittenQuery] = OrderedDict()
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _make_cache_key(query: str, history: list[dict] | None) -> str:
        h = hashlib.md5()
        h.update(query.encode("utf-8"))
        h.update(b"|")
        if history:
            for msg in history[-3:]:
                h.update(str(msg.get("content", "")).encode("utf-8"))
                h.update(b"|")
        return h.hexdigest()

    def _cache_get(self, key: str) -> RewrittenQuery | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        self._cache_hits += 1
        return self._cache[key]

    def _cache_put(self, key: str, value: RewrittenQuery) -> None:
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

    def _needs_rewriting(self, query: str) -> bool:
        """代词检测 + 长度检测 — 同步但纯 CPU,无需 await"""
        en_pronouns = r"\b(i|you|he|she|it|we|they|this|that|these|those|what|which)\b"
        zh_pronouns = r"[这那它她他我你咱咱们哪哪些]"

        has_pronoun = bool(
            re.search(en_pronouns, query.lower()) or re.search(zh_pronouns, query)
        )

        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
        if has_chinese:
            is_short = len(query) <= 8
        else:
            is_short = len(query.split()) < 6 and len(query) < 30

        return has_pronoun or is_short

    async def rewrite_async(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> RewrittenQuery:
        """异步唯一入口

        流程:
        1. LRU cache lookup — 命中直接返回
        2. 代词检测:无代词 + 长度OK → 不调 LLM,confidence=1.0
        3. 无 LLM client → confidence=0.5,原 query 返回
        4. LLM 重写 + JSON 解析;confidence<0.7 回退
        """
        cache_key = self._make_cache_key(query, conversation_history)
        cached = self._cache_get(cache_key)
        if cached is not None:
            # P3.1(plan §4.4): 记录改写缓存命中
            create_metrics_collector().record_query_rewrite_cache(hit=True)
            return cached
        # cache miss — 走后续路径
        create_metrics_collector().record_query_rewrite_cache(hit=False)

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
        """使用 LLM 重写查询(OpenAI 兼容协议)"""
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
