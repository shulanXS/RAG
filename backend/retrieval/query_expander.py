"""
query_expander.py — Query Expansion (多角度查询扩展)
================================================================================
技术决策记录:
- 为什么需要 Query Expansion: 单个用户查询往往只覆盖一个角度，
  但文档中相关信息可能分布在不同表述下（不同术语、不同语言风格、
  不同粒度）。通过生成多个检索子查询，可以覆盖更多相关文档。
- DeepSeek 作为首选 LLM: 成本约为 Claude Haiku 的 1/20，
  生成 3-5 个子查询的质量足够生产使用。
- 与 HyDE 的区别: HyDE 生成假设性答案来弥合问句与陈述句的语义鸿沟；
  Query Expansion 从不同角度/表述生成检索查询，覆盖不同的文档风格。
- 适用场景: 比较型查询（"A vs B"）、多实体查询、技术术语查询。
- 不适用场景: 简单事实型查询（直接用原查询即可）。

业务难点:
- 子查询数量: 太少 → 覆盖不足；太多 → 检索成本增加 + 结果冗余。
  决策: 默认 3-5 个，通过 max_expansions 配置控制。
- 子查询之间的依赖: 比较型查询通常需要先分别查询 A 和 B，再做比较。
  决策: 通过 depends_on 字段表达依赖关系，由 PlanExecute 处理。
- 与 Query Rewrite 的关系: Rewrite 在 Expansion 之前，
  确保 Expansion 基于完整语义进行。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """
    查询意图分类 — 用于决定扩展策略

    - factual: 事实型查询，需要精确匹配
    - comparative: 比较型查询，需要多角度对比
    - analytical: 分析型查询，需要多来源综合
    - definitional: 定义型查询，需要权威定义
    - procedural: 流程型查询，需要步骤性文档
    - summarization: 摘要型查询，需要全面覆盖
    """
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    DEFINITIONAL = "definitional"
    PROCEDURAL = "procedural"
    SUMMARIZATION = "summarization"


@dataclass
class ExpandedQuery:
    """
    扩展后的单个查询

    字段说明:
    - query: 扩展后的检索查询文本
    - query_type: 查询类型 (factual/comparative/analytical/...)
    - keywords: 关键词列表，用于精确匹配
    - priority: 优先级 (1=最高，先执行)
    - depends_on: 依赖的其他子查询 ID（可选）
    - expansion_type: 扩展类型 (角度扩展/术语扩展/粒度扩展)
    """
    id: str
    query: str
    query_type: QueryIntent = QueryIntent.FACTUAL
    keywords: list[str] = field(default_factory=list)
    priority: int = 1
    depends_on: list[str] = field(default_factory=list)
    expansion_type: Literal["angle", "terminology", "granularity", "original"] = "original"
    original_text: str = ""


@dataclass
class ExpansionResult:
    """
    查询扩展结果

    字段说明:
    - original_query: 原始查询
    - expanded_queries: 扩展后的子查询列表
    - primary_intent: 主要查询意图
    - confidence: 扩展置信度 (0-1)
    - needs_hyde: 是否建议启用 HyDE
    """
    original_query: str
    expanded_queries: list[ExpandedQuery]
    primary_intent: QueryIntent
    confidence: float
    needs_hyde: bool = False


class QueryExpander:
    """
    查询扩展器 — 将单查询扩展为多角度检索子查询

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Intent Detection (DeepSeek)                              │
    │     分析查询意图，确定扩展策略                               │
    │  2. Multi-angle Expansion                                   │
    │     根据意图生成 3-5 个不同角度的子查询                       │
    │  3. Keyword Extraction                                       │
    │     提取关键词，用于精确匹配                                │
    │  4. Dependency Analysis                                      │
    │     分析子查询之间的依赖关系                                │
    │  5. Priority Assignment                                      │
    │     根据依赖关系分配执行优先级                              │
    └─────────────────────────────────────────────────────────────┘

    设计模式: 策略模式
    - 不同意图类型对应不同的扩展策略
    - 可通过配置切换扩展策略

    技术要点:
    - 使用 DeepSeek 的 deepseek-chat 生成子查询
    - 子查询包含不同角度（术语、粒度、语言风格）
    - 支持并行检索（独立子查询）或顺序执行（依赖子查询）
    - 配置化：所有策略参数均可通过 config 调整
    """

    DEFAULT_SYSTEM_PROMPT = """你是一个专业的查询扩展专家。你的任务是根据用户查询，生成多个用于信息检索的子查询。

查询扩展原则:
1. 从不同角度扩展：同义词、不同表述、不同语言风格
2. 从不同粒度扩展：整体 vs 局部、宏观 vs 微观
3. 保持语义一致：所有子查询都应围绕原查询的核心语义
4. 适合检索：子查询应能直接用于向量检索或关键词检索

输出要求:
- 生成 3-5 个子查询
- 每个子查询用英文输出（因为检索系统基于英文）
- JSON 数组格式，每个元素包含:
  - "id": 唯一标识符 (sq1, sq2, sq3...)
  - "query": 扩展后的检索查询 (英文)
  - "query_type": 类型 (factual|comparative|analytical|definitional|procedural|summarization)
  - "keywords": 关键词列表 (2-5个)
  - "priority": 优先级 (1=最高，数字越小越高)
  - "depends_on": 依赖的子查询 ID (可选，空数组表示无依赖)
  - "expansion_type": 扩展类型 (angle|terminology|granularity|original)
  - "original_text": 对应的中文子查询原文

示例输入: "Apple 和 Google 在 AI 领域的投资策略有何不同？"
示例输出:
[
  {"id": "sq1", "query": "Apple AI investments strategy 2024", "query_type": "factual",
   "keywords": ["Apple", "AI", "investment", "strategy"], "priority": 1, "depends_on": [], "expansion_type": "angle"},
  {"id": "sq2", "query": "Google AI investments strategy 2024", "query_type": "factual",
   "keywords": ["Google", "AI", "investment", "strategy"], "priority": 1, "depends_on": [], "expansion_type": "angle"},
  {"id": "sq3", "query": "Apple vs Google AI strategy comparison", "query_type": "comparative",
   "keywords": ["Apple", "Google", "AI", "comparison"], "priority": 2, "depends_on": ["sq1", "sq2"], "expansion_type": "angle"}
]"""

    def __init__(
        self,
        llm_client=None,
        max_expansions: int = 5,
        min_expansions: int = 1,
        intent_threshold: float = 0.6,
    ):
        """
        Args:
            llm_client: DeepSeek LLM client，用于生成扩展查询
            max_expansions: 最大扩展数量
            min_expansions: 最小扩展数量（至少保留原查询）
            intent_threshold: 意图检测置信度阈值
        """
        self._llm = llm_client
        self._max_expansions = max_expansions
        self._min_expansions = min_expansions
        self._intent_threshold = intent_threshold

    async def expand(self, query: str) -> ExpansionResult:
        """
        将单个查询扩展为多个检索子查询

        Args:
            query: 原始用户查询

        Returns:
            ExpansionResult: 包含所有扩展子查询及元信息
        """
        import json

        if not query or not query.strip():
            return ExpansionResult(
                original_query=query,
                expanded_queries=[],
                primary_intent=QueryIntent.FACTUAL,
                confidence=0.0,
            )

        if self._llm is None:
            return ExpansionResult(
                original_query=query,
                expanded_queries=[
                    ExpandedQuery(
                        id="original",
                        query=query,
                        query_type=QueryIntent.FACTUAL,
                        keywords=self._extract_keywords(query),
                        priority=1,
                        expansion_type="original",
                        original_text=query,
                    )
                ],
                primary_intent=QueryIntent.FACTUAL,
                confidence=0.5,
            )

        try:
            prompt = self._build_expansion_prompt(query)
            response = await self._llm.generate(
                prompt,
                max_tokens=1024,
                temperature=0.3,
            )

            parsed = self._parse_expansion_response(response)
            if not parsed:
                return self._fallback_expansion(query)

            expanded = self._build_expansion_result(query, parsed)
            logger.debug(
                f"Query expansion: '{query}' → {len(expanded.expanded_queries)} sub-queries"
            )
            return expanded

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return self._fallback_expansion(query)

    def _build_expansion_prompt(self, query: str) -> str:
        """构建扩展提示词"""
        return f"{self.DEFAULT_SYSTEM_PROMPT}\n\nUser query: {query}\n\nGenerate {self._max_expansions} sub-queries:"

    def _parse_expansion_response(self, response: str) -> list[dict]:
        """解析 LLM 返回的 JSON 响应"""
        import json

        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "queries" in data:
                return data["queries"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            try:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except Exception:
                pass

        return []

    def _build_expansion_result(
        self,
        original: str,
        parsed: list[dict],
    ) -> ExpansionResult:
        """从解析结果构建 ExpansionResult"""
        expanded_queries: list[ExpandedQuery] = []
        intent_counts: dict[str, int] = {}

        for item in parsed[: self._max_expansions]:
            try:
                intent_str = item.get("query_type", "factual")
                intent = self._parse_intent(intent_str)
                intent_counts[intent.value] = intent_counts.get(intent.value, 0) + 1

                eq = ExpandedQuery(
                    id=item.get("id", f"sq{len(expanded_queries)+1}"),
                    query=item.get("query", original),
                    query_type=intent,
                    keywords=item.get("keywords", [])[:5],
                    priority=int(item.get("priority", 1)),
                    depends_on=item.get("depends_on", []),
                    expansion_type=item.get("expansion_type", "angle"),
                    original_text=item.get("original_text", ""),
                )
                expanded_queries.append(eq)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse expanded query item: {e}")
                continue

        if not expanded_queries:
            return self._fallback_expansion(original)

        primary_intent = max(
            intent_counts.items(), key=lambda x: x[1], default=("factual", 0)
        )[0]
        primary_intent_enum = self._parse_intent(primary_intent)

        needs_hyde = primary_intent in [
            "analytical",
            "comparative",
            "summarization",
        ]

        return ExpansionResult(
            original_query=original,
            expanded_queries=expanded_queries,
            primary_intent=primary_intent_enum,
            confidence=min(0.9, 0.5 + len(expanded_queries) * 0.1),
            needs_hyde=needs_hyde,
        )

    def _fallback_expansion(self, query: str) -> ExpansionResult:
        """降级方案：使用原查询 + 关键词扩展"""
        keywords = self._extract_keywords(query)
        return ExpansionResult(
            original_query=query,
            expanded_queries=[
                ExpandedQuery(
                    id="original",
                    query=query,
                    query_type=QueryIntent.FACTUAL,
                    keywords=keywords,
                    priority=1,
                    expansion_type="original",
                    original_text=query,
                ),
                ExpandedQuery(
                    id="angle_1",
                    query=f"definition {query}",
                    query_type=QueryIntent.DEFINITIONAL,
                    keywords=keywords[:3],
                    priority=2,
                    expansion_type="angle",
                    original_text=f"定义：{query}",
                ),
            ],
            primary_intent=QueryIntent.FACTUAL,
            confidence=0.3,
        )

    @staticmethod
    def _parse_intent(intent_str: str) -> QueryIntent:
        """解析意图字符串"""
        mapping = {
            "factual": QueryIntent.FACTUAL,
            "comparative": QueryIntent.COMPARATIVE,
            "analytical": QueryIntent.ANALYTICAL,
            "definitional": QueryIntent.DEFINITIONAL,
            "procedural": QueryIntent.PROCEDURAL,
            "summarization": QueryIntent.SUMMARIZATION,
        }
        return mapping.get(intent_str.lower(), QueryIntent.FACTUAL)

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """简单的关键词提取（fallback）"""
        import re

        words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower())
        cn_chars = re.findall(r"[\u4e00-\u9fff]+", query)
        keywords = words[:5]
        if cn_chars:
            keywords.extend(cn_chars[:2])
        return keywords[:5]

    async def expand_for_retrieval(
        self,
        query: str,
        hybrid_search_fn=None,
    ) -> tuple[list[dict], ExpansionResult]:
        """
        扩展查询并执行并行检索

        Args:
            query: 原始查询
            hybrid_search_fn: 检索函数，签名为 async def(query, **kwargs) -> list[dict]

        Returns:
            (all_results, expansion_result): 所有检索结果 + 扩展元信息
        """
        expansion = await self.expand(query)

        if not hybrid_search_fn or not expansion.expanded_queries:
            return [], expansion

        independent_queries = [
            eq for eq in expansion.expanded_queries if not eq.depends_on
        ]

        import asyncio

        tasks = [hybrid_search_fn(eq.query) for eq in independent_queries]
        results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[dict] = []
        seen_ids: set[str] = set()

        for eq, results in zip(independent_queries, results_per_query):
            if isinstance(results, Exception):
                logger.warning(f"Retrieval failed for {eq.id}: {results}")
                continue

            for r in results:
                chunk_id = r.get("chunk_id", "")
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    r["_expansion_id"] = eq.id
                    r["_expansion_type"] = eq.expansion_type
                    r["_priority"] = eq.priority
                    all_results.append(r)

        all_results.sort(key=lambda x: (x.get("_priority", 99), -x.get("rerank_score", x.get("rrf_score", 0))))

        logger.info(
            f"Query expansion retrieval: '{query}' → "
            f"{len(independent_queries)} parallel queries → {len(all_results)} unique chunks"
        )

        return all_results, expansion
