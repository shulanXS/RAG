"""
subquestion_decomposer.py — Sub-question Decomposition (子问题分解)
================================================================================
技术决策记录:
- Sub-question Decomposition 是 FAANG RAG 的核心能力之一，用于处理复杂多跳推理。
- 不同于简单的 keyword splitting，子问题分解需要理解查询的语义结构，
  识别各部分之间的依赖关系。
- 与 Plan-and-Execute 的关系: SubQuestion 是 Plan 的原子单元，
  SubQuestionDecomposer 是 PlanExecuteAgent 的核心组件。

核心能力:
1. 语义分解: 理解问题结构，识别实体、关系、条件
2. 依赖分析: 识别子问题之间的依赖（串行 vs 并行）
3. 类型标注: fact/comparison/analysis/list
4. 优先级排序: 根据依赖关系确定执行顺序

示例:
Input: "Apple 和 Google 在 AI 领域的投资策略有何不同？"
Output:
  - sq1: "Apple AI investments strategy 2024" (factual, priority=1, no dependencies)
  - sq2: "Google AI investments strategy 2024" (factual, priority=1, no dependencies)
  - sq3: "Compare Apple vs Google AI strategy" (comparison, priority=2, depends_on=[sq1, sq2])

技术方案:
- 使用 DeepSeek 的 deepseek-chat 生成子问题分解
- 依赖关系通过 depends_on 字段表达
- 执行顺序由 PlanExecuteAgent 根据依赖拓扑排序确定
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class SubQuestionType(Enum):
    """子问题类型"""
    FACTUAL = "factual"           # 事实型，需要精确检索
    COMPARATIVE = "comparative"   # 比较型，需要多实体对比
    ANALYSIS = "analysis"          # 分析型，需要综合推理
    LIST = "list"                 # 列表型，需要收集多个相关项
    DEFINITION = "definition"      # 定义型，需要权威定义


@dataclass
class SubQuestion:
    """
    子问题 — 分解后的最小可执行检索单元

    字段说明:
    - id: 唯一标识符 (sq1, sq2, ...)
    - question: 具体的子问题文本（用于检索）
    - question_type: 问题类型
    - keywords: 检索关键词
    - depends_on: 依赖的其他子问题 ID（拓扑排序用）
    - priority: 执行优先级 (1=最高)
    - expected_answer_type: 期望的答案类型
    - original_aspect: 原始问题中被哪个子问题覆盖
    """
    id: str
    question: str
    question_type: SubQuestionType = SubQuestionType.FACTUAL
    keywords: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    priority: int = 1
    expected_answer_type: Literal["fact", "number", "comparison", "list", "analysis"] = "fact"
    original_aspect: str = ""
    metadata: dict = field(default_factory=dict)

    def is_independent(self) -> bool:
        """是否独立（无依赖）"""
        return len(self.depends_on) == 0


@dataclass
class DecompositionResult:
    """
    分解结果

    字段说明:
    - original_query: 原始复杂查询
    - sub_questions: 分解后的子问题列表
    - execution_order: 拓扑排序后的执行顺序
    - estimated_steps: 预估执行步数
    - is_multi_hop: 是否为多跳查询
    """
    original_query: str
    sub_questions: list[SubQuestion]
    execution_order: list[SubQuestion] = field(default_factory=list)
    estimated_steps: int = 0
    is_multi_hop: bool = False

    def get_independent_questions(self) -> list[SubQuestion]:
        """获取所有独立子问题（可并行执行）"""
        return [sq for sq in self.execution_order if sq.is_independent()]

    def get_dependent_questions(self, completed: list[str]) -> list[SubQuestion]:
        """获取依赖已满足的子问题"""
        return [
            sq for sq in self.execution_order
            if sq.id not in completed
            and all(dep in completed for dep in sq.depends_on)
        ]


class SubQuestionDecomposer:
    """
    子问题分解器

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 结构分析: 识别查询中的实体、关系、条件                   │
    │  2. 语义分解: 将复杂问题拆解为可独立回答的子问题            │
    │  3. 依赖分析: 识别子问题之间的依赖关系                    │
    │  4. 拓扑排序: 根据依赖关系确定执行顺序                    │
    │  5. 类型标注: 为每个子问题标注类型和检索策略               │
    └─────────────────────────────────────────────────────────────┘

    设计模式: 策略模式
    - 不同类型的问题有不同的分解策略
    - 可通过配置切换策略
    """

    SYSTEM_PROMPT = """你是一个复杂查询分解专家。你的任务是将复杂的用户查询分解为多个最小可回答的子问题。

分解原则:
1. 每个子问题应该是一个独立的、最小可回答的检索单元
2. 子问题之间如果有依赖关系（如比较需要先分别查询），需要标注 depends_on
3. 根据问题类型确定子问题类型:
   - factual: 简单事实查询，直接检索
   - comparative: 需要多实体对比，先分别查询再比较
   - analysis: 需要综合分析，多角度检索
   - list: 需要列出多个相关项
   - definition: 需要权威定义或解释
4. 优先级: 无依赖的子问题优先（可并行），有依赖的后执行

依赖关系分析:
- 比较型查询: 先分别查询各实体，再执行比较
- 条件型查询: 先查条件，再查结果
- 多跳查询: 先查中间结果，再查最终结果

输出格式 (JSON):
{
  "analysis": "对原查询结构的简要分析",
  "sub_questions": [
    {
      "id": "sq1",
      "question": "子问题文本（用于检索，英文）",
      "question_type": "factual|comparative|analysis|list|definition",
      "keywords": ["keyword1", "keyword2", ...],
      "depends_on": [],
      "priority": 1,
      "expected_answer_type": "fact|number|comparison|list|analysis",
      "original_aspect": "该子问题对应原查询的哪个方面"
    },
    ...
  ]
}

示例输入: "Apple 和 Google 在 AI 领域的投资策略有何不同？"
示例输出:
{
  "analysis": "这是一个比较型查询，需要先分别查询 Apple 和 Google 的 AI 投资策略，再进行对比",
  "sub_questions": [
    {"id": "sq1", "question": "Apple AI investments strategy 2023 2024", "question_type": "factual",
     "keywords": ["Apple", "AI", "investment", "strategy"], "depends_on": [], "priority": 1,
     "expected_answer_type": "fact", "original_aspect": "Apple 的 AI 投资"},
    {"id": "sq2", "question": "Google AI investments strategy 2023 2024", "question_type": "factual",
     "keywords": ["Google", "AI", "investment", "strategy"], "depends_on": [], "priority": 1,
     "expected_answer_type": "fact", "original_aspect": "Google 的 AI 投资"},
    {"id": "sq3", "question": "Compare Apple vs Google AI strategy differences", "question_type": "comparative",
     "keywords": ["Apple", "Google", "AI", "strategy", "comparison"], "depends_on": ["sq1", "sq2"], "priority": 2,
     "expected_answer_type": "comparison", "original_aspect": "两者差异对比"}
  ]
}"""

    def __init__(
        self,
        llm_client=None,
        max_subquestions: int = 8,
        enable_dependency_analysis: bool = True,
    ):
        """
        Args:
            llm_client: DeepSeek LLM client，用于生成子问题分解
            max_subquestions: 最大子问题数量
            enable_dependency_analysis: 是否启用依赖分析
        """
        self._llm = llm_client
        self._max_subquestions = max_subquestions
        self._enable_dependency = enable_dependency_analysis

    async def decompose(self, query: str) -> DecompositionResult:
        """
        将复杂查询分解为子问题

        Args:
            query: 原始复杂查询

        Returns:
            DecompositionResult: 包含所有子问题及执行顺序
        """
        if not query or not query.strip():
            return DecompositionResult(original_query=query, sub_questions=[])

        if self._llm is None:
            return self._rule_based_decompose(query)

        try:
            result = await self._llm_decompose(query)
            return result
        except Exception as e:
            logger.warning(f"Sub-question decomposition failed: {e}")
            return self._rule_based_decompose(query)

    async def _llm_decompose(self, query: str) -> DecompositionResult:
        """使用 LLM 进行子问题分解"""
        import json

        prompt = f"""{self.SYSTEM_PROMPT}

User query: {query}

Generate sub-questions (max {self._max_subquestions}):"""

        response = await self._llm.generate(
            prompt,
            max_tokens=1024,
            temperature=0.3,
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
            logger.warning(f"Failed to parse decomposition JSON: {e}")
            return self._rule_based_decompose(query)

        sub_questions = self._parse_sub_questions(data.get("sub_questions", []))
        is_multi_hop = any(len(sq.depends_on) > 0 for sq in sub_questions)

        execution_order = self._topological_sort(sub_questions)

        return DecompositionResult(
            original_query=query,
            sub_questions=sub_questions,
            execution_order=execution_order,
            estimated_steps=len(sub_questions),
            is_multi_hop=is_multi_hop,
        )

    def _parse_sub_questions(self, raw_list: list[dict]) -> list[SubQuestion]:
        """解析子问题列表"""
        type_map = {
            "factual": SubQuestionType.FACTUAL,
            "comparative": SubQuestionType.COMPARATIVE,
            "analysis": SubQuestionType.ANALYSIS,
            "list": SubQuestionType.LIST,
            "definition": SubQuestionType.DEFINITION,
        }

        answer_type_map = {
            "fact": "fact",
            "number": "number",
            "comparison": "comparison",
            "list": "list",
            "analysis": "analysis",
        }

        result = []
        for item in raw_list[: self._max_subquestions]:
            try:
                qtype = type_map.get(item.get("question_type", "factual"), SubQuestionType.FACTUAL)
                atype = answer_type_map.get(item.get("expected_answer_type", "fact"), "fact")

                sq = SubQuestion(
                    id=item.get("id", f"sq{len(result)+1}"),
                    question=item.get("question", ""),
                    question_type=qtype,
                    keywords=item.get("keywords", [])[:5],
                    depends_on=item.get("depends_on", []),
                    priority=int(item.get("priority", 1)),
                    expected_answer_type=atype,
                    original_aspect=item.get("original_aspect", ""),
                )
                result.append(sq)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse sub-question item: {e}")
                continue

        return result

    def _topological_sort(self, sub_questions: list[SubQuestion]) -> list[SubQuestion]:
        """
        拓扑排序：根据依赖关系确定执行顺序

        策略:
        1. 优先执行无依赖的子问题
        2. 有依赖的子问题在所有依赖完成后执行
        3. 同一优先级的子问题可以并行执行
        """
        if not sub_questions:
            return []

        sorted_list: list[SubQuestion] = []
        completed: set[str] = set()
        remaining = {sq.id: sq for sq in sub_questions}

        max_iterations = len(sub_questions) * 2
        iteration = 0

        while remaining and iteration < max_iterations:
            iteration += 1

            ready = [
                sq for sq in remaining.values()
                if all(dep in completed for dep in sq.depends_on)
            ]

            if not ready:
                ready = list(remaining.values())

            ready.sort(key=lambda x: x.priority)

            for sq in ready[:3]:
                sorted_list.append(sq)
                completed.add(sq.id)
                remaining.pop(sq.id)

        for sq in remaining.values():
            sorted_list.append(sq)

        return sorted_list

    def _rule_based_decompose(self, query: str) -> DecompositionResult:
        """基于规则的降级分解方案"""
        sq1 = SubQuestion(
            id="sq1",
            question=query,
            question_type=SubQuestionType.FACTUAL,
            keywords=self._extract_keywords(query),
            priority=1,
            original_aspect="完整查询",
        )

        return DecompositionResult(
            original_query=query,
            sub_questions=[sq1],
            execution_order=[sq1],
            estimated_steps=1,
            is_multi_hop=False,
        )

    def _extract_keywords(self, query: str) -> list[str]:
        """简单的关键词提取"""
        import re
        words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower())
        return words[:5]

    async def execute_decomposed_plan(
        self,
        decomposition: DecompositionResult,
        retrieval_fn,
    ) -> dict:
        """
        执行分解后的子问题计划

        Args:
            decomposition: 分解结果
            retrieval_fn: 检索函数，签名为 async def(query) -> list[dict]

        Returns:
            执行结果，包含每个子问题的检索结果和最终综合答案
        """
        results: dict[str, dict] = {}
        all_chunks: list[dict] = []
        completed: set[str] = set()

        pending = list(decomposition.execution_order)

        while pending:
            ready = [
                sq for sq in pending
                if all(dep in completed for dep in sq.depends_on)
            ]

            if not ready:
                ready = [pending[0]]

            import asyncio
            tasks = [retrieval_fn(sq.question) for sq in ready]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for sq, retrieval_result in zip(ready, task_results):
                if isinstance(retrieval_result, Exception):
                    logger.warning(f"Retrieval failed for {sq.id}: {retrieval_result}")
                    retrieval_result = []

                results[sq.id] = {
                    "sub_question": sq,
                    "chunks": retrieval_result if isinstance(retrieval_result, list) else [],
                    "status": "completed",
                }

                all_chunks.extend(
                    retrieval_result if isinstance(retrieval_result, list) else []
                )
                completed.add(sq.id)

            pending = [sq for sq in pending if sq.id not in completed]

        return {
            "results": results,
            "all_chunks": all_chunks,
            "execution_order": [sq.id for sq in decomposition.execution_order],
        }
