"""
query_router.py — 查询复杂度路由器
================================================================================
技术决策记录:
- 为什么要路由: 简单事实型查询不需要复杂的 Agentic 推理，直接混合检索就够。
  用复杂方案解决简单问题是过度工程化，会增加延迟和成本。
- 三层复杂度定义:
  (1) Simple: 单实体、无逻辑推理 → 直接混合检索（~50ms）
  (2) Moderate: 多实体、需一定推理 → ReAct 循环（~300-500ms）
  (3) Complex: 多跳关系、跨文档综合 → Plan-and-Execute（~1-2s）
- Haiku 4.5 路由: 速度极快（~100ms），成本仅为 Sonnet 的 1/20，
  路由决策的准确度对质量要求不高（主要是避免用简单方案处理复杂问题）。

业务难点:
- 路由准确性: LLM 路由不可能 100% 准确，误判会导致质量问题。
  缓解: 置信度阈值（threshold=0.6），低于阈值则升级到更高复杂度。
- 升级路径: Simple 可升级到 Moderate，Moderate 可升级到 Complex。
  不允许降级（Complex → Moderate → Simple），因为降级可能导致信息遗漏。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.agentic.query_signals import QuerySignals

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    """
    查询复杂度枚举

    技术决策:
    - 使用 str, Enum 而非普通 Enum：支持 JSON 序列化，
      直接作为 LangGraph state 的值传递。
    """
    SIMPLE = "simple"        # 直接混合检索
    MODERATE = "moderate"    # ReAct Agent
    COMPLEX = "complex"      # Plan-and-Execute
    BEYOND_KB = "beyond_kb"  # 模型可直接回答（无需检索）


@dataclass
class RoutingDecision:
    """
    路由决策结果

    字段说明:
    - complexity: 查询复杂度分类
    - confidence: 分类置信度（0-1）
    - reasoning: LLM 的分类理由（用于 debug 和审计）
    - recommended_approach: 推荐的检索策略描述
    - signals: P2-B6 提取的纯规则信号 (无 LLM, < 1ms)
              下游 (B7 动态 RRF, OTel span) 都靠这个字段
    """
    complexity: QueryComplexity
    confidence: float
    reasoning: str
    recommended_approach: str
    original_query: str
    signals: "QuerySignals | None" = None


class QueryRouter:
    """
    查询复杂度路由器

    设计思路:
    - 输入: 用户查询（可能包含对话历史）
    - 输出: 复杂度分类 + 置信度
    - 内部: Haiku 4.5 LLM 调用

    技术要点:
    - 提示词设计: 给出 3 类查询的具体示例，帮助 LLM 做出准确分类
    - 输出格式: JSON Schema 约束，减少解析失败
    - 降级策略: 置信度低于 threshold 时，升级到更高复杂度
    """

    SYSTEM_PROMPT = """你是一个查询复杂度分类器。你的任务是将用户查询分类为以下四类之一：

1. **simple**: 简单事实型查询。
   - 特征: 单实体、无逻辑推理、无需跨文档
   - 示例: 「合同编号 A-2024-001 的甲方是谁？」、「X产品的价格是多少？」
   - 策略: 直接混合检索即可，无需复杂推理

2. **moderate**: 中等复杂度查询。
   - 特征: 多实体、需一定推理、无需多跳关系
   - 示例: 「X供应商的交付能力与Y供应商相比如何？」、「A政策和B政策有何异同？」
   - 策略: ReAct 推理循环，边推理边检索

3. **complex**: 复杂多跳查询。
   - 特征: 需要跨多个文档、多跳关系推理、或综合分析
   - 示例: 「如果供应商X断供，对哪些客户的产品交付有影响？」、「本季度跨部门风险主题有哪些？」
   - 策略: Plan-and-Execute，先规划再逐步执行

4. **beyond_kb**: 模型可直接回答，无需检索。
   - 特征: 通用知识、常识问题、明显无需检索
   - 示例: 「什么是RAG？」、「如何泡咖啡？」
   - 策略: 直接 LLM 生成，跳过检索（节省成本和延迟）

请根据以上定义，对查询进行分类。"""

    def __init__(
        self,
        llm_client=None,
        complexity_threshold: float = 0.6,
    ):
        """
        Args:
            llm_client: LLM client 实例（用于分类）
            complexity_threshold: 置信度低于此值则升级复杂度
        """
        self._llm = llm_client
        self._threshold = complexity_threshold

    def route(self, query: str, history: list[dict] | None = None) -> RoutingDecision:
        """
        对查询进行复杂度分类

        Args:
            query: 用户查询
            history: 对话历史

        Returns:
            RoutingDecision: 包含复杂度分类和置信度 + QuerySignals
        """
        # P2-B6: 无论 LLM 是否可用, 都先跑一遍规则 analyzer
        # (无 LLM 时用 signals.complexity_hint() 兜底)
        from backend.agentic.query_signals import QueryAnalyzer
        signals = QueryAnalyzer().analyze(query)

        if self._llm is None:
            # 无 LLM 时, 用 signals 给的 hint 兜底 (避免无脑 MODERATE)
            hint = signals.complexity_hint()
            try:
                complexity = QueryComplexity(hint)
            except ValueError:
                complexity = QueryComplexity.MODERATE
            return RoutingDecision(
                complexity=complexity,
                confidence=0.5,
                reasoning=f"No LLM client; signals-based hint: {hint}",
                recommended_approach=(
                    "直接混合检索" if complexity == QueryComplexity.SIMPLE
                    else "ReAct Agent" if complexity == QueryComplexity.MODERATE
                    else "Plan-and-Execute"
                ),
                original_query=query,
                signals=signals,
            )

        history_context = ""
        if history:
            history_context = "\n\n对话历史:\n" + "\n".join(
                f"用户: {h['content']}" if h.get("role") == "user" else f"助手: {h['content']}"
                for h in history[-2:]
            )

        # P2-B6: 把 signals 拼进 LLM prompt, 让 LLM 决策时能参考规则信号
        # (pronoun / multi-hop / length 等结构化 hint 比纯文本稳)
        signals_context = (
            f"\n\n规则信号 (供参考):\n"
            f"- 包含代词: {signals.has_pronoun}\n"
            f"- 实体数: {signals.entity_count}\n"
            f"- 多跳关键词: {signals.is_multi_hop}\n"
            f"- 查询长度: {signals.query_length}\n"
            f"- 包含引号: {signals.has_quote}"
        )

        prompt = f"""{self.SYSTEM_PROMPT}

{history_context}
{signals_context}

请分类以下查询:

查询: {query}

请以 JSON 格式输出:
{{"complexity": "simple|moderate|complex|beyond_kb", "confidence": 0.0-1.0, "reasoning": "分类理由（1-2句话）"}}
"""

        try:
            import json
            response = self._llm.generate(
                prompt,
                max_tokens=256,
                temperature=0.1,
            )
            result = json.loads(response.strip())

            complexity_str = result.get("complexity", "moderate")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")

            # 映射到枚举
            try:
                complexity = QueryComplexity(complexity_str)
            except ValueError:
                complexity = QueryComplexity.MODERATE

            # 降级路径: 置信度低于阈值时升级复杂度
            if confidence < self._threshold and complexity == QueryComplexity.SIMPLE:
                complexity = QueryComplexity.MODERATE
                reasoning += " (置信度过低，升级到 MODERATE)"
            elif confidence < self._threshold and complexity == QueryComplexity.MODERATE:
                complexity = QueryComplexity.COMPLEX
                reasoning += " (置信度过低，升级到 COMPLEX)"

            approach_map = {
                QueryComplexity.SIMPLE: "直接混合检索（BM25 + Dense + RRF + Reranker）",
                QueryComplexity.MODERATE: "ReAct 推理循环（边推理边检索）",
                QueryComplexity.COMPLEX: "Plan-and-Execute（先规划再逐步执行）",
                QueryComplexity.BEYOND_KB: "直接 LLM 生成（无需检索）",
            }

            return RoutingDecision(
                complexity=complexity,
                confidence=confidence,
                reasoning=reasoning,
                recommended_approach=approach_map[complexity],
                original_query=query,
                signals=signals,
            )

        except Exception as e:
            logger.warning(f"路由失败: {e}，默认 MODERATE")
            # P2-B6: 即使 LLM 异常也保留 signals (debug 价值)
            return RoutingDecision(
                complexity=QueryComplexity.MODERATE,
                confidence=0.0,
                reasoning=f"路由异常: {e}",
                recommended_approach="ReAct Agent",
                original_query=query,
                signals=signals,
            )
