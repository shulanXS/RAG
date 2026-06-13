"""
react_agent.py — ReAct Agent (LangGraph 实现)
================================================================================
技术决策记录:
- ReAct (Reasoning + Acting) 循环: LLM 在每一步推理中决定「是检索还是生成」。
  这是 2026 年 Agentic RAG 的主流方案，比 Plan-and-Execute 更轻量。
- LangGraph: 微软开源的生产级 Agent 框架。相比 LangChain Agents，
  LangGraph 提供显式状态机、确定性的条件分支、人类介入中断点。
- max_iterations=5: 防止无限循环的标准工程实践。
  5 步足以处理大多数中等复杂度查询（每步检索-推理-决策）。

业务难点:
- 检索循环: LLM 可能反复检索相似内容而不推进推理。
  缓解: early_stop_threshold=0.85，达到置信度阈值时提前退出。
- 工具调用过度: 每次迭代只允许一次检索调用，避免重复检索。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

logger = logging.getLogger(__name__)

# =============================================================================
# 1. 状态定义 (TypedDict)
# =============================================================================


class ReActState(TypedDict, total=False):
    """
    ReAct Agent 状态机

    字段说明:
    - query: 用户原始查询
    - rewritten_query: 改写后的查询
    - iterations: 当前迭代次数
    - retrieved_chunks: 本次迭代检索到的 chunks
    - reasoning: LLM 当前推理步骤
    - action: LLM 当前决定采取的行动
    - next_query: 用于下一步检索的查询（从 LLM reasoning 中提取）
    - memory_bank_summary: Memory Bank 摘要
    - final_answer: 最终答案（当 action == "FINISH" 时）
    - confidence: 当前置信度
    - error: 错误信息
    - trace: 推理轨迹（用于展示和 debug）
    """
    query: str
    rewritten_query: str
    iterations: int
    retrieved_chunks: list
    reasoning: str
    action: Literal["retrieve", "think", "finish", "error"]
    next_query: str
    memory_bank_summary: str
    final_answer: str | None
    confidence: float
    error: str | None
    trace: list[dict]


# =============================================================================
# 2. ReAct Prompt Templates
# =============================================================================

REACT_SYSTEM_PROMPT = """你是一个企业知识助手的推理引擎。

你的任务是通过迭代推理和检索，回答用户问题。

可用工具:
- retrieve(query): 检索相关文档片段
  输入: 检索查询字符串
  输出: 相关文档片段列表

推理模式:
1. think: 分析当前状态，决定下一步行动
2. retrieve: 调用检索工具，获取相关上下文
3. finish: 当置信度足够高时，生成最终答案

输出格式 (JSON):
{
  "reasoning": "你的推理过程",
  "action": "think|retrieve|finish",
  "next_query": "如果 action=retrieve，输入检索查询",
  "confidence": 0.0-1.0,
  "answer": "如果 action=finish，输入最终答案"
}

约束:
- 最多执行 {max_iterations} 次迭代
- 每次迭代最多检索一次
- 如果置信度 >= {early_stop_threshold}，立即生成答案
- 如果迭代次数耗尽仍未达到置信度阈值，给出当前最佳答案并说明局限性
"""

REACT_USER_PROMPT = """用户问题: {query}

当前迭代: {iterations}/{max_iterations}
当前置信度: {confidence}

已检索到的上下文:
{context}

请决定下一步行动。"""


# =============================================================================
# 3. ReAct Agent 实现
# =============================================================================

class ReActAgent:
    """
    ReAct Agent — LangGraph 状态机实现

    流程图:
    ┌─────────────────────────────────────────┐
    │           START (iterations=0)           │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   think_node    │ ← LLM 推理下一步行动
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    action=          action=      action=
    "retrieve"     "think"      "finish"
         │             │             │
         ▼             │             ▼
    ┌────────┐        │     ┌────────────┐
    │retrieve │        │     │ finish_node│
    │  _node  │        │     │  (生成答案) │
    └────┬────┘        │     └────────────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
              ┌────────▼────────┐
              │  iterations ≥   │──No──┐
              │ max_iterations?  │      │
              └────────┬────────┘      │
                       │ Yes           │
                       ▼               │
              ┌─────────────────┐      │
              │   max_iter     │      │
              │   (强制 finish) │      │
              └────────┬────────┘      │
                       │               │
                       └───────────────┘

    技术要点:
    - LangGraph StateGraph: 显式状态机，每步有明确的输入输出
    - Conditional edges: 根据 LLM 输出动态决定下一步走哪个分支
    - 最大迭代次数限制: 防止无限循环
    - 置信度提前退出: confidence >= early_stop_threshold 时跳过推理直接生成

    风险考量:
    - LLM 推理质量不稳定: 提示词工程是 ReAct Agent 的关键
    - 检索结果质量差: 依赖下游 HybridSearchEngine 的质量
    """

    def __init__(
        self,
        llm_client=None,
        retrieval_fn=None,
        max_iterations: int = 5,
        early_stop_threshold: float = 0.85,
    ):
        """
        Args:
            llm_client: LLM client（用于推理决策）
            retrieval_fn: 检索函数，签名为 async def(query) -> list[chunks]
            max_iterations: 最大迭代次数
            early_stop_threshold: 置信度阈值，达到此值则提前退出
        """
        self._llm = llm_client
        self._retrieve_fn = retrieval_fn
        self._max_iters = max_iterations
        self._early_stop = early_stop_threshold

        self._current_chunks: list[dict] = []
        self._trace: list[dict] = []
        self._compiled_graph = self._build_graph()

    def _build_graph(self):
        """
        构建 LangGraph 状态机

        技术要点:
        - StateGraph 定义状态结构和节点
        - add_node 添加处理节点
        - add_edge 添加固定边
        - add_conditional_edges 添加条件边（根据 state 决定下一步）
        """
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            raise ImportError("需要安装 langgraph: pip install langgraph")

        # 定义图
        graph = StateGraph(ReActState)

        # 添加节点
        graph.add_node("think", self._think_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("finish", self._finish_node)
        graph.add_node("max_iter", self._max_iter_node)

        # 设置入口点
        graph.set_entry_point("think")

        # 条件边: think → ?
        def route_action(state: ReActState) -> Literal["retrieve", "finish", "max_iter"]:
            action = state.get("action", "think")
            iters = state.get("iterations", 0)

            if action == "finish" or state.get("confidence", 0) >= self._early_stop:
                return "finish"
            elif action == "retrieve" and iters < self._max_iters:
                return "retrieve"
            elif iters >= self._max_iters:
                return "max_iter"
            else:
                return "think"

        graph.add_conditional_edges(
            "think",
            route_action,
            {
                "retrieve": "retrieve",
                "finish": "finish",
                "max_iter": "max_iter",
                "think": "think",  # 如果 action=think，继续 think
            },
        )

        # 固定边
        graph.add_edge("retrieve", "think")
        graph.add_edge("max_iter", "finish")
        graph.add_edge("finish", END)

        return graph.compile()

    async def run(self, query: str, rewritten_query: str = "") -> tuple[str, float, list[dict]]:
        """
        执行 ReAct Agent

        Args:
            query: 用户原始查询
            rewritten_query: 改写后的查询

        Returns:
            (final_answer, confidence, retrieved_chunks)
        """
        self._current_chunks = []
        self._trace = []

        if rewritten_query:
            display_query = rewritten_query
        else:
            display_query = query

        initial_state: ReActState = {
            "query": query,
            "rewritten_query": rewritten_query or query,
            "iterations": 0,
            "retrieved_chunks": [],
            "reasoning": "",
            "action": "think",
            "memory_bank_summary": "",
            "final_answer": None,
            "confidence": 0.0,
            "error": None,
            "trace": [],
        }

        try:
            result = await self._compiled_graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"ReAct 执行异常: {e}")
            return f"Agent 执行失败: {e}", 0.0, []

        final_answer = result.get("final_answer", "无法生成答案")
        confidence = result.get("confidence", 0.0)
        all_chunks = self._current_chunks

        return final_answer, confidence, all_chunks

    async def _think_node(self, state: ReActState) -> dict:
        """
        推理节点: LLM 分析当前状态，决定下一步行动
        """
        if self._llm is None:
            # 无 LLM 时，直接 finish
            return {"action": "finish", "confidence": 0.0}

        iters = state.get("iterations", 0)
        context = self._format_context(state.get("retrieved_chunks", []))

        user_prompt = REACT_USER_PROMPT.format(
            query=state.get("rewritten_query", state.get("query", "")),
            iterations=iters,
            max_iterations=self._max_iters,
            confidence=state.get("confidence", 0.0),
            context=context or "(暂无上下文，请检索)",
        )

        prompt = f"{REACT_SYSTEM_PROMPT.format(max_iterations=self._max_iters, early_stop_threshold=self._early_stop)}\n\n{user_prompt}"

        try:
            import json
            response = await self._llm.generate_async(prompt, max_tokens=512, temperature=0.1)
            parsed = json.loads(response.strip())

            action = parsed.get("action", "think")
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            next_query = parsed.get("next_query", state.get("rewritten_query", state.get("query", "")))

            # 记录推理轨迹
            self._trace.append({
                "iteration": iters,
                "action": action,
                "reasoning": reasoning,
                "confidence": confidence,
            })

            return {
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "next_query": next_query,
                "iterations": iters + 1,
                "trace": self._trace,
            }

        except Exception as e:
            logger.warning(f"推理节点异常: {e}")
            return {"action": "finish", "confidence": 0.0, "error": str(e)}

    async def _retrieve_node(self, state: ReActState) -> dict:
        """
        检索节点: 根据 LLM 决策执行检索
        """
        if self._retrieve_fn is None:
            return {"retrieved_chunks": state.get("retrieved_chunks", [])}

        iters = state.get("iterations", 0)
        reasoning = state.get("reasoning", "")

        # 使用 LLM 生成的 next_query 执行检索（而非固定用 rewritten_query）
        query = state.get("next_query", state.get("rewritten_query", state.get("query", "")))

        try:
            chunks = await self._retrieve_fn(query)
            self._current_chunks.extend(chunks)

            return {
                "retrieved_chunks": self._current_chunks,
                "memory_bank_summary": self._format_context(chunks),
            }

        except Exception as e:
            logger.warning(f"检索节点异常: {e}")
            return {"error": str(e)}

    async def _finish_node(self, state: ReActState) -> dict:
        """生成节点: 基于收集的上下文生成最终答案"""
        context = self._format_context(state.get("retrieved_chunks", []))
        query = state.get("rewritten_query", state.get("query", ""))

        if self._llm is None:
            answer = f"基于检索结果，无法给出精确答案。请提供更多信息。"
            return {"final_answer": answer}

        prompt = f"""你是一个企业知识助手。请基于以下检索到的上下文信息回答用户问题。

上下文:
{context}

用户问题: {query}

要求:
1. 直接回答问题，不要重复问题
2. 每个关键陈述必须标注来源，格式: [来源]
3. 如果上下文信息不足以回答，明确说明
4. 保持专业、简洁
"""

        try:
            answer = await self._llm.generate_async(prompt, max_tokens=1024, temperature=0.3)
            return {"final_answer": answer}
        except Exception as e:
            logger.warning(f"生成节点异常: {e}")
            return {"final_answer": "答案生成失败", "error": str(e)}

    async def _max_iter_node(self, state: ReActState) -> dict:
        """
        最大迭代节点: 达到最大迭代次数时强制生成答案
        """
        logger.warning(f"ReAct Agent 达到最大迭代次数 ({self._max_iters})，强制生成答案")
        return {"final_answer": "达到最大推理步骤，请基于已有上下文给出最佳答案"}

    def _format_context(self, chunks: list[dict]) -> str:
        """将检索结果格式化为上下文文本"""
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks[:10], 1):  # 最多 10 个 chunk
            lines.append(f"[{i}] {chunk.get('text', '')[:300]}...")
        return "\n".join(lines)
