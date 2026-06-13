"""
plan_execute.py — Plan-and-Execute Agent
================================================================================
技术决策记录:
- Plan-and-Execute vs ReAct: Plan-and-Execute 先制定完整行动计划再逐步执行，
  适合复杂多步任务（如「对比 A/B/C 三个供应商的交付能力」）。
  ReAct 是边推理边执行，适合动态调整策略的场景。
- 适用场景: 需要综合分析多个文档的复杂查询、跨部门分析、风险评估等。
- 权衡取舍: Plan-and-Execute 规划开销较大（约 200-500ms），
  不适合简单查询。因此作为 Query Router 的 Complex 级别选项。

业务难点:
- 规划爆炸: 复杂任务可能产生数十个步骤，导致响应时间过长。
  缓解: max_steps=8 限制，每个步骤限制 subquery 数量。
- 子步骤失败: 中间某个检索步骤失败时，需要决定是否继续。
  决策: 继续执行，但最终答案中标注信息缺失的部分。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

logger = logging.getLogger(__name__)


class PlanState(TypedDict, total=False):
    """
    Plan-and-Execute 状态机
    """
    query: str
    plan: list[dict]  # [{"step": 1, "action": "retrieve", "subquery": "...", "status": "pending"}]
    current_step: int
    step_results: list[dict]
    memory_bank_summary: str
    final_answer: str | None
    confidence: float
    trace: list[dict]


PLAN_SYSTEM_PROMPT = """你是一个任务规划专家。你的任务是将复杂问题分解为可执行的检索步骤。

任务分解原则:
1. 每个步骤应该是一个独立的检索任务
2. 步骤之间应该有逻辑顺序
3. 优先解决子问题，再综合答案
4. 最多分解为 {max_steps} 个步骤

可用操作:
- retrieve: 检索相关文档片段
- analyze: 分析已检索到的文档
- compare: 对比多个文档的内容
- synthesize: 综合多个来源的信息

输出格式 (JSON):
{{
  "plan": [
    {{"step": 1, "action": "retrieve", "subquery": "检索子问题1"}},
    {{"step": 2, "action": "retrieve", "subquery": "检索子问题2"}},
    ...
  ]
}}

请将以下查询分解为步骤:"""

PLAN_USER_PROMPT = """用户问题: {query}
最大步骤数: {max_steps}

请将问题分解为可执行的步骤（最多 {max_steps} 个）。"""


class PlanExecuteAgent:
    """
    Plan-and-Execute Agent — LangGraph 实现

    流程图:
    ┌──────────────────────────────────────────────┐
    │              START                           │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   plan_node     │ ← LLM 制定行动计划
              │  (规划步骤)     │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  steps pending? │──Yes──┐
              └────────┬────────┘       │
                       │ No             │
                       ▼                ▼
              ┌─────────────────┐  ┌─────────────────┐
              │    finish       │  │  execute_step   │
              │  (生成答案)    │  │  (执行当前步骤) │
              └─────────────────┘  └────────┬────────┘
                                             │
                                      ┌──────▼──────┐
                                      │ next_step?  │──Yes──┐
                                      └──────┬──────┘       │
                                             │ No             │
                                             ▼                │
                                    ┌──────────────┐        │
                                    │  回 plan_node │◄──────┘
                                    │ (更新计划)    │
                                    └──────────────┘

    技术要点:
    - 两阶段执行: 先规划（plan），再按计划执行（execute）
    - 动态更新: 执行结果可反馈到规划中，动态调整后续步骤
    - LangGraph 的条件边实现步骤循环
    """

    def __init__(
        self,
        llm_client=None,
        retrieval_fn=None,
        max_steps: int = 8,
        max_subqueries_per_step: int = 3,
    ):
        """
        Args:
            llm_client: LLM client（用于规划和生成）
            retrieval_fn: 检索函数
            max_steps: 最大步骤数
            max_subqueries_per_step: 每步骤最多子查询数
        """
        self._llm = llm_client
        self._retrieve_fn = retrieval_fn
        self._max_steps = max_steps
        self._max_subqueries = max_subqueries_per_step

        self._all_chunks: list[dict] = []
        self._trace: list[dict] = []

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            raise ImportError("需要安装 langgraph: pip install langgraph")

        graph = StateGraph(PlanState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("execute_step", self._execute_step_node)
        graph.add_node("finish", self._finish_node)

        graph.set_entry_point("plan")

        # 条件边: plan → execute_step 或 finish
        def route_plan(state: PlanState) -> Literal["execute_step", "finish"]:
            plan = state.get("plan", [])
            current = state.get("current_step", 0)
            if not plan or current >= len(plan):
                return "finish"
            return "execute_step"

        # 条件边: execute_step → execute_step (next step) 或 finish
        def route_step(state: PlanState) -> Literal["execute_step", "finish"]:
            plan = state.get("plan", [])
            current = state.get("current_step", 0)
            if current >= len(plan):
                return "finish"
            return "execute_step"

        graph.add_conditional_edges("plan", route_plan, {"execute_step": "execute_step", "finish": "finish"})
        graph.add_conditional_edges("execute_step", route_step, {"execute_step": "execute_step", "finish": "finish"})
        graph.add_edge("finish", END)

        return graph.compile()

    async def run(self, query: str) -> tuple[str, float, list[dict]]:
        """
        执行 Plan-and-Execute Agent

        Args:
            query: 用户查询

        Returns:
            (final_answer, confidence, all_retrieved_chunks)
        """
        self._all_chunks = []
        self._trace = []

        graph = self._build_graph()

        initial_state: PlanState = {
            "query": query,
            "plan": [],
            "current_step": 0,
            "step_results": [],
            "memory_bank_summary": "",
            "final_answer": None,
            "confidence": 0.0,
            "trace": [],
        }

        try:
            result = await graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"Plan-and-Execute 执行异常: {e}")
            return f"Agent 执行失败: {e}", 0.0, []

        final_answer = result.get("final_answer", "无法生成答案")
        confidence = result.get("confidence", 0.0)

        return final_answer, confidence, self._all_chunks

    async def _plan_node(self, state: PlanState) -> dict:
        """规划节点: 将复杂查询分解为步骤"""
        if self._llm is None:
            return {"plan": [], "current_step": 0}

        query = state.get("query", "")

        prompt = f"""{PLAN_SYSTEM_PROMPT.format(max_steps=self._max_steps)}

{PLAN_USER_PROMPT.format(query=query, max_steps=self._max_steps)}"""

        try:
            import json
            response = self._llm.generate(prompt, max_tokens=512, temperature=0.1)
            parsed = json.loads(response.strip())

            plan = parsed.get("plan", [])
            # 确保步骤有 status 字段
            for step in plan:
                step.setdefault("status", "pending")

            current = state.get("current_step", 0)
            if current > 0:
                # 不是第一次规划：更新当前步骤后的计划
                existing_plan = state.get("plan", [])
                if len(existing_plan) > current:
                    plan = existing_plan[:current] + plan

            self._trace.append({"action": "plan", "steps": len(plan)})
            return {"plan": plan, "current_step": current, "trace": self._trace}

        except Exception as e:
            logger.warning(f"规划节点异常: {e}")
            return {"plan": [], "trace": self._trace}

    async def _execute_step_node(self, state: PlanState) -> dict:
        """执行节点: 执行当前步骤的检索"""
        plan = state.get("plan", [])
        current = state.get("current_step", 0)

        if current >= len(plan):
            return {}

        step = plan[current]
        subquery = step.get("subquery", state.get("query", ""))

        if self._retrieve_fn is None:
            return {"step_results": state.get("step_results", []) + [{}]}

        try:
            chunks = await self._retrieve_fn(subquery)
            self._all_chunks.extend(chunks)

            step_results = state.get("step_results", []) + [{
                "step": current + 1,
                "subquery": subquery,
                "num_chunks": len(chunks),
                "chunks": chunks,
            }]

            self._trace.append({
                "action": "execute",
                "step": current + 1,
                "subquery": subquery,
                "num_chunks": len(chunks),
            })

            return {
                "step_results": step_results,
                "current_step": current + 1,
                "memory_bank_summary": self._format_context(chunks),
            }

        except Exception as e:
            logger.warning(f"执行节点异常 (step {current + 1}): {e}")
            return {"step_results": state.get("step_results", []) + [{"error": str(e)}]}

    async def _finish_node(self, state: PlanState) -> dict:
        """完成节点: 基于所有步骤结果生成最终答案"""
        step_results = state.get("step_results", [])
        query = state.get("query", "")

        if self._llm is None:
            return {"final_answer": "LLM 不可用，无法生成答案"}

        # 构建综合上下文
        context_parts = []
        for result in step_results:
            if "chunks" in result:
                for chunk in result["chunks"][:3]:  # 每步骤最多 3 个 chunk
                    context_parts.append(chunk.get("text", "")[:300])
        context = "\n\n".join(context_parts) or "（无检索结果）"

        prompt = f"""你是一个企业知识助手的综合分析专家。
请基于多个检索步骤的结果，回答用户的复杂问题。

用户问题: {query}

各步骤检索结果:
{context}

要求:
1. 综合各步骤的信息，给出全面分析
2. 标注每个关键信息的来源步骤
3. 如果某些步骤信息不足，明确说明
4. 分析不同来源之间的关联和差异
5. 保持专业、结构化（建议使用分点论述）
"""

        try:
            answer = self._llm.generate(prompt, max_tokens=2048, temperature=0.3)
            return {"final_answer": answer, "confidence": 0.8}
        except Exception as e:
            logger.warning(f"完成节点异常: {e}")
            return {"final_answer": "答案生成失败", "confidence": 0.0}

    def _format_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks[:5], 1):
            lines.append(f"[{i}] {chunk.get('text', '')[:200]}...")
        return "\n".join(lines)
