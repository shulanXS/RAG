"""
agentic 模块 — Agentic 编排层
================================================================================
技术决策记录:
- 三大 Agentic 模式: Router（路由）+ ReAct（推理+行动）+ Plan-and-Execute（规划+执行）。
  三层递进是 2026 年主流架构：从简单到复杂，渐进式引入能力。
- LangGraph: 2026 年生产级 Agent 框架的事实标准。
  核心优势: 状态机抽象、条件分支、人类介入中断、确定性审计轨迹。
- Haiku 路由: 用轻量 LLM（Haiku 4.5）判断查询复杂度，成本可忽略（$0.8/1M tokens）。

业务难点:
- Agentic 系统的失败模式: 检索循环、路由错误、over-retrieval。
  缓解: max_iterations 限制 + 置信度阈值提前退出。
"""

from backend.agentic.query_router import QueryComplexity, QueryRouter
from backend.agentic.react_agent import ReActAgent, ReActState
from backend.agentic.plan_execute import PlanExecuteAgent, PlanState
from backend.agentic.memory_bank import MemoryBank, Claim, Evidence
from backend.agentic.orchestrator import AgenticOrchestrator, OrchestratorResult

__all__ = [
    "QueryComplexity",
    "QueryRouter",
    "ReActAgent",
    "ReActState",
    "PlanExecuteAgent",
    "PlanState",
    "MemoryBank",
    "Claim",
    "Evidence",
    "AgenticOrchestrator",
    "OrchestratorResult",
]
