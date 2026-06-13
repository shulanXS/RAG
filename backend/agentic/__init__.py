"""
agentic 模块 — Agentic 编排层
================================================================================
技术决策记录:
- 三大 Agentic 模式: Router（路由）+ ReAct（推理+行动）+ Plan-and-Execute（规划+执行）。
- 三层递进是 2026 年主流架构：从简单到复杂，渐进式引入能力。
- LangGraph: 2026 年生产级 Agent 框架的事实标准。
"""

from backend.agentic.query_router import QueryComplexity, QueryRouter
from backend.agentic.react_agent import ReActAgent, ReActState
from backend.agentic.plan_execute import PlanExecuteAgent, PlanState
from backend.agentic.orchestrator import AgenticOrchestrator, OrchestratorResult, SimpleMemoryBank
from backend.agentic.tool_registry import (
    ToolRegistry,
    ToolCall,
    get_tool_registry,
)

__all__ = [
    # Core
    "QueryComplexity",
    "QueryRouter",
    "ReActAgent",
    "ReActState",
    "PlanExecuteAgent",
    "PlanState",
    "AgenticOrchestrator",
    "OrchestratorResult",
    "SimpleMemoryBank",
    # Tool Use
    "ToolRegistry",
    "ToolCall",
    "get_tool_registry",
]
