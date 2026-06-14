"""
agentic 模块 — Agentic 编排层
================================================================================
技术决策记录:
- 两大 Agentic 模式: Router（路由）+ ReAct（推理+行动）。
  Plan-and-Execute 在 P0 阶段被移除 — 实际生产中 Plan-and-Execute
  容易陷入规划爆炸（max_steps 强制退出是常见 anti-pattern），FAANG
  2026 主流只用 Router + ReAct。
- LangGraph: 2026 年生产级 Agent 框架的事实标准。
"""

from backend.agentic.query_router import QueryComplexity, QueryRouter
from backend.agentic.react_agent import ReActAgent, ReActState
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
    "AgenticOrchestrator",
    "OrchestratorResult",
    "SimpleMemoryBank",
    # Tool Use
    "ToolRegistry",
    "ToolCall",
    "get_tool_registry",
]
