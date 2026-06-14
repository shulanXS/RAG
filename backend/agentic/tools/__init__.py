"""Agent 工具集 — Calculator / DateTime（通过 ToolRegistry 注册）

P1-B27: WebSearchTool 已删除 (DuckDuckGo 限流/不稳定，生产前应替换商业 API；
Demo 阶段不在范围内)
"""
from backend.agentic.tools.base import BaseTool, ToolResult
from backend.agentic.tools.calculator import CalculatorTool
from backend.agentic.tools.datetime_tool import DateTimeTool
