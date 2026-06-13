"""
tool_registry.py — Tool Registry
================================================================================
技术决策记录:
- Tool Registry 是 Agent 调用外部工具的统一入口。
- 支持工具注册、查找、执行、schema 生成。
- 当前内置工具: Calculator, DateTime, (WebSearch 可扩展)。
- 工具以策略模式实现，可热插拔。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.agentic.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """工具调用请求"""
    tool_name: str
    arguments: dict


class ToolRegistry:
    """
    工具注册表

    设计模式: 注册表模式 + 策略模式
    - 所有工具注册到 registry
    - Agent 通过名称查找和调用工具
    - 新增工具只需实现 BaseTool 接口并注册
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        try:
            from backend.agentic.tools.calculator import CalculatorTool
            self.register(CalculatorTool())
            logger.debug("CalculatorTool registered")
        except ImportError as e:
            logger.warning(f"Failed to register CalculatorTool: {e}")

        try:
            from backend.agentic.tools.datetime_tool import DateTimeTool
            self.register(DateTimeTool())
            logger.debug("DateTimeTool registered")
        except ImportError as e:
            logger.warning(f"Failed to register DateTimeTool: {e}")

    def register(self, tool: BaseTool):
        """
        注册工具

        Args:
            tool: 实现了 BaseTool 接口的工具实例
        """
        self._tools[tool.name] = tool
        logger.info(f"Tool registered: {tool.name}")

    def unregister(self, tool_name: str) -> bool:
        """取消注册工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> BaseTool | None:
        """获取工具实例"""
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名称"""
        return list(self._tools.keys())

    def get_tool_schemas(self) -> list[dict]:
        """获取所有工具的 JSON Schema（用于 ReAct Agent）"""
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        执行工具调用

        Args:
            tool_call: 工具调用请求

        Returns:
            ToolResult: 执行结果
        """
        tool = self.get(tool_call.tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_call.tool_name}",
            )

        try:
            result = await tool.execute(**tool_call.arguments)
            logger.debug(f"Tool executed: {tool_call.tool_name} -> {result.success}")
            return result
        except Exception as e:
            logger.warning(f"Tool execution failed: {tool_call.tool_name}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
            )

    async def execute_by_name(
        self,
        tool_name: str,
        arguments: dict,
    ) -> ToolResult:
        """通过名称执行工具"""
        return await self.execute(ToolCall(tool_name=tool_name, arguments=arguments))


# 全局工具注册表实例
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
