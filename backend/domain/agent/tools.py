"""
tools.py — 原生 Tool Use 实现(plan §2.1)
================================================================================
2026 FAANG 标准:OpenAI native `tools` 数组 / Anthropic `tool_use` block
替代 prompt JSON 约束(脆弱,容易 hallucinate 字段)。

P3.1(plan §2.1)核心收益:
- 原 _tool_node 是 no-op,ReAct 实际是双步 retrieve+finish
- 修复后 ReAct 名副其实(Reasoning + Acting)
- 3 个真实工具实现:RetrieveTool / CalculatorTool / DateTimeTool
- ToolRegistry + to_openai_tools() 转换函数

面试口径(plan §2.1):
"原 ReAct 用 prompt JSON 强行约束 action,违反 ReAct 原意。
2026 主流是 OpenAI native tools,SDK 解析而非 prompt 解析。"
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Tool 抽象基类

    设计:
    - name: 工具唯一标识(供 LLM 引用)
    - description: 工具功能描述(供 LLM 决定何时调用)
    - parameters: JSON Schema 描述参数
    - execute: 异步执行入口
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """OpenAI-compatible JSON Schema"""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """执行工具并返回文本结果(供 LLM observation)"""
        ...

    def to_openai_schema(self) -> dict:
        """转 OpenAI tools 数组元素

        Output:
        {
            "type": "function",
            "function": {
                "name": ...,
                "description": ...,
                "parameters": ...
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class RetrieveTool(Tool):
    """RAG 检索工具 — 包 hybrid_search"""

    def __init__(self, retrieval_fn: Callable[[str], Awaitable[list[dict]]]):
        self._retrieve_fn = retrieval_fn

    @property
    def name(self) -> str:
        return "retrieve"

    @property
    def description(self) -> str:
        return "从企业知识库中检索与查询相关的文档片段。返回 top-5 chunk 文本。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询,应是完整、独立的问题",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str) -> str:
        try:
            chunks = await self._retrieve_fn(query)
        except Exception as e:
            logger.warning(f"RetrieveTool 失败: {e}")
            return f"检索失败: {e}"

        if not chunks:
            return "未检索到相关文档。"

        top = chunks[:5]
        lines = [f"[{i+1}] (score={c.get('score', 0):.3f}) {c.get('text', '')[:400]}"
                 for i, c in enumerate(top)]
        return "\n\n".join(lines)


class CalculatorTool(Tool):
    """数学计算工具 — Python eval 沙箱(2026 demo 级安全)

    注意:生产环境应使用 AST 限制 + 禁止 dunder/import,本文档 demo 范围内 OK。
    """

    # 禁止的属性/函数黑名单
    _BLOCKED = {"__import__", "eval", "exec", "open", "compile", "globals", "locals"}

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "执行数学表达式(支持 + - * / ** 幂、math 库函数,以及 abs/round/min/max)。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python 数学表达式,如 '(2+3)*4' 或 'math.sqrt(16)'",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, expression: str) -> str:
        try:
            # 基础安全过滤
            for bad in self._BLOCKED:
                if bad in expression:
                    return f"禁止的操作: {bad}"
            # 限制可用命名空间
            safe_ns = {
                "__builtins__": {},
                "math": math,
                "abs": abs, "round": round, "min": min, "max": max,
                "pow": pow, "sum": sum,
            }
            result = eval(expression, safe_ns)
            return f"结果: {result}"
        except Exception as e:
            return f"计算失败: {e}"


class DateTimeTool(Tool):
    """当前时间查询工具"""

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return "获取当前时间(UTC 或本地时区),支持返回 ISO 8601 格式。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "date", "time", "timestamp"],
                    "description": "返回格式:iso=完整时间戳,date=仅日期,time=仅时间,timestamp=Unix 秒",
                },
                "tz": {
                    "type": "string",
                    "description": "时区名(如 Asia/Shanghai),默认 UTC",
                },
            },
            "required": [],
        }

    async def execute(self, format: str = "iso", tz: str = "UTC") -> str:
        try:
            from zoneinfo import ZoneInfo
            zone = ZoneInfo(tz)
        except Exception:
            zone = timezone.utc
        now = datetime.now(zone)
        if format == "date":
            return now.date().isoformat()
        if format == "time":
            return now.time().isoformat()
        if format == "timestamp":
            return str(int(now.timestamp()))
        return now.isoformat()


class ToolRegistry:
    """Tool 注册中心 + 转换器

    Usage:
        registry = ToolRegistry()
        registry.register(RetrieveTool(retrieve_fn))
        registry.register(CalculatorTool())
        registry.register(DateTimeTool())

        tools_schema = registry.to_openai_tools()  # 给 LLM
        result = await registry.execute("calculator", expression="1+1")
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' 重复注册,后注册的会覆盖")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict]:
        """生成 OpenAI `tools` 参数数组(供 chat.completions 调用)"""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, **kwargs) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"未找到工具: {name}"
        return await tool.execute(**kwargs)

    @staticmethod
    def default(retrieval_fn: Callable[[str], Awaitable[list[dict]]] | None = None) -> "ToolRegistry":
        """默认注册集(plan §2.1 面试口径: 3 个具体实现)"""
        reg = ToolRegistry()
        if retrieval_fn is not None:
            reg.register(RetrieveTool(retrieval_fn))
        reg.register(CalculatorTool())
        reg.register(DateTimeTool())
        return reg
