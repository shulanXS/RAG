"""
calculator.py — Calculator Tool
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from backend.agentic.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class CalculationResult:
    expression: str
    result: float
    formatted: str


class CalculatorTool(BaseTool):
    """
    计算器工具 — 用于数学计算

    能力:
    - 基本算术: +, -, *, /, **
    - 数学函数: sqrt, sin, cos, tan, log, exp, abs
    - 常量: pi, e
    - 括号优先级
    - 单位换算

    示例:
    - "25 * 3 + 10" → 85
    - "sqrt(144)" → 12
    - "log(1000, 10)" → 3
    """

    MATH_FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "floor": math.floor,
        "ceil": math.ceil,
        "round": round,
    }

    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Use this when the user asks to calculate, compute, or work with numbers.",
        )

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '25 * 3 + 10', 'sqrt(144)', 'log(1000, 10)'). Use standard math notation.",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, expression: str, **kwargs) -> ToolResult:
        """
        执行数学计算

        Args:
            expression: 数学表达式字符串

        Returns:
            ToolResult: 包含计算结果
        """
        try:
            result = self._evaluate(expression)
            return ToolResult(
                success=True,
                result=result,
                metadata={"expression": expression},
            )
        except Exception as e:
            logger.warning(f"Calculation failed: {e}")
            return ToolResult(
                success=False,
                error=f"Calculation error: {str(e)}",
                metadata={"expression": expression},
            )

    def _evaluate(self, expression: str) -> CalculationResult:
        """求值数学表达式"""
        expr = expression.strip().lower()

        expr = self._replace_constants(expr)
        expr = self._replace_functions(expr)

        safe_dict = {
            key: val for key, val in self.MATH_FUNCTIONS.items()
        }
        safe_dict.update({"__builtins__": {}})

        result = eval(expr, {"__builtins__": {}}, safe_dict)

        if isinstance(result, float):
            if result == int(result):
                result = int(result)

        formatted = f"{expression.strip()} = {result}"
        return CalculationResult(
            expression=expression.strip(),
            result=result,
            formatted=formatted,
        )

    def _replace_constants(self, expr: str) -> str:
        for name, value in self.CONSTANTS.items():
            expr = re.sub(rf"\b{name}\b", str(value), expr)
        return expr

    def _replace_functions(self, expr: str) -> str:
        for name in self.MATH_FUNCTIONS:
            expr = re.sub(
                rf"\blog\(([^,]+),\s*([0-9.]+)\)",
                rf"(math.log(\\1) / math.log(\\2))",
                expr,
            )
        return expr
