"""
base.py — Tool Interface Definitions
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """
    工具执行结果

    字段说明:
    - success: 是否成功
    - result: 执行结果（文本或结构化数据）
    - error: 错误信息（如果有）
    - metadata: 额外的元信息
    """
    success: bool
    result: Any = None
    error: str = ""
    metadata: dict = field(default_factory=dict)


class BaseTool(ABC):
    """
    工具抽象基类

    设计模式: 策略模式
    - 所有工具都实现统一的 execute() 接口
    - ToolRegistry 统一管理工具的注册和调用
    """

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        执行工具

        Args:
            **kwargs: 工具特定参数

        Returns:
            ToolResult: 执行结果
        """
        ...

    def get_schema(self) -> dict:
        """
        获取工具的 JSON Schema 定义（用于 ReAct Agent 的工具选择）

        Returns:
            dict: OpenAI tool calling 格式的 schema
        """
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": self._get_parameters_schema(),
            },
        }

    def _get_parameters_schema(self) -> dict:
        """子类可覆盖，返回自定义参数 schema"""
        return {"type": "object", "properties": {}, "required": []}
