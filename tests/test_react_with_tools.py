"""
test_react_with_tools.py — P2.4 golden test
================================================================================
测试 ReActAgent 接入 ToolRegistry 后能选择正确的工具。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agentic.react_agent import ReActAgent
from backend.agentic.tool_registry import ToolRegistry
from backend.agentic.tools.calculator import CalculatorTool
from backend.agentic.tools.datetime_tool import DateTimeTool


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    # 只用同步可测的工具
    reg.unregister("web_search")
    reg.register(CalculatorTool())
    reg.register(DateTimeTool())
    return reg


@pytest.fixture
def mock_llm_calc() -> Any:
    """LLM 第一次 think 决定 tool_call(calculator)，第二次 think 决定 finish"""
    llm = MagicMock()
    responses = [
        json.dumps({
            "reasoning": "user asked 1+2, use calculator",
            "action": "tool_call",
            "tool_name": "calculator",
            "tool_args": {"expression": "1+2"},
            "next_query": "",
            "confidence": 0.95,
            "answer": "",
        }),
        json.dumps({
            "reasoning": "got 3 from calculator, answer",
            "action": "finish",
            "next_query": "",
            "confidence": 0.95,
            "answer": "1+2 = 3",
        }),
    ]
    iter_resp = iter(responses)

    async def _gen(prompt: str, **kwargs: Any) -> str:
        return next(iter_resp)

    llm.generate_async = _gen
    return llm


@pytest.mark.asyncio
async def test_react_selects_calculator_tool(registry: ToolRegistry, mock_llm_calc: Any) -> None:
    agent = ReActAgent(
        llm_client=mock_llm_calc,
        retrieval_fn=None,
        max_iterations=3,
        tool_registry=registry,
    )
    answer, confidence, chunks = await agent.run("What is 1+2?")
    assert "3" in answer
    assert confidence >= 0.9
    # verify tool was called (trace should contain tool_call action)
    tool_actions = [t for t in agent._trace if t.get("action") == "tool_call"]
    assert len(tool_actions) >= 1
    assert tool_actions[0].get("tool_name") == "calculator"


@pytest.mark.asyncio
async def test_react_selects_datetime_tool(registry: ToolRegistry) -> None:
    llm = MagicMock()
    responses = [
        json.dumps({
            "reasoning": "user asked today's date, use datetime",
            "action": "tool_call",
            "tool_name": "datetime",
            "tool_args": {"format": "%Y-%m-%d"},
            "next_query": "",
            "confidence": 0.95,
            "answer": "",
        }),
        json.dumps({
            "reasoning": "got date, finish",
            "action": "finish",
            "next_query": "",
            "confidence": 0.95,
            "answer": "Today is 2026-06-14",
        }),
    ]
    iter_resp = iter(responses)

    async def _gen(prompt: str, **kwargs: Any) -> str:
        return next(iter_resp)

    llm.generate_async = _gen

    agent = ReActAgent(
        llm_client=llm,
        retrieval_fn=None,
        max_iterations=3,
        tool_registry=registry,
    )
    answer, confidence, _ = await agent.run("What is today's date?")
    assert "2026" in answer or "today" in answer.lower()


@pytest.mark.asyncio
async def test_react_tool_registry_schemas_in_prompt(registry: ToolRegistry) -> None:
    """验证 ToolRegistry schema 被注入到 LLM prompt"""
    schemas = registry.get_tool_schemas()
    tool_names = {s["function"]["name"] for s in schemas}
    assert "calculator" in tool_names
    assert "datetime" in tool_names
    # web_search 在 fixture 中被 unregister 了
    assert "web_search" not in tool_names


@pytest.mark.asyncio
async def test_react_falls_back_when_no_tool_registry() -> None:
    """不传 tool_registry 时不报错（旧行为兼容）"""
    llm = MagicMock()

    async def _gen(prompt: str, **kwargs: Any) -> str:
        return json.dumps({
            "reasoning": "no registry, finish",
            "action": "finish",
            "confidence": 0.5,
            "answer": "ok",
        })

    llm.generate_async = _gen
    agent = ReActAgent(llm_client=llm, retrieval_fn=None, tool_registry=None)
    answer, _, _ = await agent.run("hi")
    assert answer == "ok"
