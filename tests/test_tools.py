"""
test_tools.py — 原生 Tool Use 单元测试(plan §2.1)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.domain.agent.tools import (
    CalculatorTool,
    DateTimeTool,
    RetrieveTool,
    ToolRegistry,
)


@pytest.mark.asyncio
async def test_calculator_basic():
    t = CalculatorTool()
    assert await t.execute(expression="2+2") == "结果: 4"
    assert "20" in await t.execute(expression="(2+3)*4")
    assert "16" in await t.execute(expression="math.sqrt(256)")


@pytest.mark.asyncio
async def test_calculator_blocks_unsafe():
    t = CalculatorTool()
    out = await t.execute(expression="__import__('os')")
    assert "禁止" in out
    out = await t.execute(expression="open('/etc/passwd').read()")
    assert "禁止" in out


@pytest.mark.asyncio
async def test_datetime_default_iso():
    t = DateTimeTool()
    out = await t.execute()
    # ISO 8601 contains T and date
    assert "T" in out
    assert len(out) >= 19


@pytest.mark.asyncio
async def test_datetime_date_only():
    t = DateTimeTool()
    out = await t.execute(format="date")
    assert len(out) == 10
    assert out[4] == "-"


@pytest.mark.asyncio
async def test_datetime_timestamp():
    t = DateTimeTool()
    out = await t.execute(format="timestamp")
    assert out.isdigit()
    assert int(out) > 1_700_000_000


@pytest.mark.asyncio
async def test_retrieve_tool_happy_path():
    fake_fn = AsyncMock(return_value=[
        {"chunk_id": "c1", "score": 0.9, "text": "alpha content"},
        {"chunk_id": "c2", "score": 0.5, "text": "beta content"},
    ])
    t = RetrieveTool(retrieval_fn=fake_fn)
    out = await t.execute(query="hello")
    assert "[1]" in out
    assert "alpha" in out
    assert "[2]" in out


@pytest.mark.asyncio
async def test_retrieve_tool_empty():
    fake_fn = AsyncMock(return_value=[])
    t = RetrieveTool(retrieval_fn=fake_fn)
    out = await t.execute(query="x")
    assert "未检索到" in out


@pytest.mark.asyncio
async def test_retrieve_tool_exception():
    fake_fn = AsyncMock(side_effect=RuntimeError("boom"))
    t = RetrieveTool(retrieval_fn=fake_fn)
    out = await t.execute(query="x")
    assert "检索失败" in out


def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(DateTimeTool())
    assert reg.list_names() == ["calculator", "datetime"]
    assert isinstance(reg.get("calculator"), CalculatorTool)
    assert reg.get("nope") is None


def test_registry_to_openai_tools_shape():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(DateTimeTool())
    schemas = reg.to_openai_tools()
    assert len(schemas) == 2
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]


@pytest.mark.asyncio
async def test_registry_execute_unknown():
    reg = ToolRegistry()
    out = await reg.execute("ghost")
    assert "未找到" in out


@pytest.mark.asyncio
async def test_registry_default_factory():
    reg = ToolRegistry.default(retrieval_fn=AsyncMock(return_value=[]))
    assert "calculator" in reg.list_names()
    assert "datetime" in reg.list_names()
    assert "retrieve" in reg.list_names()


def test_tool_schema_compatibility_with_openai_spec():
    """确保 parameters 字段符合 OpenAI tools spec(type=object + properties + required)"""
    t = CalculatorTool()
    s = t.to_openai_schema()
    p = s["function"]["parameters"]
    assert p["type"] == "object"
    assert "properties" in p
    assert "expression" in p["properties"]
    assert p["required"] == ["expression"]
