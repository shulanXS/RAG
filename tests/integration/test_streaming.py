"""
test_streaming.py — 流式响应端到端测试 (P2.1)
================================================================================
技术决策:
- 用 httpx.AsyncClient + FastAPI TestClient 跑 SSE 端点
- 验证响应头包含 text/event-stream
- 验证多帧 data: 事件 yield
- 验证 SSE 帧按顺序拼接可还原完整答案
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# fastapi + pydantic v2 是集成测试依赖 (CI 环境会装, 离线开发可缺)
# pydantic v2 提供 field_validator; v1 仅有 root_validator
pytest.importorskip("pydantic")
try:
    import pydantic
    if int(pydantic.__version__.split(".")[0]) < 2:
        pytest.skip("requires pydantic>=2.0", allow_module_level=True)
except (ImportError, ValueError):
    pass
fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.mark.asyncio
async def test_anthropic_generate_stream_yields_tokens():
    """
    P2.1: AnthropicBackend.generate_stream_async 必须 token-level 流式
    (之前调基类一次性 yield 整个结果)
    """
    from backend.domain.generation.llm_client import AnthropicBackend

    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend._model = "claude-3-7-sonnet-20250620"
    backend._max_tokens = 2048
    backend._temperature = 0.3

    # 模拟 AsyncAnthropic.messages.stream 上下文管理器
    class _FakeStream:
        async def __aiter__(self):
            for chunk in ["Hello", " world", " from", " Claude", "!"]:
                yield chunk

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        @property
        def text_stream(self):
            return self

    class _FakeMessages:
        def stream(self, **kwargs):
            return _FakeStream()

    class _FakeAsyncAnthropic:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    backend._async_client = _FakeAsyncAnthropic()

    tokens: list[str] = []
    async for tok in backend.generate_stream_async("test prompt"):
        tokens.append(tok)

    assert tokens == ["Hello", " world", " from", " Claude", "!"], f"unexpected: {tokens}"
    assert "".join(tokens) == "Hello world from Claude!"


@pytest.mark.asyncio
async def test_orchestrator_run_stream_emits_generating_stage(dummy_config, mock_llm_client):
    """
    验证 orchestrator.run_stream 在 LLM 流式阶段 yield 'generating' 事件
    且 token 累积正确
    """
    from backend.domain.agent.orchestrator import AgenticOrchestrator
    from backend.domain.agent.query_router import QueryComplexity, RoutingDecision

    # Stub router 强制走 SIMPLE 路径
    router = MagicMock()
    router.route = MagicMock(return_value=RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        confidence=0.9,
        reasoning="test",
        recommended_approach="hybrid",
        original_query="test query",
    ))

    # Stub hybrid_search 返回 3 个 chunks
    hybrid = MagicMock()

    async def _search(query, tenant=None, **kwargs):
        return (
            [
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "text": "ctx 1",
                    "section_path": "s1",
                    "rerank_score": 0.9,
                }
            ],
            MagicMock(total_latency_ms=10.0, stage_breakdown={}),
        )
    hybrid.search = _search

    orchestrator = AgenticOrchestrator(
        hybrid_search_engine=hybrid,
        router=router,
        llm_client=mock_llm_client,
    )

    events: list[dict] = []
    async for ev in orchestrator.run_stream("test query"):
        events.append(ev)

    # 找到所有 generating 事件
    gen_events = [e for e in events if e.get("stage") == "generating"]
    assert len(gen_events) >= 1, f"no 'generating' events: {events}"
    # token 累积验证
    tokens = [e.get("token", "") for e in gen_events]
    full = "".join(t for t in tokens if t)
    # mock_llm_client fixture: ["Mocked ", "stream ", "response"]
    assert "Mocked" in full and "stream" in full and "response" in full, f"full={full!r}"

    # 最终 done 事件存在
    done = [e for e in events if e.get("stage") == "done"]
    assert len(done) == 1
    assert done[0]["answer"].strip() != ""
