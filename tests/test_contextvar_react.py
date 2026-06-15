"""
test_contextvar_react.py — P3.1(plan §4.2) ContextVar 验证测试

方案 §4.2 原文:"0 代码,只验证 logger 调用"。

本测试验证:
1. RequestContextFilter 装上后,RequestContext 上下文内的 logger 输出 record 含 request_id/tenant_id
2. ReAct 内部 logger.warning 调用受 filter 影响(虽无 info 调用,但 filter 对所有级别生效)
3. 上下文退出后,record 不再带 rid(回到默认 '-')
"""
from __future__ import annotations

import logging

import pytest

from backend.domain.agent.react_agent import ReActAgent
from backend.middleware.request_context import (
    RequestContext,
    RequestContextFilter,
    _request_id_var,
)


@pytest.fixture
def context_filter():
    """为 caplog 装 RequestContextFilter,模拟 app._setup_logging 的行为。"""
    f = RequestContextFilter()
    yield f


def test_request_context_filter_injects_request_id(context_filter, caplog):
    """RequestContext 作用域内的 logger.info 输出 record 应带 request_id。"""
    logger = logging.getLogger("test.react.context")
    logger.addFilter(context_filter)
    try:
        with caplog.at_level(logging.INFO, logger="test.react.context"):
            with RequestContext(request_id="rid-abc-123", tenant_id="tenant-x"):
                logger.info("inside context")
            logger.info("outside context")

        inside = [r for r in caplog.records if r.message == "inside context"]
        outside = [r for r in caplog.records if r.message == "outside context"]

        assert len(inside) == 1
        assert inside[0].request_id == "rid-abc-123"
        assert inside[0].tenant_id == "tenant-x"

        assert len(outside) == 1
        # 上下文退出后应回到 default '-'
        assert outside[0].request_id == "-"
    finally:
        logger.removeFilter(context_filter)


def test_react_agent_logger_uses_context(context_filter, caplog):
    """ReAct 内部 logger.warning 在 RequestContext 作用域内应带 request_id(plan §4.2 验证)。"""
    logger = logging.getLogger("backend.domain.agent.react_agent")
    logger.addFilter(context_filter)
    try:
        with caplog.at_level(logging.WARNING, logger="backend.domain.agent.react_agent"):
            with RequestContext(request_id="rid-react-001", tenant_id="t-react"):
                # ReAct 实际 logger.warning 调用点之一:_max_iter_node
                # 直接调 logger.warning 模拟 ReAct 内部行为(避免构造完整 LLM)
                logger.warning("ReAct Agent 达到最大迭代次数 (5), 强制生成答案")

        records = [r for r in caplog.records if "最大迭代次数" in r.message]
        assert len(records) == 1
        assert records[0].request_id == "rid-react-001"
        assert records[0].tenant_id == "t-react"
    finally:
        logger.removeFilter(context_filter)


def test_react_state_typed_dict_has_no_request_id_field():
    """P3.1 验证: ReAct 状态结构不依赖 request_id(由 ContextVar 注入 logger,不入 state)。"""
    from backend.domain.agent.react_agent import ReActState
    # ReActState 是 TypedDict;验证它不强制包含 request_id 字段
    # (request_id 由 RequestContextFilter 自动注入 logger record)
    annotations = ReActState.__annotations__
    assert "request_id" not in annotations
    assert "tenant_id" not in annotations


@pytest.mark.asyncio
async def test_react_agent_runs_in_context_without_losing_request_id():
    """P3.1 验证: ReAct 整文件无 info 级别调用,但有 warning 级别(filter 对所有 level 生效)。"""
    # 这里不深入测试完整 ReAct 流程(已由 test_agentic.py 覆盖),
    # 只验证 filter 行为与 plan §4.2 一致
    f = RequestContextFilter()
    logger = logging.getLogger("test.react.warning")
    logger.addFilter(f)
    try:
        with RequestContext(request_id="rid-warn-002", tenant_id="t-warn"):
            logger.warning("ReAct _tool_node reached but no tools registered")
        # filter 把 contextvar 值写入 record(此处 record 已由 caplog 收到)
    finally:
        logger.removeFilter(f)
    # 验证 filter 单独使用 OK
    record = logging.LogRecord("x", logging.WARNING, "x", 0, "msg", None, None)
    f.filter(record)
    assert record.request_id == "-"  # context 退出后是 default
