"""
test_agentic.py — Agentic 编排核心测试
================================================================================
P0-3: Calculator / DateTime Tool 已删除，故 TestCalculatorTool / TestDateTimeTool /
TestToolRegistry 全部移除。本文件保留 QueryRouter 路由分类测试。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from backend.agentic.query_router import (
    QueryComplexity,
    QueryRouter,
    RoutingDecision,
)


# =============================================================================
# 1. QueryRouter
# =============================================================================


class TestQueryRouter:
    """QueryRouter 路由分类测试 — 覆盖无 LLM 退化 + JSON 解析 + 置信度升级"""

    def test_no_llm_defaults_to_moderate(self):
        """无 LLM client 时降级到 MODERATE（保守策略, P2-B6 用 signals 兜底）"""
        router = QueryRouter(llm_client=None, complexity_threshold=0.6)
        decision = router.route("合同编号 A-2024-001 的甲方是谁？")

        assert isinstance(decision, RoutingDecision)
        assert decision.original_query == "合同编号 A-2024-001 的甲方是谁？"
        assert decision.complexity == QueryComplexity.MODERATE
        assert decision.confidence == 0.5
        assert "signals-based hint" in decision.reasoning.lower()
        assert decision.signals is not None
        assert decision.signals.query_length == 22

    def test_no_llm_short_simple_query_yields_simple(self):
        """P2-B6: 短查询无 LLM 时, signals 兜底为 SIMPLE"""
        router = QueryRouter(llm_client=None, complexity_threshold=0.6)
        decision = router.route("什么是RAG？")
        assert decision.complexity == QueryComplexity.SIMPLE
        assert decision.signals is not None
        assert decision.signals.has_quote is False

    def test_routing_decision_carries_signals(self):
        """P2-B6: RoutingDecision 增字段 signals, 不论 LLM 是否可用"""
        router = QueryRouter(llm_client=None, complexity_threshold=0.6)
        decision = router.route("Compare X and Y")
        assert decision.signals is not None
        assert decision.signals.is_multi_hop is True

    def test_llm_json_parsed_simple(self):
        """LLM 返回 simple 分类时正常解析"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "complexity": "simple",
            "confidence": 0.95,
            "reasoning": "单实体事实查询",
        })

        router = QueryRouter(llm_client=mock_llm, complexity_threshold=0.6)
        decision = router.route("X产品的价格是多少？")

        assert decision.complexity == QueryComplexity.SIMPLE
        assert decision.confidence == 0.95
        assert "ReAct" not in decision.recommended_approach
        mock_llm.generate.assert_called_once()

    def test_low_confidence_escalates_simple_to_moderate(self):
        """低置信度的 simple 应升级到 moderate"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "complexity": "simple",
            "confidence": 0.4,
            "reasoning": "可能是简单查询",
        })

        router = QueryRouter(llm_client=mock_llm, complexity_threshold=0.6)
        decision = router.route("某个查询")

        assert decision.complexity == QueryComplexity.MODERATE
        assert "升级" in decision.reasoning

    def test_low_confidence_escalates_moderate_to_complex(self):
        """低置信度的 moderate 应升级到 complex"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "complexity": "moderate",
            "confidence": 0.3,
            "reasoning": "不确定",
        })

        router = QueryRouter(llm_client=mock_llm, complexity_threshold=0.6)
        decision = router.route("复杂查询")

        assert decision.complexity == QueryComplexity.COMPLEX

    def test_invalid_complexity_falls_back_to_moderate(self):
        """LLM 返回非法 complexity 值时降级到 moderate"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "complexity": "unknown_garbage",
            "confidence": 0.7,
            "reasoning": "测试",
        })

        router = QueryRouter(llm_client=mock_llm)
        decision = router.route("test query")

        assert decision.complexity == QueryComplexity.MODERATE

    def test_llm_exception_falls_back_to_moderate(self):
        """LLM 调用异常时不崩, 默认 moderate"""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("API timeout")

        router = QueryRouter(llm_client=mock_llm)
        decision = router.route("任意查询")

        assert decision.complexity == QueryComplexity.MODERATE
        assert decision.confidence == 0.0
        assert "异常" in decision.reasoning or "timeout" in decision.reasoning

    def test_history_included_in_prompt(self):
        """对话历史被注入 prompt（最后 2 轮）"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "complexity": "simple",
            "confidence": 0.9,
            "reasoning": "test",
        })

        router = QueryRouter(llm_client=mock_llm)
        marker_old = "唯一标识_旧消息XYZ123"
        marker_new = "唯一标识_新消息ABC789"
        history = [
            {"role": "user", "content": marker_old},
            {"role": "assistant", "content": marker_old + "_resp"},
            {"role": "user", "content": marker_new},
            {"role": "assistant", "content": marker_new + "_resp"},
        ]
        router.route("当前查询", history=history)

        call_prompt = mock_llm.generate.call_args[0][0]
        assert marker_new in call_prompt
        assert marker_old not in call_prompt
