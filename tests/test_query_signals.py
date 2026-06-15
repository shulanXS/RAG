"""
test_query_signals.py — QueryAnalyzer + QuerySignals (P2-B6)

覆盖:
- 各种 pronoun / entity / multi-hop / quote / length 启发式
- complexity_hint() 兜底逻辑 (无 LLM 时的复杂度判断)
- to_dict() 序列化
"""
from __future__ import annotations

import pytest

from backend.agentic.query_signals import (
    QueryAnalyzer,
    QuerySignals,
)


# ===========================================================================
# TestQueryAnalyzer: 各种信号检测
# ===========================================================================

class TestQueryAnalyzer:
    """QueryAnalyzer 纯规则信号的准确率 (规则集是手工调过的, 这里只 smoke)"""

    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        return QueryAnalyzer()

    # ---- pronoun ----

    def test_chinese_pronoun_detected(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("他的合同和她的合同哪个先签？")
        assert s.has_pronoun is True

    def test_english_pronoun_detected(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("What did they say about his contract?")
        assert s.has_pronoun is True

    def test_no_pronoun(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("Acme公司2023年Q4的营收是多少？")
        assert s.has_pronoun is False

    # ---- multi-hop ----

    def test_chinese_multihop_detected(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("X供应商断供之后，对Y客户的产品交付有何影响？")
        assert s.is_multi_hop is True

    def test_english_multihop_detected(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("Compare X and Y, then synthesize the impact.")
        assert s.is_multi_hop is True

    def test_simple_query_no_multihop(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("合同编号 A-2024-001 的甲方是谁？")
        assert s.is_multi_hop is False

    # ---- entity_count ----

    def test_english_capitalized_entities(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("Compare Acme Corp and Beta Industries contracts")
        assert s.entity_count >= 2

    def test_chinese_entities_with_suffix(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("苹果公司和微软公司哪个营收更高？")
        assert s.entity_count >= 2

    def test_short_query_low_entity_count(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("今天天气？")
        assert s.entity_count <= 1

    # ---- quote ----

    def test_chinese_double_quote(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze('什么是"向量检索"？')
        assert s.has_quote is True

    def test_english_double_quote(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze('What does "RAG" mean?')
        assert s.has_quote is True

    def test_no_quote(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("查询合同的甲方是谁")
        assert s.has_quote is False

    # ---- length ----

    def test_length_counts_chars(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("abcdefghij")
        assert s.query_length == 10

    def test_length_chinese_chars(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("中文查询")
        assert s.query_length == 4

    # ---- empty / edge cases ----

    def test_empty_query(self, analyzer: QueryAnalyzer):
        s = analyzer.analyze("")
        assert s.has_pronoun is False
        assert s.entity_count == 0
        assert s.is_multi_hop is False
        assert s.query_length == 0
        assert s.has_quote is False

    def test_performance_under_5ms(self, analyzer: QueryAnalyzer):
        """P2 验收: analyzer 必须 < 1ms (这里宽松到 5ms 给 CI 余量)"""
        import time
        query = "X供应商断供之后，对Y客户的产品交付有何影响？请详细对比" * 5
        t0 = time.perf_counter()
        for _ in range(100):
            analyzer.analyze(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / 100
        assert elapsed_ms < 5, f"avg {elapsed_ms:.2f}ms > 5ms threshold"


# ===========================================================================
# TestQuerySignals: 数据结构本身
# ===========================================================================

class TestQuerySignals:
    def test_default_construction(self):
        s = QuerySignals()
        assert s.has_pronoun is False
        assert s.entity_count == 0
        assert s.is_multi_hop is False
        assert s.query_length == 0
        assert s.has_quote is False
        # Phase 1.12: `extra` 字段已删除（无外部消费方）
        assert not hasattr(s, "extra") or "extra" not in s.to_dict()

    def test_to_dict_round_trip(self):
        s = QuerySignals(
            has_pronoun=True,
            entity_count=3,
            is_multi_hop=True,
            query_length=42,
            has_quote=False,
        )
        d = s.to_dict()
        assert d["has_pronoun"] is True
        assert d["entity_count"] == 3
        assert d["is_multi_hop"] is True
        assert d["query_length"] == 42
        assert d["has_quote"] is False


# ===========================================================================
# TestComplexityHint: 无 LLM 兜底
# ===========================================================================

class TestComplexityHint:
    """signals.complexity_hint() 在 QueryRouter 无 LLM 时使用"""

    def test_short_simple_query(self):
        s = QuerySignals(query_length=10, entity_count=0, is_multi_hop=False, has_pronoun=False)
        assert s.complexity_hint() == "simple"

    def test_short_query_with_pronoun(self):
        """短 + 代词 → moderate (需查 history)"""
        s = QuerySignals(query_length=8, has_pronoun=True)
        assert s.complexity_hint() == "moderate"

    def test_long_multihop_complex(self):
        s = QuerySignals(query_length=50, is_multi_hop=True, entity_count=3)
        assert s.complexity_hint() == "complex"

    def test_moderate_default(self):
        s = QuerySignals(query_length=25, is_multi_hop=False, entity_count=1)
        assert s.complexity_hint() == "moderate"
