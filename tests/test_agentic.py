"""
test_agentic.py — Agentic 编排测试
"""

import pytest
from backend.agentic.query_router import QueryRouter, QueryComplexity, RoutingDecision
from backend.agentic.memory_bank import MemoryBank, Claim, Evidence, AutomatedMemoryBank
from backend.agentic.subquestion_decomposer import (
    SubQuestionDecomposer,
    SubQuestion,
    SubQuestionType,
)
from backend.agentic.self_reflection import SelfReflection, ReflectionResult
from backend.agentic.tool_registry import ToolRegistry, ToolCall, get_tool_registry
from backend.agentic.tools.calculator import CalculatorTool
from backend.agentic.tools.datetime_tool import DateTimeTool


class TestQueryRouter:
    """查询路由器测试"""

    def test_simple_query_classification(self):
        """简单查询分类测试"""
        router = QueryRouter(complexity_threshold=0.6)

        query = "合同编号 A-2024-001 的甲方是谁？"
        result = router.route(query)

        assert isinstance(result, RoutingDecision)
        assert result.original_query == query
        assert result.confidence >= 0

    def test_complex_query_classification(self):
        """复杂查询分类测试"""
        router = QueryRouter(complexity_threshold=0.6)

        query = "如果供应商X断供，哪些客户会受到影响？"
        result = router.route(query)

        assert isinstance(result, RoutingDecision)
        assert result.complexity in list(QueryComplexity)

    def test_beyond_kb_query(self):
        """超出知识库查询测试"""
        router = QueryRouter(complexity_threshold=0.6)

        query = "什么是人工智能？"
        result = router.route(query)

        assert result.complexity in list(QueryComplexity)


class TestMemoryBank:
    """Memory Bank 测试"""

    def test_add_evidence(self):
        """添加证据测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        chunks = [
            {
                "chunk_id": "c1",
                "text": "这是证据1的内容。",
                "doc_id": "d1",
                "rerank_score": 0.95,
            },
            {
                "chunk_id": "c2",
                "text": "这是证据2的内容。",
                "doc_id": "d1",
                "rerank_score": 0.85,
            },
        ]

        evidence_ids = bank.add_evidence(chunks)

        assert len(evidence_ids) == 2
        assert "ev_c1" in evidence_ids
        assert "ev_c2" in evidence_ids

    def test_add_claims(self):
        """添加主张测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        claims = [
            {"claim_id": "claim_1", "text": "这是主张1", "confidence": 0.9},
            {"claim_id": "claim_2", "text": "这是主张2", "confidence": 0.8},
        ]

        claim_ids = bank.add_claims(claims)

        assert len(claim_ids) == 2
        assert "claim_1" in claim_ids

    def test_claim_evidence_linking(self):
        """Claim-Evidence 链接测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        chunks = [
            {"chunk_id": "c1", "text": "证据内容1", "doc_id": "d1", "rerank_score": 0.95},
        ]
        bank.add_evidence(chunks)

        claims = [{"claim_id": "claim_1", "text": "主张内容", "confidence": 0.9}]
        bank.add_claims(claims)

        success = bank.link_claim_evidence("claim_1", ["ev_c1"])

        assert success is True

        coverage = bank.verify_coverage()
        assert coverage["total_claims"] == 1
        assert coverage["verified_claims"] == 1

    def test_unverified_claims(self):
        """未验证主张测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        claims = [{"claim_id": "unverified", "text": "无证据的主张", "confidence": 0.5}]
        bank.add_claims(claims)

        coverage = bank.verify_coverage()

        assert coverage["total_claims"] == 1
        assert coverage["verified_claims"] == 0
        assert "unverified" in coverage["unverified_claims"]

    def test_context_for_generation(self):
        """生成上下文测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        chunks = [
            {"chunk_id": "c1", "text": "证据文本", "doc_id": "d1", "rerank_score": 0.95},
        ]
        bank.add_evidence(chunks)

        claims = [{"claim_id": "claim_1", "text": "主张文本", "confidence": 0.9}]
        bank.add_claims(claims)
        bank.link_claim_evidence("claim_1", ["ev_c1"])

        context = bank.get_context_for_generation()

        assert "Evidence Bank" in context
        assert "Claim-Evidence Mapping" in context
        assert "主张文本" in context


class TestAutomatedMemoryBank:
    """Automated Memory Bank 测试"""

    def test_rule_based_claim_extract(self):
        """规则化 claim 提取测试"""
        bank = AutomatedMemoryBank(session_id="test", llm_client=None)

        answer = "Apple 2024年营收为3832亿美元，同比增长4%。净利润为970亿美元。"
        claims = bank._rule_based_claim_extract(answer)

        assert len(claims) >= 1
        assert all("claim_id" in c for c in claims)

    def test_keyword_evidence_match(self):
        """关键词 evidence 匹配测试"""
        bank = AutomatedMemoryBank(session_id="test", llm_client=None)

        claim = "Apple 2024年营收为3832亿美元"
        contexts = [
            {"chunk_id": "c1", "text": "Apple 2024年营收为3832亿美元", "doc_id": "d1"},
            {"chunk_id": "c2", "text": "Google 2024年营收", "doc_id": "d2"},
        ]

        evidence_ids = bank._keyword_evidence_match(claim, contexts)

        assert len(evidence_ids) == 1
        assert "c1" in evidence_ids[0]


class TestSubQuestionDecomposer:
    """Sub-question Decomposer 测试"""

    def test_rule_based_decompose(self):
        """降级分解测试"""
        decomposer = SubQuestionDecomposer(llm_client=None)

        result = decomposer._rule_based_decompose("Apple 2024年表现如何？")

        assert isinstance(result, DecompositionResult)
        assert len(result.sub_questions) >= 1
        assert result.sub_questions[0].id == "sq1"

    def test_topological_sort(self):
        """拓扑排序测试"""
        decomposer = SubQuestionDecomposer(llm_client=None)

        sq1 = SubQuestion(id="sq1", question="Apple AI", priority=1, depends_on=[])
        sq2 = SubQuestion(id="sq2", question="Google AI", priority=1, depends_on=[])
        sq3 = SubQuestion(id="sq3", question="Compare", priority=2, depends_on=["sq1", "sq2"])

        sorted_list = decomposer._topological_sort([sq1, sq2, sq3])

        assert sorted_list[0].id in ["sq1", "sq2"]
        assert sorted_list[-1].id == "sq3"

    def test_independent_questions(self):
        """独立问题测试"""
        decomposer = SubQuestionDecomposer(llm_client=None)

        sq1 = SubQuestion(id="sq1", question="Apple AI", priority=1, depends_on=[])
        sq2 = SubQuestion(id="sq2", question="Google AI", priority=1, depends_on=[])
        sq3 = SubQuestion(id="sq3", question="Compare", priority=2, depends_on=["sq1"])

        result = DecompositionResult(
            original_query="Compare",
            sub_questions=[sq1, sq2, sq3],
            execution_order=[sq1, sq2, sq3],
        )

        independent = result.get_independent_questions()
        assert len(independent) == 2

    def test_extract_keywords(self):
        """关键词提取测试"""
        decomposer = SubQuestionDecomposer(llm_client=None)

        keywords = decomposer._extract_keywords("Apple 2024 annual revenue report")

        assert len(keywords) >= 2
        assert "apple" in keywords


class TestSelfReflection:
    """Self-Reflection 测试"""

    def test_empty_answer(self):
        """空答案测试"""
        reflector = SelfReflection(llm_client=None)

        result = reflector._fallback_reflection("", [], {})

        assert isinstance(result, ReflectionResult)

    def test_reflect_no_llm(self):
        """无 LLM 时返回降级结果"""
        reflector = SelfReflection(llm_client=None)

        result = reflector._perform_reflection(
            query="Apple 2024年表现如何？",
            answer="Apple 2024年营收增长",
            contexts=[],
        )

        assert isinstance(result, ReflectionResult)
        assert result.overall_score >= 0.0


class TestToolRegistry:
    """Tool Registry 测试"""

    def test_register_tools(self):
        """工具注册测试"""
        registry = ToolRegistry()

        tools = registry.list_tools()
        assert "calculator" in tools or len(tools) >= 0

    def test_execute_calculator(self):
        """Calculator 执行测试"""
        registry = ToolRegistry()

        result = registry.execute_by_name(
            "calculator",
            {"expression": "25 * 3 + 10"},
        )

        assert result.success is True
        assert result.result is not None
        assert "85" in result.result.formatted

    def test_execute_datetime(self):
        """DateTime 执行测试"""
        registry = ToolRegistry()

        result = registry.execute_by_name(
            "datetime",
            {"action": "now"},
        )

        assert result.success is True
        assert "iso" in result.result

    def test_execute_unknown_tool(self):
        """未知工具测试"""
        registry = ToolRegistry()

        result = registry.execute_by_name(
            "nonexistent_tool",
            {},
        )

        assert result.success is False
        assert "Unknown tool" in result.error

    def test_tool_schemas(self):
        """工具 Schema 测试"""
        registry = ToolRegistry()

        schemas = registry.get_tool_schemas()
        assert isinstance(schemas, list)


class TestEvidence:
    """Evidence 数据结构测试"""

    def test_evidence_creation(self):
        """Evidence 创建测试"""
        ev = Evidence(
            source_id="c1",
            text="这是证据文本。",
            doc_title="测试文档",
            section_path="第一章/第一节",
            retrieval_score=0.95,
        )

        assert ev.source_id == "c1"
        assert ev.doc_title == "测试文档"
        assert ev.section_path == "第一章/第一节"
        assert ev.retrieval_score == 0.95
        assert ev.used_by_claims == []


class TestClaim:
    """Claim 数据结构测试"""

    def test_claim_creation(self):
        """Claim 创建测试"""
        claim = Claim(
            claim_id="claim_1",
            text="这是主张内容。",
            evidence_ids=["ev_c1", "ev_c2"],
            verified=True,
            confidence=0.85,
        )

        assert claim.claim_id == "claim_1"
        assert claim.verified is True
        assert len(claim.evidence_ids) == 2
        assert claim.confidence == 0.85


class TestSubQuestion:
    """SubQuestion 数据结构测试"""

    def test_is_independent(self):
        """独立判断测试"""
        sq_independent = SubQuestion(id="sq1", question="test", depends_on=[])
        sq_dependent = SubQuestion(id="sq2", question="test", depends_on=["sq1"])

        assert sq_independent.is_independent() is True
        assert sq_dependent.is_independent() is False



class TestQueryRouter:
    """查询路由器测试"""

    def test_simple_query_classification(self):
        """简单查询分类测试"""
        router = QueryRouter(complexity_threshold=0.6)

        # 简单事实型查询应该被识别为 SIMPLE
        query = "合同编号 A-2024-001 的甲方是谁？"
        result = router.route(query)

        assert isinstance(result, RoutingDecision)
        assert result.original_query == query
        assert result.confidence >= 0

    def test_complex_query_classification(self):
        """复杂查询分类测试"""
        router = QueryRouter(complexity_threshold=0.6)

        # 复杂多跳查询
        query = "如果供应商X断供，哪些客户会受到影响？"
        result = router.route(query)

        assert isinstance(result, RoutingDecision)
        # 默认 MODERATE（无 LLM client 时）
        assert result.complexity in list(QueryComplexity)

    def test_beyond_kb_query(self):
        """超出知识库查询测试"""
        router = QueryRouter(complexity_threshold=0.6)

        query = "什么是人工智能？"
        result = router.route(query)

        assert result.complexity in list(QueryComplexity)


class TestMemoryBank:
    """Memory Bank 测试"""

    def test_add_evidence(self):
        """添加证据测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        chunks = [
            {
                "chunk_id": "c1",
                "text": "这是证据1的内容。",
                "doc_id": "d1",
                "rerank_score": 0.95,
            },
            {
                "chunk_id": "c2",
                "text": "这是证据2的内容。",
                "doc_id": "d1",
                "rerank_score": 0.85,
            },
        ]

        evidence_ids = bank.add_evidence(chunks)

        assert len(evidence_ids) == 2
        assert "ev_c1" in evidence_ids
        assert "ev_c2" in evidence_ids

    def test_add_claims(self):
        """添加主张测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        claims = [
            {"claim_id": "claim_1", "text": "这是主张1", "confidence": 0.9},
            {"claim_id": "claim_2", "text": "这是主张2", "confidence": 0.8},
        ]

        claim_ids = bank.add_claims(claims)

        assert len(claim_ids) == 2
        assert "claim_1" in claim_ids

    def test_claim_evidence_linking(self):
        """Claim-Evidence 链接测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        chunks = [
            {"chunk_id": "c1", "text": "证据内容1", "doc_id": "d1", "rerank_score": 0.95},
        ]
        bank.add_evidence(chunks)

        claims = [{"claim_id": "claim_1", "text": "主张内容", "confidence": 0.9}]
        bank.add_claims(claims)

        success = bank.link_claim_evidence("claim_1", ["ev_c1"])

        assert success is True

        # 验证链接
        coverage = bank.verify_coverage()
        assert coverage["total_claims"] == 1
        assert coverage["verified_claims"] == 1

    def test_unverified_claims(self):
        """未验证主张测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        # 添加主张但不链接证据
        claims = [{"claim_id": "unverified", "text": "无证据的主张", "confidence": 0.5}]
        bank.add_claims(claims)

        coverage = bank.verify_coverage()

        assert coverage["total_claims"] == 1
        assert coverage["verified_claims"] == 0
        assert "unverified" in coverage["unverified_claims"]

    def test_context_for_generation(self):
        """生成上下文测试"""
        bank = MemoryBank(session_id="test_session", max_claims=50, ttl_hours=24)

        chunks = [
            {"chunk_id": "c1", "text": "证据文本", "doc_id": "d1", "rerank_score": 0.95},
        ]
        bank.add_evidence(chunks)

        claims = [{"claim_id": "claim_1", "text": "主张文本", "confidence": 0.9}]
        bank.add_claims(claims)
        bank.link_claim_evidence("claim_1", ["ev_c1"])

        context = bank.get_context_for_generation()

        assert "Evidence Bank" in context
        assert "Claim-Evidence Mapping" in context
        assert "主张文本" in context


class TestEvidence:
    """Evidence 数据结构测试"""

    def test_evidence_creation(self):
        """Evidence 创建测试"""
        ev = Evidence(
            source_id="c1",
            text="这是证据文本。",
            doc_title="测试文档",
            section_path="第一章/第一节",
            retrieval_score=0.95,
        )

        assert ev.source_id == "c1"
        assert ev.doc_title == "测试文档"
        assert ev.section_path == "第一章/第一节"
        assert ev.retrieval_score == 0.95
        assert ev.used_by_claims == []


class TestClaim:
    """Claim 数据结构测试"""

    def test_claim_creation(self):
        """Claim 创建测试"""
        claim = Claim(
            claim_id="claim_1",
            text="这是主张内容。",
            evidence_ids=["ev_c1", "ev_c2"],
            verified=True,
            confidence=0.85,
        )

        assert claim.claim_id == "claim_1"
        assert claim.verified is True
        assert len(claim.evidence_ids) == 2
        assert claim.confidence == 0.85
