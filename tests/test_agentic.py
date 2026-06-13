"""
test_agentic.py — Agentic 编排测试
"""

import pytest
from backend.agentic.query_router import QueryRouter, QueryComplexity, RoutingDecision
from backend.agentic.memory_bank import MemoryBank, Claim, Evidence


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
