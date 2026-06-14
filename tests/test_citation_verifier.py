"""
test_citation_verifier.py — P3.3
================================================================================
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.generation.citation_verifier import CitationVerifier


@pytest.mark.asyncio
async def test_verify_no_llm_returns_fallback():
    cv = CitationVerifier(llm_client=None, fallback_citations=[{"doc_id": "d1"}])
    chunks = [{"doc_id": "d1", "chunk_id": "c1", "text": "x"}]
    result = await cv.verify("q", "answer", chunks)
    # fallback 应被使用
    assert result.verified_claims or result.unverified_claims
    # score 应当存在
    assert hasattr(result, "groundedness_score")


@pytest.mark.asyncio
async def test_verify_llm_returns_valid_mapping():
    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(return_value=json.dumps({
        "claims": [
            {"claim": "RAG combines retrieval", "chunk_id": "c1", "supported": True, "confidence": 0.95},
            {"claim": "Qdrant is a database", "chunk_id": "c2", "supported": True, "confidence": 0.90},
            {"claim": "RAG was invented in 2026", "chunk_id": "c1", "supported": False, "confidence": 0.85},
        ],
        "overall_groundedness": 0.8,
    }))

    cv = CitationVerifier(llm_client=mock_llm)
    chunks = [
        {"doc_id": "d1", "chunk_id": "c1", "text": "RAG combines retrieval and generation."},
        {"doc_id": "d1", "chunk_id": "c2", "text": "Qdrant is a vector database."},
    ]
    result = await cv.verify("什么是 RAG", "RAG combines retrieval. Qdrant is a database. RAG was invented in 2026.", chunks)
    # 至少应有 1 个被支撑 + 1 个不被支撑
    assert len(result.verified_claims) >= 1
    assert len(result.unverified_claims) >= 1
    assert 0.0 <= result.groundedness_score <= 1.0


@pytest.mark.asyncio
async def test_verify_llm_returns_garbage_handles_gracefully():
    mock_llm = MagicMock()
    mock_llm.generate_async = AsyncMock(return_value="not json at all")
    cv = CitationVerifier(llm_client=mock_llm)
    chunks = [{"doc_id": "d1", "chunk_id": "c1", "text": "x"}]
    # 不应抛错
    result = await cv.verify("q", "a", chunks)
    assert result is not None


def test_format_chunks_contains_marker():
    cv = CitationVerifier(llm_client=None)
    chunks = [{"doc_id": "d1", "chunk_id": "c1", "text": "hello world"}]
    formatted = cv._format_chunks(chunks)
    assert "[c1]" in formatted
    assert "hello world" in formatted
