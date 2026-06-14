"""
test_reranker.py — P3.3
================================================================================
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.retrieval.reranker import BGEReranker, RerankResult


def _make_bge_with_mock_scores(scores: list[float]) -> BGEReranker:
    """构造 BGEReranker 但 mock 掉 cross_encoder"""
    from backend.retrieval import reranker as r

    # 跳过真实 __init__，直接构造实例
    rr = BGEReranker.__new__(BGEReranker)
    rr._model_name = "mock-bge"
    rr._device = "cpu"
    rr._use_fp16 = False
    ce = MagicMock()
    ce.predict = MagicMock(return_value=MagicMock(tolist=lambda: scores))
    rr._cross_encoder = ce
    return rr


def test_rerank_truncates_to_top_k():
    chunks = [
        {"chunk_id": f"c{i}", "doc_id": "d1", "text": f"text {i}"}
        for i in range(10)
    ]
    # 倒序打分: c9 最高, c0 最低
    scores = [0.1 * i for i in range(10)]
    rr = _make_bge_with_mock_scores(scores)
    results = rr.rerank("query", chunks, top_k=3)
    assert len(results) == 3
    assert results[0].chunk_id == "c9"  # 最高分
    assert results[1].chunk_id == "c8"
    assert results[2].chunk_id == "c7"
    assert all(isinstance(r, RerankResult) for r in results)


def test_rerank_empty_chunks():
    rr = _make_bge_with_mock_scores([])
    assert rr.rerank("q", [], top_k=5) == []


def test_rerank_preserves_text_and_doc_id():
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "hello world", "section_path": "intro"},
    ]
    rr = _make_bge_with_mock_scores([0.9])
    results = rr.rerank("q", chunks, top_k=5)
    assert results[0].doc_id == "d1"
    assert results[0].text == "hello world"
    assert results[0].rerank_score == 0.9
    assert results[0].final_rank == 1


def test_rerank_text_truncated_to_2000_chars():
    """CrossEncoder 输入文本超过 2000 字符应被截断"""
    long_text = "a" * 5000
    chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": long_text}]
    rr = _make_bge_with_mock_scores([0.5])

    rr.rerank("q", chunks, top_k=5)

    # 验证传给 predict 的 pair 第二个元素是截断后的
    call_args = rr._cross_encoder.predict.call_args
    pairs = call_args[0][0]
    assert len(pairs[0][1]) == 2000
