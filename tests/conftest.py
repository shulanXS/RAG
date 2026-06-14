"""
conftest.py — pytest 共享 fixtures
================================================================================
技术决策:
- 所有 LLM/Embedder mock 在 fixture 级别统一定义，避免每个测试文件重复 mock setup。
- 提供 dummy config fixture，确保 backend.config.get_config() 不会真正初始化外部服务。
- 提供 temp paths fixture 用于文档/缓存文件操作。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# 将项目根加入 sys.path，确保 backend.* 导入在 pytest 下也能工作
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------
# 1. 假 Config fixture（避免真实加载外部服务）
# --------------------------------------------------------------------------


@pytest.fixture
def dummy_config(monkeypatch):
    """提供最小化的 AppConfig 实例，避免 config.yaml 加载副作用"""
    from backend.config import (
        AppConfig,
        ChunkingConfig,
        EmbeddingConfig,
        EvalCIConfig,
        EvaluationConfig,
        EvalTestSet,
        EvalThresholds,
        HybridSearchConfig,
        LLMConfig,
        LLMGeneratorConfig,
        LLMRouterConfig,
        LoggingConfig,
        PathsConfig,
        RerankerConfig,
        SemanticCacheConfig,
        VectorDBConfig,
    )

    cfg = AppConfig(
        paths=PathsConfig(),
        embedding=EmbeddingConfig(backend="voyage", dimension=1024),
        chunking=ChunkingConfig(),
        vector_db=VectorDBConfig(),
        reranker=RerankerConfig(),
        llm=LLMConfig(
            generator=LLMGeneratorConfig(provider="anthropic", model="claude-3-7-sonnet-20250620"),
            router=LLMRouterConfig(provider="anthropic", model="claude-3-5-haiku-20250620"),
        ),
        hybrid_search=HybridSearchConfig(),
        cache=SemanticCacheConfig(),
        evaluation=EvaluationConfig(
            thresholds=EvalThresholds(),
            test_set=EvalTestSet(),
            ci=EvalCIConfig(),
        ),
        logging=LoggingConfig(),
    )

    from backend.config import ConfigLoader
    monkeypatch.setattr(ConfigLoader, "_instance", cfg)
    return cfg


# --------------------------------------------------------------------------
# 2. LLM mock fixture
# --------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client():
    """统一的 LLM mock fixture

    返回的 mock 同时提供:
    - generate_async(prompt) -> str
    - generate_stream_async(prompt) -> async iterator of str
    - generator_client 子 mock
    - router_client 子 mock
    """
    client = MagicMock()

    # 异步生成
    client.generate_async = AsyncMock(return_value="Mocked LLM response")

    # 同步生成
    client.generate = MagicMock(return_value="Mocked LLM response")

    # 流式生成（async iterator）
    async def _stream(_prompt, **_kwargs):
        for tok in ["Mocked ", "stream ", "response"]:
            yield tok

    client.generate_stream_async = _stream

    # 嵌套 mock
    client.generator_client = MagicMock()
    client.generator_client.generate_async = AsyncMock(return_value="Mocked generator response")
    client.generator_model = "claude-3-7-sonnet-20250620"
    client.router_model = "claude-3-5-haiku-20250620"

    return client


@pytest.fixture
def mock_circuit_breaker():
    """Mock 熔断器，让 LLM 调用总是通过"""
    from backend.middleware.circuit_breaker import CircuitBreakerConfig, CircuitState

    breaker = MagicMock()
    breaker.state = CircuitState.CLOSED
    breaker.call_async = AsyncMock(side_effect=lambda fn: fn())
    breaker.call = MagicMock(side_effect=lambda fn, *a, **k: fn(*a, **k))
    breaker.config = CircuitBreakerConfig()
    return breaker


# --------------------------------------------------------------------------
# 3. Embedder mock fixture
# --------------------------------------------------------------------------


@pytest.fixture
def mock_embedder():
    """统一 embedder mock"""
    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=[0.1] * 1024)
    embedder.embed_batch = MagicMock(return_value=[[0.1] * 1024])
    return embedder


# --------------------------------------------------------------------------
# 4. 临时路径 fixture
# --------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path):
    """为文档/缓存测试提供隔离的临时目录"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# --------------------------------------------------------------------------
# 5. 测试样本 fixture
# --------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[dict]:
    """测试用的样本 chunks"""
    return [
        {
            "chunk_id": "chunk_1",
            "doc_id": "doc_a",
            "text": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation.",
            "section_path": "Introduction",
            "rerank_score": 0.92,
            "rrf_score": 0.85,
            "metadata": {"chunk_index": 0, "token_count": 25},
        },
        {
            "chunk_id": "chunk_2",
            "doc_id": "doc_a",
            "text": "Vector databases like Qdrant store dense embeddings for similarity search.",
            "section_path": "Architecture",
            "rerank_score": 0.85,
            "rrf_score": 0.78,
            "metadata": {"chunk_index": 1, "token_count": 20},
        },
        {
            "chunk_id": "chunk_3",
            "doc_id": "doc_b",
            "text": "Hybrid search combines BM25 and dense retrieval for better recall.",
            "section_path": "Retrieval",
            "rerank_score": 0.78,
            "rrf_score": 0.72,
            "metadata": {"chunk_index": 0, "token_count": 18},
        },
    ]


@pytest.fixture
def sample_test_case() -> dict:
    """单条评估测试用例"""
    return {
        "question": "什么是 RAG？",
        "answer": "RAG 是一种结合检索和生成的技术。",
        "contexts": [
            "RAG (Retrieval-Augmented Generation) 结合了信息检索与文本生成。",
            "向量数据库存储用于相似度搜索的稠密向量。",
        ],
        "ground_truth": "RAG 是结合检索与生成的技术。",
    }
