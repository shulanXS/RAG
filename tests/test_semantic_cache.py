"""
test_semantic_cache.py — P3.1 补充测试
================================================================================
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# redisvl 是可选依赖 (CI 环境会装, 离线开发环境可缺)
redisvl = pytest.importorskip("redisvl")


@pytest.mark.asyncio
async def test_semantic_cache_get_miss():
    from backend.domain.cache.semantic_cache import RedisSemanticCache

    cache = RedisSemanticCache(
        host="localhost",
        port=6379,
        similarity_threshold=0.85,
        embedding_dim=4,
    )

    mock_index = MagicMock()
    mock_index.search = AsyncMock(return_value=[])
    cache._index = mock_index
    cache._ensure_index = MagicMock(return_value=mock_index)

    result = await cache.get("hello", [0.1, 0.2, 0.3, 0.4])
    assert result is None
    assert cache._misses == 1
    assert cache._total_requests == 1


@pytest.mark.asyncio
async def test_semantic_cache_get_hit():
    from backend.domain.cache.semantic_cache import RedisSemanticCache

    cache = RedisSemanticCache(
        host="localhost",
        port=6379,
        similarity_threshold=0.85,
        embedding_dim=4,
    )

    mock_index = MagicMock()
    mock_index.search = AsyncMock(return_value=[
        {
            "vector_distance": 0.05,  # 1-0.05 = 0.95 >= 0.85
            "response_json": '{"answer": "hi"}',
        }
    ])
    cache._index = mock_index
    cache._ensure_index = MagicMock(return_value=mock_index)

    result = await cache.get("hello", [0.1, 0.2, 0.3, 0.4])
    assert result is not None
    assert result["answer"] == "hi"
    assert cache._hits == 1


@pytest.mark.asyncio
async def test_semantic_cache_normalization():
    """query embedding 应在 KNN 搜索前归一化"""
    from backend.domain.cache.semantic_cache import RedisSemanticCache

    cache = RedisSemanticCache(host="localhost", port=6379, embedding_dim=4)
    captured: dict = {}

    async def _capture(qv, **_):
        captured["qv"] = qv
        return []

    mock_index = MagicMock()
    mock_index.search = _capture
    cache._index = mock_index
    cache._ensure_index = MagicMock(return_value=mock_index)

    await cache.get("test", [3.0, 4.0, 0.0, 0.0])
    # [3,4,0,0] norm=5, normalized = [0.6, 0.8, 0, 0]
    import math
    norm = math.sqrt(sum(x * x for x in captured["qv"]))
    assert abs(norm - 1.0) < 1e-5
