"""
test_semantic_cache_tenant.py — Tenant 隔离单元测试(plan §4.3)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.domain.cache.semantic_cache import REDISVL_AVAILABLE, RedisSemanticCache


@pytest.mark.skipif(not REDISVL_AVAILABLE, reason="redisvl not installed")
def test_tenant_id_isolates_index_name():
    """不同 tenant 生成的 Redis index name 必须不同,避免 embedding 串台。"""
    cache_a = RedisSemanticCache(tenant_id="tenant_a")
    cache_b = RedisSemanticCache(tenant_id="tenant_b")
    assert cache_a._index_name != cache_b._index_name
    assert "tenanta" in cache_a._index_name.lower() or "tenant_a" in cache_a._index_name
    assert "tenantb" in cache_b._index_name.lower() or "tenant_b" in cache_b._index_name


@pytest.mark.skipif(not REDISVL_AVAILABLE, reason="redisvl not installed")
def test_default_tenant_uses_default_suffix():
    cache = RedisSemanticCache()
    assert "default" in cache._index_name


@pytest.mark.skipif(not REDISVL_AVAILABLE, reason="redisvl not installed")
def test_tenant_id_sanitized_against_injection():
    """tenant_id 含特殊字符应被清洗,避免 Redis key 注入。"""
    cache = RedisSemanticCache(tenant_id="evil tenant; FLUSHALL")
    # 空格/分号/空字符应被过滤
    assert " " not in cache._index_name
    assert ";" not in cache._index_name


@pytest.mark.skipif(not REDISVL_AVAILABLE, reason="redisvl not installed")
def test_empty_tenant_falls_back_to_default():
    cache = RedisSemanticCache(tenant_id="")
    assert "default" in cache._index_name
