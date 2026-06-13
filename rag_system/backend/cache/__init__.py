"""
cache 模块 — 语义缓存
"""

from backend.cache.semantic_cache import SemanticCache, RedisSemanticCache

__all__ = ["SemanticCache", "RedisSemanticCache"]
