"""
deps.py — FastAPI 依赖注入工厂
================================================================================
技术决策记录:
- FastAPI 最佳实践: 所有可复用实例通过 Depends 注入，而非全局单例。
  这使得每个请求可独立测试、mock 和隔离。
- lru_cache 确保每个进程只初始化一次实例，与单例效果等价。
- 依赖链清晰: embedder -> hybrid_search -> orchestrator，
  符合构造函数依赖顺序。
- 懒加载: 首次请求时才初始化，避免启动时因外部服务不可用而卡死。
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Depends

    from backend.ingestion.embedder import Embedder
    from backend.retrieval.hybrid_search import HybridSearchEngine
    from backend.agentic.orchestrator import AgenticOrchestrator
    from backend.generation.llm_client import LLMClient
    from backend.agentic.query_router import QueryRouter
    from backend.session.chat_store import ChatStore


@lru_cache(maxsize=1)
def get_embedder() -> "Embedder":
    """获取 Embedder 单例（每个进程一个实例）"""
    from backend.config import get_config
    from backend.ingestion.embedder import Embedder

    cfg = get_config()
    return Embedder(backend=cfg.embedding.backend)


@lru_cache(maxsize=1)
def get_llm_client() -> "LLMClient":
    """获取 LLMClient 单例（每个进程一个实例，含 Router + Generator 分离）"""
    from backend.config import get_config
    from backend.generation.llm_client import LLMClient

    cfg = get_config()
    return LLMClient(
        generator_provider=cfg.llm.generator.provider,
        generator_model=cfg.llm.generator.model,
        router_provider=cfg.llm.router.provider,
        router_model=cfg.llm.router.model,
        generator_api_key=cfg.llm.deepseek.api_key or None,
        generator_base_url=cfg.llm.deepseek.base_url,
        router_api_key=cfg.llm.deepseek.api_key or None,
        router_base_url=cfg.llm.deepseek.base_url,
    )


@lru_cache(maxsize=1)
def get_hybrid_search() -> "HybridSearchEngine":
    """获取混合检索引擎单例"""
    from backend.config import get_config
    from backend.ingestion.embedder import Embedder
    from backend.retrieval.hybrid_search import HybridSearchEngine

    cfg = get_config()
    embedder: Embedder = get_embedder()
    return HybridSearchEngine.from_config(cfg, embedder)


@lru_cache(maxsize=1)
def get_orchestrator() -> "AgenticOrchestrator":
    """获取编排器单例（自动注入所有依赖）"""
    from backend.agentic.query_router import QueryRouter
    from backend.agentic.orchestrator import AgenticOrchestrator
    from backend.session.chat_store import ChatStore

    hs: "HybridSearchEngine" = get_hybrid_search()
    llm: "LLMClient" = get_llm_client()
    router = QueryRouter(llm_client=llm.router_client)
    chat_store: "ChatStore" = get_chat_store()
    semantic_cache = get_semantic_cache()

    # P1.2: 把 SemanticCache 包装成 orchestrator 期望的 callable 接口
    # (get) -> dict | None / (set) -> None
    async def semantic_cache_fn(query: str, response: dict | None = None):
        if semantic_cache is None:
            return None
        if response is None:
            # 读：get(query, query_embedding)
            try:
                query_embedding = get_embedder().embed(query)
                return await semantic_cache.get(query, query_embedding)
            except Exception:
                return None
        else:
            # 写：set(query, query_embedding, response)
            try:
                query_embedding = get_embedder().embed(query)
                await semantic_cache.set(query, query_embedding, response)
            except Exception:
                pass

    return AgenticOrchestrator(
        hybrid_search_engine=hs,
        router=router,
        llm_client=llm,
        chat_store=chat_store,
        semantic_cache_fn=semantic_cache_fn,
    )


@lru_cache(maxsize=1)
def get_chat_store() -> "ChatStore":
    """获取 Chat Store 单例"""
    from backend.session.chat_store import ChatStore

    return ChatStore()


@lru_cache(maxsize=1)
def get_semantic_cache():
    """
    获取 Semantic Cache 单例（懒加载）。

    P1.2: 之前 orchestrator 的 `semantic_cache_fn` 参数接受 callable 但没人注入，
    实际 cache_hit 永远为 False。本依赖工厂补齐这个断点。

    Returns:
        RedisSemanticCache 实例（若 redisvl/Redis 不可用则返回 None，
        调用方在 cache_fn 内做 graceful degradation）。
    """
    from backend.config import get_config
    from backend.cache.semantic_cache import RedisSemanticCache

    try:
        cfg = get_config()
        return RedisSemanticCache(
            host=cfg.semantic_cache.host,
            port=cfg.semantic_cache.port,
            similarity_threshold=cfg.semantic_cache.similarity_threshold,
            embedding_dim=cfg.embedding.dimension,
        )
    except Exception as e:
        # 没有 redisvl/redis 也不阻塞服务启动
        import logging
        logging.getLogger(__name__).warning(f"semantic cache 初始化失败（已降级到 no-op）: {e}")
        return None


# --------------------------------------------------------------------------
# 2. 请求级依赖
# --------------------------------------------------------------------------


def get_tenant_from_token(token_payload: dict | None = None) -> "TenantContext":
    """从 JWT 构造租户上下文

    用法:
        @router.post(...)
        async def endpoint(
            request: RequestSchema,
            token_payload: dict = Depends(require_current_user),
            tenant: TenantContext = Depends(get_tenant_from_token),
        ):
            ...

    与 require_current_user 配合时需手动传 token_payload；
    也可以用 Depends(get_current_user) 让 None 自然传播。
    """
    from backend.security.tenant import TenantContext

    return TenantContext.from_token(token_payload)
