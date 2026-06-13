"""
search.py — 检索相关 API 路由
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict | None = None
    raw: bool = Field(default=False, description="调试模式：仅返回检索结果，不走生成")


class SearchResult(BaseModel):
    id: str
    title: str | None = None
    snippet: str
    score: float | None = None
    metadata: dict | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    total: int = 0
    query: str | None = None
    answer: str | None = None


_search_engine = None
_orchestrator = None


def get_search_engine():
    """获取或初始化搜索引擎（用于 raw 调试模式）"""
    global _search_engine
    if _search_engine is None:
        from backend.config import get_config
        from backend.ingestion.embedder import Embedder
        from backend.retrieval.hybrid_search import HybridSearchEngine

        config = get_config()
        embedder = Embedder(backend=config.embedding.backend)
        _search_engine = HybridSearchEngine.from_config(config, embedder)

    return _search_engine


def get_orchestrator():
    """获取或初始化编排器实例"""
    global _orchestrator
    if _orchestrator is None:
        from backend.config import get_config
        from backend.ingestion.embedder import Embedder
        from backend.retrieval.hybrid_search import HybridSearchEngine
        from backend.agentic import QueryRouter, AgenticOrchestrator
        from backend.generation import LLMClient

        config = get_config()
        embedder = Embedder(backend=config.embedding.backend)
        hybrid_search = HybridSearchEngine.from_config(config, embedder)
        llm_client = LLMClient(
            generator_provider=config.llm.generator.provider,
            generator_model=config.llm.generator.model,
            router_provider=config.llm.router.provider,
            router_model=config.llm.router.model,
        )
        router_q = QueryRouter(llm_client=llm_client.router_client)
        _orchestrator = AgenticOrchestrator(
            hybrid_search_engine=hybrid_search,
            router=router_q,
            llm_client=llm_client,
        )

    return _orchestrator


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    检索接口。

    - raw=True: 仅返回检索结果（调试模式）
    - raw=False: 走编排器，返回答案 + 引用
    """
    try:
        if request.raw:
            engine = get_search_engine()
            chunks, _ = await engine.search(
                query=request.query,
                acl_filter=request.filters,
            )
            results = [
                SearchResult(
                    id=chunk.get("chunk_id", ""),
                    title=chunk.get("section_path", ""),
                    snippet=chunk.get("text", "")[:300],
                    score=chunk.get("rerank_score", chunk.get("rrf_score")),
                    metadata=chunk.get("metadata"),
                )
                for chunk in chunks[: request.top_k]
            ]
            return SearchResponse(results=results, total=len(results), query=request.query)
        else:
            orch = get_orchestrator()
            result = await orch.run(query=request.query, conversation_history=None)
            results = [
                SearchResult(
                    id=cit.get("chunk_id", ""),
                    snippet=cit.get("quote", ""),
                    score=cit.get("score"),
                )
                for cit in result.citations[: request.top_k]
            ]
            return SearchResponse(
                results=results,
                total=len(results),
                query=request.query,
                answer=result.answer,
            )
    except Exception as e:
        logger.exception(f"Search request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
