"""
search.py — 检索相关 API 路由
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict | None = None


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


_search_engine = None


def get_search_engine():
    """获取或初始化搜索引擎"""
    global _search_engine
    if _search_engine is None:
        from backend.config import get_config
        from backend.ingestion.embedder import Embedder
        from backend.retrieval.hybrid_search import HybridSearchEngine

        config = get_config()
        embedder = Embedder(backend=config.embedding.backend)
        _search_engine = HybridSearchEngine.from_config(config, embedder)

    return _search_engine


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    执行混合检索查询，返回相关文档片段。

    支持:
    - BM25 + Dense 混合检索
    - Cross-Encoder 重排序
    - 可选的 ACL 过滤
    """
    try:
        engine = get_search_engine()
        chunks, context = await engine.search(
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

        return SearchResponse(
            results=results,
            total=len(results),
            query=request.query,
        )
    except Exception as e:
        logger.exception(f"Search request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=SearchResponse)
async def search_get(
    query: Annotated[str, Query(min_length=1, max_length=1000)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 5,
) -> SearchResponse:
    """GET 方式的检索接口（便于调试）"""
    return await search(SearchRequest(query=query, top_k=top_k))
