"""
search.py — 检索相关 API 路由
================================================================================
依赖注入: 通过 FastAPI Depends 注入 HybridSearchEngine 和 AgenticOrchestrator。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api import deps

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


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_engine=Depends(deps.get_hybrid_search),
    orchestrator=Depends(deps.get_orchestrator),
) -> SearchResponse:
    """
    检索接口。

    - raw=True: 仅返回检索结果（调试模式）
    - raw=False: 走编排器，返回答案 + 引用
    """
    try:
        if request.raw:
            chunks, _ = await search_engine.search(
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
            result = await orchestrator.run(
                query=request.query,
                conversation_history=None,
            )
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
