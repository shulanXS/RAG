"""
chat.py — 对话相关 API 路由
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    history: list[dict] | None = None


class SourceRef(BaseModel):
    doc_id: str
    chunk_id: str | None = None
    title: str | None = None
    content: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    session_id: str | None = None
    confidence: str | None = None
    latency_ms: float | None = None


class StreamChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None


# 全局编排器实例（懒加载）
_orchestrator = None


def get_orchestrator():
    """获取或初始化编排器实例"""
    global _orchestrator
    if _orchestrator is None:
        import asyncio
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


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    发送对话请求，返回 RAG 生成的答案。

    支持:
    - 多轮对话上下文（通过 history 字段）
    - 会话保持（通过 session_id）
    - 语义缓存自动命中
    """
    start = time.perf_counter()

    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.run(
            query=request.query,
            conversation_history=request.history,
        )

        sources = [
            SourceRef(
                doc_id=cit.get("doc_id", ""),
                chunk_id=cit.get("chunk_id"),
                content=cit.get("quote"),
                score=cit.get("score"),
            )
            for cit in result.citations
        ]

        return ChatResponse(
            answer=result.answer,
            sources=sources,
            session_id=request.session_id,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
        )
    except Exception as e:
        logger.exception(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
