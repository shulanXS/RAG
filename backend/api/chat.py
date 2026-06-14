"""
chat.py — 对话相关 API 路由
================================================================================
技术决策记录:
- 依赖注入: 使用 FastAPI Depends 而非全局单例，每个端点独立声明依赖。
  这使得单元测试时可以注入 mock 实例，而无需 patch global 变量。
- 路由级依赖: get_orchestrator 和 get_chat_store 通过 lru_cache 单例化，
  等效于全局单例但不污染全局命名空间。
"""

from __future__ import annotations

import logging
import time
from typing import TypedDict

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api import deps
from backend.security.auth import require_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# --------------------------------------------------------------------------
# TypedDicts (P3.4: 强类型约束 orchestrator.citations 字段)
# --------------------------------------------------------------------------


class CitationDict(TypedDict, total=False):
    """Orchestrator 返回的 citation 结构约束。

    之前 `cit.get("doc_id", "")` 这种隐式 dict 访问现在用 TypedDict 强类型化。
    """
    doc_id: str
    chunk_id: str
    quote: str
    score: float
    section_path: str
    rerank_score: float


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

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


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]
    total: int


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token_payload: dict[str, object] = Depends(require_current_user),
    orchestrator: object = Depends(deps.get_orchestrator),
) -> ChatResponse:
    """
    发送对话请求，返回 RAG 生成的答案。

    支持:
    - 多轮对话上下文（通过 history 字段）
    - 会话保持（通过 session_id）
    - 语义缓存自动命中
    - 多租户隔离（从 JWT 提取 tenant_id 自动注入检索 filter）
    """
    start = time.perf_counter()

    # 提取租户上下文（从 JWT payload 派生），确保跨租户数据不可见
    from backend.security.tenant import TenantContext
    tenant = TenantContext.from_token(token_payload)

    try:
        result = await orchestrator.run(
            query=request.query,
            conversation_history=request.history,
            session_id=request.session_id,
            tenant=tenant,
        )

        sources: list[SourceRef] = [
            SourceRef(
                doc_id=str(cit.get("doc_id", "")),
                chunk_id=cit.get("chunk_id"),
                content=cit.get("quote") if isinstance(cit, dict) else None,
                score=cit.get("score") if isinstance(cit, dict) and isinstance(cit.get("score"), (int, float)) else None,
            )
            for cit in result.citations
            if isinstance(cit, dict)
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


@router.get("/history", response_model=ChatHistoryResponse)
async def get_history(
    session_id: str = Query(..., description="会话 ID"),
    limit: int = Query(default=20, ge=1, le=100),
    token_payload: dict = Depends(require_current_user),
    chat_store=Depends(deps.get_chat_store),
) -> ChatHistoryResponse:
    """获取指定会话的聊天历史"""
    try:
        # P0-4: chat_store 改 async
        messages = await chat_store.get_history(session_id, limit=limit)
        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages,
            total=len(messages),
        )
    except Exception as e:
        logger.exception(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def delete_history(
    session_id: str = Query(..., description="会话 ID"),
    token_payload: dict = Depends(require_current_user),
    chat_store=Depends(deps.get_chat_store),
) -> dict:
    """清空指定会话的历史"""
    try:
        # P0-4: chat_store 改 async
        await chat_store.clear_session(session_id)
        return {"success": True, "message": f"Session {session_id} cleared"}
    except Exception as e:
        logger.exception(f"Failed to clear chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
