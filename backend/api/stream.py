"""
backend/api/stream.py — SSE 流式聊天接口
================================================================================
依赖注入: orchestrator 通过 FastAPI Depends 注入。
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api import deps
from backend.security.auth import require_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stream", tags=["stream"])


# --------------------------------------------------------------------------
# Request model
# --------------------------------------------------------------------------

class StreamRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    history: list[dict] | None = None


# --------------------------------------------------------------------------
# SSE event helpers
# --------------------------------------------------------------------------

def sse_event(data: dict) -> str:
    """将 dict 序列化为 SSE 事件行"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_generator(
    query: str,
    session_id: str | None,
    history: list[dict] | None,
    orchestrator,
    tenant,
) -> AsyncIterator[str]:
    """
    流式生成器：从 orchestrator.run_stream yield SSE 事件
    """
    try:
        async for event in orchestrator.run_stream(
            query=query,
            conversation_history=history,
            session_id=session_id,
            tenant=tenant,
        ):
            yield sse_event(event)
    except Exception as e:
        logger.exception(f"Stream failed: {e}")
        yield sse_event({"stage": "error", "message": str(e)})

    yield sse_event({"stage": "done"})


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------

@router.post("")
async def stream_chat(
    request: StreamRequest,
    token_payload: dict = Depends(require_current_user),
    orchestrator=Depends(deps.get_orchestrator),
) -> StreamingResponse:
    """
    SSE 流式聊天接口。

    Event stream 格式（每条 data 为 JSON）：
    - {"stage": "cache_check", "cache_hit": true/false}
    - {"stage": "rewrite", "query": "...", "was_rewritten": true/false}
    - {"stage": "routing", "complexity": "simple|moderate|complex"}
    - {"stage": "retrieval", "num_chunks": N, "latency_ms": M}
    - {"stage": "generating", "token": "...", "done": false}
    - {"stage": "done", "answer": "...", "citations": [...], "session_id": "..."}
    - {"stage": "error", "message": "..."}
    """
    from backend.domain.tenant import TenantContext
    tenant = TenantContext.from_token(token_payload)

    return StreamingResponse(
        stream_generator(
            query=request.query,
            session_id=request.session_id,
            history=request.history,
            orchestrator=orchestrator,
            tenant=tenant,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
