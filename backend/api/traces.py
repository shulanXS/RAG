"""
traces.py — Trace Viewer API (P2.2)
================================================================================
技术决策:
- 简化版：从内存 ring buffer 读取最近 trace。
- 列出 + 详情两个端点，足够 demo。
- 生产环境应替换为从 Jaeger / Tempo 查询（OtelExporter API）。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.observability.trace_store import get_trace_store
from backend.security.auth import require_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("")
async def list_traces(
    limit: int = Query(default=100, ge=1, le=1000),
    complexity: str | None = Query(default=None, description="simple|moderate|complex|beyond_kb"),
    token_payload: dict = Depends(require_current_user),
) -> dict[str, Any]:
    """列出最近 N 条 trace 摘要（不包含完整 spans，节省带宽）"""
    store = get_trace_store()
    items = store.list_recent(limit=limit, complexity=complexity)
    # 列表页只需 summary
    summaries = [
        {
            "trace_id": t.get("trace_id"),
            "started_at_ms": t.get("started_at_ms"),
            "ended_at_ms": t.get("ended_at_ms"),
            "latency_ms": t.get("latency_ms"),
            "complexity": t.get("complexity"),
            "routing_confidence": t.get("routing_confidence"),
            "cache_hit": t.get("cache_hit"),
            "answer_length": t.get("answer_length"),
            "span_count": len(t.get("spans", [])),
        }
        for t in items
    ]
    return {"traces": summaries, "total": len(summaries)}


@router.get("/{trace_id}")
async def get_trace(
    trace_id: str,
    token_payload: dict = Depends(require_current_user),
) -> dict[str, Any]:
    """获取单条 trace 详情（包含完整 spans）"""
    store = get_trace_store()
    trace = store.get_by_id(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace
