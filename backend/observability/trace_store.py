"""
trace_store.py — 内存 ring buffer 存储最近 N 条 trace 摘要（P2.2）
================================================================================
技术决策:
- 内存版：避免接 Jaeger / Tempo 等后端的运维负担
  （作品集 demo 阶段完全够用；生产再切 OTLP + Jaeger）。
- ring buffer 大小 1000：保留最近 1000 条请求 trace，
  可调配置 `max_traces`。
- 持久化为可选项：True 时把 trace 写入 SQLite，便于复盘
  （"那 5 分钟前的诡异 query 长什么样？"）。
- 不在 OTEL span 层加 hook：因为 create_span 嵌套过深不便。
  改成在 orchestrator.run() 末尾主动调 record()，结构化存储。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class TraceStore:
    """
    最近 N 条 trace 摘要的 ring buffer（带可选 SQLite 持久化）。

    Schema:
    {
        "trace_id": "abc123",
        "started_at_ms": 1718000000000,
        "ended_at_ms": 1718000001500,
        "latency_ms": 1500.0,
        "complexity": "simple",
        "routing_confidence": 0.9,
        "cache_hit": false,
        "answer_length": 250,
        "spans": [
            {"name": "rag.cache_lookup", "duration_ms": 5.0, "attrs": {...}},
            {"name": "rag.query_rewrite", "duration_ms": 80.0, "attrs": {...}},
            ...
        ],
    }
    """

    def __init__(self, max_traces: int = 1000, persist_to_sqlite: bool = False, db_path: str | None = None):
        self._max = max_traces
        self._buf: deque[dict] = deque(maxlen=max_traces)
        self._lock = threading.RLock()
        self._persist = persist_to_sqlite
        self._db_path: Path | None = None
        if persist_to_sqlite:
            if db_path is None:
                db_path = str(Path(__file__).parent.parent.parent / "data" / "traces.db")
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_db(self) -> None:
        assert self._db_path is not None
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    started_at_ms INTEGER,
                    ended_at_ms INTEGER,
                    latency_ms REAL,
                    complexity TEXT,
                    routing_confidence REAL,
                    cache_hit INTEGER,
                    answer_length INTEGER,
                    payload_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_started_at ON traces(started_at_ms)")

    def record(self, trace: dict) -> None:
        """记录一条 trace（ring buffer + 可选持久化）"""
        with self._lock:
            self._buf.append(trace)
            if self._persist and self._db_path:
                try:
                    with sqlite3.connect(self._db_path) as conn:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO traces
                            (trace_id, started_at_ms, ended_at_ms, latency_ms,
                             complexity, routing_confidence, cache_hit, answer_length, payload_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                trace.get("trace_id", ""),
                                trace.get("started_at_ms", 0),
                                trace.get("ended_at_ms", 0),
                                trace.get("latency_ms", 0.0),
                                trace.get("complexity", ""),
                                trace.get("routing_confidence", 0.0),
                                1 if trace.get("cache_hit") else 0,
                                trace.get("answer_length", 0),
                                json.dumps(trace, ensure_ascii=False),
                            ),
                        )
                except Exception as e:
                    logger.debug(f"trace 持久化失败（已忽略）: {e}")

    def list_recent(self, limit: int = 100, complexity: str | None = None) -> list[dict]:
        with self._lock:
            items = list(self._buf)
        items.reverse()  # 最新的在前
        if complexity:
            items = [t for t in items if t.get("complexity") == complexity]
        return items[:limit]

    def get_by_id(self, trace_id: str) -> dict | None:
        with self._lock:
            for t in reversed(self._buf):
                if t.get("trace_id") == trace_id:
                    return t
        # 持久化兜底
        if self._persist and self._db_path:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    cur = conn.execute("SELECT payload_json FROM traces WHERE trace_id = ?", (trace_id,))
                    row = cur.fetchone()
                    if row:
                        return json.loads(row[0])
            except Exception:
                pass
        return None

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


# 模块级单例
_instance: TraceStore | None = None
_instance_lock = threading.Lock()


def get_trace_store() -> TraceStore:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                import os
                _instance = TraceStore(
                    max_traces=int(os.getenv("TRACE_BUFFER_SIZE", "1000")),
                    persist_to_sqlite=os.getenv("TRACE_PERSIST", "false").lower() == "true",
                )
    return _instance


def record_trace(
    trace_id: str,
    started_at_ms: int,
    ended_at_ms: int,
    complexity: str,
    routing_confidence: float,
    cache_hit: bool,
    answer_length: int,
    spans: Iterable[dict],
) -> None:
    """
    一站式接口：构建 trace dict 并写入 ring buffer。

    业务用法（在 orchestrator.run() 末尾调用）:
        record_trace(
            trace_id=str(uuid.uuid4()),
            started_at_ms=int(start * 1000),
            ended_at_ms=int(time.time() * 1000),
            complexity=complexity.value,
            routing_confidence=routing.confidence,
            cache_hit=cache_hit,
            answer_length=len(answer),
            spans=trace.get("spans", []),
        )
    """
    latency_ms = ended_at_ms - started_at_ms
    spans_list = list(spans) if spans else []
    trace = {
        "trace_id": trace_id,
        "started_at_ms": started_at_ms,
        "ended_at_ms": ended_at_ms,
        "latency_ms": float(latency_ms),
        "complexity": complexity,
        "routing_confidence": routing_confidence,
        "cache_hit": cache_hit,
        "answer_length": answer_length,
        "spans": spans_list,
    }
    get_trace_store().record(trace)
