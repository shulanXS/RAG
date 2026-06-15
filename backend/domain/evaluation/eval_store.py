"""
eval_store.py — Evaluation results persistence (P2.3)
================================================================================
技术决策:
- 之前 _persist_samples 是空方法，evaluation 结果完全丢内存。
- 本模块提供 SQLite 持久化：覆盖 online_evaluator._persist_samples
  和 batch eval (scripts/eval.py) 的结果。
- 简单两表：runs (一次跑批) + samples (一条 query 的指标)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class EvalStore:
    """SQLite-backed evaluation results store"""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "eval_results.db")
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT,
                    ended_at TEXT,
                    total_cases INTEGER,
                    passed_cases INTEGER,
                    avg_faithfulness REAL,
                    avg_answer_relevancy REAL,
                    avg_context_precision REAL,
                    avg_context_recall REAL,
                    avg_answer_correctness REAL,
                    weakest_metric TEXT,
                    metadata_json TEXT
                )
                """
            )
            # P2 阶段: prompt_version + prompt_hash 列做 eval diff gate
            # 用 ALTER TABLE 而不是重建表，保留历史 runs
            try:
                conn.execute("ALTER TABLE eval_runs ADD COLUMN prompt_version TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                conn.execute("ALTER TABLE eval_runs ADD COLUMN prompt_hash TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE eval_runs ADD COLUMN git_commit TEXT")
            except sqlite3.OperationalError:
                pass
            # 索引: 便于 "同一 prompt_hash 的纵向趋势" 查询
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_prompt_hash ON eval_runs(prompt_hash)")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    sample_id TEXT,
                    timestamp TEXT,
                    query TEXT,
                    answer TEXT,
                    faithfulness REAL,
                    answer_relevancy REAL,
                    context_precision REAL,
                    context_recall REAL,
                    answer_correctness REAL,
                    overall_pass INTEGER,
                    latency_ms REAL,
                    metadata_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_run_id ON eval_samples(run_id)")

    def save_run(
        self,
        run_id: str,
        started_at: str,
        ended_at: str,
        total_cases: int,
        passed_cases: int,
        avg_faithfulness: float,
        avg_answer_relevancy: float,
        avg_context_precision: float,
        avg_context_recall: float,
        avg_answer_correctness: float,
        weakest_metric: str = "",
        metadata: dict | None = None,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
        git_commit: str | None = None,
    ) -> None:
        # P2: 把 prompt_version / prompt_hash / git_commit 合并到 metadata_json
        # 也直接存到独立列便于 SQL 索引
        merged_metadata = dict(metadata or {})
        if prompt_version:
            merged_metadata.setdefault("prompt_version", prompt_version)
        if prompt_hash:
            merged_metadata.setdefault("prompt_hash", prompt_hash)
        if git_commit:
            merged_metadata.setdefault("git_commit", git_commit)
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO eval_runs
                    (run_id, started_at, ended_at, total_cases, passed_cases,
                     avg_faithfulness, avg_answer_relevancy, avg_context_precision,
                     avg_context_recall, avg_answer_correctness, weakest_metric,
                     metadata_json, prompt_version, prompt_hash, git_commit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        started_at,
                        ended_at,
                        total_cases,
                        passed_cases,
                        avg_faithfulness,
                        avg_answer_relevancy,
                        avg_context_precision,
                        avg_context_recall,
                        avg_answer_correctness,
                        weakest_metric,
                        json.dumps(merged_metadata, ensure_ascii=False),
                        prompt_version,
                        prompt_hash,
                        git_commit,
                    ),
                )

    def diff_against_previous(
        self,
        current_run_id: str,
        thresholds: dict | None = None,
    ) -> dict:
        """
        Eval diff gate (P2): 对比当前 run 和上一次同 prompt_hash 的 run，
        如果关键指标下降超过阈值则返回 failure。

        Args:
            current_run_id: 当前 run_id
            thresholds: 各项指标的最大允许下降幅度 (绝对值)
                        默认 {"avg_faithfulness": 0.05, "avg_answer_relevancy": 0.05,
                              "pass_rate": 0.05}

        Returns:
            dict: {
                "passed": bool,
                "current": {...},
                "previous": {...} or None,
                "diffs": {metric: {"current": x, "previous": y, "delta": y-x, "below_threshold": bool}},
                "reason": str,
            }
        """
        thresholds = thresholds or {
            "avg_faithfulness": 0.05,
            "avg_answer_relevancy": 0.05,
            "pass_rate": 0.05,
        }

        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM eval_runs WHERE run_id = ?", (current_run_id,)
                )
                current = cur.fetchone()
                if not current:
                    return {"passed": False, "reason": f"run_id {current_run_id} not found"}

                current_dict = dict(current)
                prompt_hash = current_dict.get("prompt_hash")

                if prompt_hash:
                    cur = conn.execute(
                        """
                        SELECT * FROM eval_runs
                        WHERE prompt_hash = ? AND run_id != ?
                        ORDER BY started_at DESC LIMIT 1
                        """,
                        (prompt_hash, current_run_id),
                    )
                else:
                    cur = conn.execute(
                        "SELECT * FROM eval_runs WHERE run_id != ? ORDER BY started_at DESC LIMIT 1",
                        (current_run_id,),
                    )
                previous = cur.fetchone()

        if not previous:
            return {
                "passed": True,
                "current": current_dict,
                "previous": None,
                "diffs": {},
                "reason": "no previous run to compare (first run)",
            }

        previous_dict = dict(previous)
        diffs: dict = {}
        failed_metrics: list[str] = []

        for metric, max_drop in thresholds.items():
            cur_val = current_dict.get(metric, 0.0) or 0.0
            prev_val = previous_dict.get(metric, 0.0) or 0.0
            delta = prev_val - cur_val  # positive = degraded
            below = delta > max_drop
            diffs[metric] = {
                "current": cur_val,
                "previous": prev_val,
                "delta": delta,
                "max_allowed_drop": max_drop,
                "below_threshold": below,
            }
            if below:
                failed_metrics.append(metric)

        passed = len(failed_metrics) == 0
        if not passed:
            reason = f"metrics regressed beyond threshold: {', '.join(failed_metrics)}"
        elif current_dict.get("prompt_hash") != previous_dict.get("prompt_hash"):
            reason = "prompt changed; previous comparison across prompt versions is informational only"
        else:
            reason = "no regression"

        return {
            "passed": passed,
            "current": current_dict,
            "previous": previous_dict,
            "diffs": diffs,
            "failed_metrics": failed_metrics,
            "reason": reason,
        }

    def save_samples(self, run_id: str, samples: Iterable[dict]) -> int:
        """save_samples: 批量写入样本
        sample 格式: {sample_id, timestamp, query, answer, faithfulness, ...}"""
        n = 0
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                for s in samples:
                    conn.execute(
                        """
                        INSERT INTO eval_samples
                        (run_id, sample_id, timestamp, query, answer,
                         faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness,
                         overall_pass, latency_ms, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            s.get("sample_id", ""),
                            s.get("timestamp", datetime.utcnow().isoformat()),
                            s.get("query", "")[:2000],
                            s.get("answer", "")[:8000],
                            s.get("faithfulness", 0.0),
                            s.get("answer_relevancy", 0.0),
                            s.get("context_precision", 0.0),
                            s.get("context_recall", 0.0),
                            s.get("answer_correctness", 0.0),
                            1 if s.get("overall_pass") else 0,
                            s.get("latency_ms", 0.0),
                            json.dumps(s.get("metadata") or {}, ensure_ascii=False),
                        ),
                    )
                    n += 1
        return n

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM eval_runs ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                )
                return [dict(r) for r in cur.fetchall()]

    def get_latest_run(self) -> dict | None:
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    def get_run_by_id(self, run_id: str) -> dict | None:
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def get_samples_for_run(self, run_id: str, limit: int = 200) -> list[dict]:
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM eval_samples WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                    (run_id, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def get_samples_since(self, since_iso: str, limit: int = 5000) -> list[dict]:
        """
        P1-6: 按时间窗口查询样本 (online evaluator 仪表板用)。

        Args:
            since_iso: ISO 时间字符串（>=）
            limit: 最大返回行数

        Returns:
            list of sample dicts（按 timestamp ASC）
        """
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    """
                    SELECT * FROM eval_samples
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (since_iso, limit),
                )
                return [dict(r) for r in cur.fetchall()]


# 模块级单例
_instance: EvalStore | None = None
_lock = threading.Lock()


def get_eval_store() -> EvalStore:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EvalStore()
    return _instance
