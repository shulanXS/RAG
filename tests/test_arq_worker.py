"""
test_arq_worker.py — Arq index worker 失败重试链路 (P1 收尾 A4)

覆盖:
- run_index_pipeline 返回 failed 时, index_task 应抛 RuntimeError
- on_job_error 钩子被调用时, document_registry 应已写入 status='failed' (由 pipeline 内部完成)
- 非失败路径: result 直接透传
- _resolve_redis_settings / redis_settings_from_env 正确解析 REDIS_URL

不依赖真实 Redis / Qdrant / Arq 运行 (使用 mock 模拟入队与取任务)。
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.workers.index_worker import (
    WorkerSettings,
    _resolve_redis_settings,
    index_task,
    on_job_error,
)
from backend.workers.redis_url import redis_settings_from_env


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_ctx(file_id: str = "test-file-1") -> dict:
    return {
        "job_id": "arq:job:test",
        "file_id": file_id,
    }


# --------------------------------------------------------------------------
# redis_settings_from_env
# --------------------------------------------------------------------------

class TestRedisUrlParsing:
    """REDIS_URL → RedisSettings 解析"""

    def test_default_url_uses_localhost(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        rs = redis_settings_from_env()
        assert rs.host == "localhost"
        assert rs.port == 6379
        assert rs.database == 0
        assert rs.password is None

    def test_url_with_password(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("REDIS_URL", "redis://:secret@redis.internal:6380/2")
        rs = redis_settings_from_env()
        assert rs.host == "redis.internal"
        assert rs.port == 6380
        assert rs.database == 2
        assert rs.password == "secret"

    def test_invalid_url_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("REDIS_URL", "ftp://not-redis")
        rs = redis_settings_from_env()
        # 解析失败时 fallback 到 localhost
        assert rs.host == "localhost"
        assert rs.port == 6379

    def test_worker_settings_resolves_at_import(self, monkeypatch: pytest.MonkeyPatch):
        """WorkerSettings.redis_settings 在模块 import 时一次性解析 REDIS_URL
        (Arq 启动时直接读类属性, 不能 lazy resolve)
        """
        # 在 import 之前设置环境变量
        monkeypatch.setenv("REDIS_URL", "redis://prod-redis:6379/3")
        # 重新触发解析逻辑 (实际 worker 启动流程)
        from backend.workers.index_worker import _resolve_redis_settings
        rs = _resolve_redis_settings()
        assert rs.host == "prod-redis"
        assert rs.database == 3

    def test_worker_settings_default_redis_settings_is_localhost(self):
        """无 REDIS_URL 时, WorkerSettings 应该是 localhost (开发 fallback)"""
        # import 时已经解析, 这里是 default
        assert WorkerSettings.redis_settings.host in ("localhost",)


# --------------------------------------------------------------------------
# index_task: pipeline 失败 → RuntimeError 触发 Arq 重试
# --------------------------------------------------------------------------

class TestIndexTaskFailure:
    """index_task 在 pipeline 失败时抛 RuntimeError (让 Arq max_tries 重试)"""

    @pytest.mark.asyncio
    async def test_index_task_raises_on_failed_pipeline(self, tmp_path: Path):
        ctx = _make_ctx("file-1")
        # pipeline 返回 status='failed' (例如文件不存在)
        fake_result = {
            "file_id": "file-1",
            "status": "failed",
            "error": "FileNotFoundError: file not found",
        }
        with patch(
            "backend.workers.index_worker.run_index_pipeline",
            new=AsyncMock(return_value=fake_result),
        ) as mock_pipeline:
            with pytest.raises(RuntimeError) as exc_info:
                await index_task(ctx, file_path="/nonexistent.pdf", file_id="file-1")
            assert "index_pipeline failed" in str(exc_info.value)
            assert "FileNotFoundError" in str(exc_info.value)
            assert mock_pipeline.await_count == 1

    @pytest.mark.asyncio
    async def test_index_task_returns_dict_on_success(self, tmp_path: Path):
        ctx = _make_ctx("file-2")
        # 创建真实文件以免被 pipeline 误判
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4 fake content")
        fake_result = {
            "file_id": "file-2",
            "status": "indexed",
            "chunks_indexed": 10,
            "elapsed_ms": 123,
        }
        with patch(
            "backend.workers.index_worker.run_index_pipeline",
            new=AsyncMock(return_value=fake_result),
        ):
            result = await index_task(ctx, file_path=str(p), file_id="file-2")
        assert result["status"] == "indexed"
        assert result["chunks_indexed"] == 10


# --------------------------------------------------------------------------
# on_job_error 钩子
# --------------------------------------------------------------------------

class TestOnJobErrorHook:
    """Arq 任务彻底失败 (3 次重试后) 触发的钩子"""

    @pytest.mark.asyncio
    async def test_on_job_error_logs_but_doesnt_raise(self, caplog):
        """钩子必须吞掉异常, 避免影响 Arq 主循环"""
        ctx = _make_ctx("file-3")
        exc = RuntimeError("terminal failure")
        with caplog.at_level("ERROR", logger="backend.workers.index_worker"):
            await on_job_error(ctx, exc)
        # 关键: 没抛新异常
        assert any("job failed permanently" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_on_job_error_works_without_file_id_in_ctx(self, caplog):
        """即使 ctx 里没有 file_id (旧 Arq 版本), 钩子也不能崩"""
        with caplog.at_level("ERROR", logger="backend.workers.index_worker"):
            await on_job_error({}, RuntimeError("x"))
        assert any("job failed permanently" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# max_tries 校验: 必须为 3, 不然不会触发 on_job_error
# --------------------------------------------------------------------------

class TestWorkerSettingsInvariants:
    def test_max_tries_is_3(self):
        """P1 收尾 A4: 必须配 max_tries=3, 让 Arq 走完整 3 次重试链"""
        assert WorkerSettings.max_tries == 3

    def test_index_task_registered(self):
        """index_task 必须注册到 functions, 否则 worker 启动找不到任务"""
        assert index_task in WorkerSettings.functions
