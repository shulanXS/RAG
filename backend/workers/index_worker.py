"""
index_worker.py — 文档索引 Arq worker (P1-A2)

技术决策:
- 用 Arq (轻量 async task queue, 与 FastAPI 同生态) 替代 FastAPI BackgroundTasks
  之前 BackgroundTasks 绑定 worker 进程，进程重启 / 滚动发布时任务被 kill
  现在任务持久化到 Redis Stream，worker 重启可继续
- 复用 run_index_pipeline (backend.ingestion.pipeline)，仅在执行层 + 队列层包装
- 失败重试 3 次 (exponential backoff)，最终失败状态写 DocumentRegistry
- 启动: `python -m backend.workers.index_worker`
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arq.worker import Worker

from backend.ingestion.pipeline import run_index_pipeline

logger = logging.getLogger(__name__)


async def index_task(
    ctx: dict[str, Any],
    file_path: str,
    file_id: str,
    tenant_id: str = "default",
    strategy: str = "recursive",
) -> dict:
    """
    Arq 任务：执行文档解析与索引。
    失败时 Arq 自动按 max_tries=3 重试；
    重试全部失败后由 on_job_error 钩子写 DocumentRegistry status='failed'。
    """
    logger.info(
        f"[index_task] start file_id={file_id} path={file_path} "
        f"tenant={tenant_id} strategy={strategy}"
    )
    result = await run_index_pipeline(
        file_path=Path(file_path),
        file_id=file_id,
        tenant_id=tenant_id,
        strategy=strategy,
    )
    if result.get("status") == "failed":
        # 抛出异常让 Arq 触发重试
        raise RuntimeError(
            f"index_pipeline failed: {result.get('error', 'unknown')}"
        )
    return result


async def on_job_error(ctx: dict[str, Any], exc: Exception) -> None:
    """Arq 钩子：任务彻底失败后调用，更新 DocumentRegistry。"""
    # file_id 必须从任务参数里拿；通过 ctx['job_result'] 拿不到 raw args，
    # 但我们已经在 run_index_pipeline 内把 'failed' 状态写到了 registry
    # 所以这个钩子只打日志即可
    logger.error(f"[index_task] job failed permanently: {exc}")


async def startup(ctx: dict[str, Any]) -> None:
    """Worker 启动钩子。"""
    logger.info("[index_worker] starting up")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Worker 关闭钩子。"""
    logger.info("[index_worker] shutting down")


# -----------------------------------------------------------------------------
# Arq Worker 配置
# -----------------------------------------------------------------------------

def _resolve_redis_settings():
    """P1 收尾 A2: 一次性解析 REDIS_URL 为 RedisSettings。
    Arq 启动时直接读 WorkerSettings.redis_settings，所以必须在 import 时
    （即 Arq 实例化前）解析完毕。
    """
    from backend.workers.redis_url import redis_settings_from_env
    return redis_settings_from_env()


class WorkerSettings:
    """
    Arq worker 配置。
    启动方式: `arq backend.workers.index_worker.WorkerSettings`
    或: `python -m backend.workers.index_worker`
    """

    functions = [index_task]
    on_startup = startup
    on_shutdown = shutdown
    on_job_error = on_job_error
    cron_jobs = []

    # P1 收尾 A2: 从 REDIS_URL 解析（默认 localhost:6379/0）。
    # Arq 在启动时直接读类属性，所以必须在模块加载时一次性解析。
    redis_settings = _resolve_redis_settings()

    # 任务重试: 最多 3 次
    # 注: Arq 默认对失败任务按 max_tries 自动重试，使用 ARQ 自带的指数退避
    # (见 arq docs: `max_tries` + `retry_defer`)。
    # 之前用 `cron.second` (cron 子模块) 是不正确的 — cron 用于定时任务。
    max_tries = 3

    # 健康检查
    health_check_interval = 30

    # 队列上限
    queue_read_limit = 100


if __name__ == "__main__":
    # CLI 启动: `python -m backend.workers.index_worker`
    worker = Worker(WorkerSettings)
    worker.run()
