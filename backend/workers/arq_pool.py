"""
arq_pool.py — 提交 Arq 任务的辅助工具 (P1-A2 + 收尾 A2)

设计:
|- 启动时按需建立 Redis 连接池 (懒加载)
|- 提供 `enqueue_index_task` 统一封装，调用方只需传 file_path/file_id/tenant_id
|- 失败时 fallback 到直接调 run_index_pipeline (与之前 BackgroundTasks 等效)
  这样开发环境无 Redis 也能跑
|- Redis 连接信息从 REDIS_URL 环境变量读 (P1 收尾 A2)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def enqueue_index_task(
    file_path: Path | str,
    file_id: str,
    tenant_id: str = "default",
    strategy: str = "recursive",
) -> bool:
    """
    把索引任务推入 Arq 队列。
    Returns True 表示成功入队，False 表示 fallback 到直接执行。
    """
    try:
        from arq.connections import ArqRedis
        from backend.workers.redis_url import redis_settings_from_env
        redis = ArqRedis(redis_settings_from_env())
        await redis.enqueue_job(
            "index_task",
            str(file_path),
            file_id,
            tenant_id,
            strategy,
        )
        logger.info(
            f"[enqueue_index_task] queued file_id={file_id} path={file_path}"
        )
        return True
    except Exception as e:
        logger.warning(
            f"[enqueue_index_task] Arq 入队失败 ({e}); "
            f"fallback 到直接执行 — 注意：此模式下任务不持久化"
        )
        # Fallback: 同步执行 (与之前 BackgroundTasks 行为一致)
        from backend.domain.ingestion.pipeline import run_index_pipeline
        await run_index_pipeline(
            file_path=Path(file_path),
            file_id=file_id,
            tenant_id=tenant_id,
            strategy=strategy,
        )
        return False


async def shutdown_pool() -> None:
    """关闭 Arq 连接 (lifespan 末尾调用)。"""
    try:
        from arq.connections import ArqRedis
        from backend.workers.redis_url import redis_settings_from_env
        redis = ArqRedis(redis_settings_from_env())
        await redis.aclose()
    except Exception:
        pass
