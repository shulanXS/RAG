"""
redis_url.py — 解析 REDIS_URL 环境变量为 Arq RedisSettings (P1 收尾 A2)

用法:
  from backend.workers.redis_url import redis_settings_from_env
  rs = redis_settings_from_env()  # 默认读 REDIS_URL，否则 fallback localhost:6379/0
"""
from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from arq.connections import RedisSettings

logger = logging.getLogger(__name__)


def redis_settings_from_env() -> RedisSettings:
    """
    从 REDIS_URL 环境变量解析 RedisSettings。
    默认值: redis://localhost:6379/0（开发环境无 Redis 时能起 fallback）。
    """
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        u = urlparse(url)
        if u.scheme not in ("redis", "rediss"):
            raise ValueError(f"unsupported scheme: {u.scheme!r}")

        # 数据库默认 0
        database = int(u.path.lstrip("/") or "0") if u.path else 0

        return RedisSettings(
            host=u.hostname or "localhost",
            port=u.port or 6379,
            database=database,
            password=u.password,
        )
    except Exception as e:
        logger.warning(
            f"REDIS_URL 解析失败 ({e}); 使用默认 localhost:6379/0"
        )
        return RedisSettings(host="localhost", port=6379, database=0)
