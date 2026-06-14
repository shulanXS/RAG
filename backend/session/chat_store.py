"""
backend/session/chat_store.py — Redis Chat History Store
================================================================================
技术决策:
- P0-4: 改用 redis.asyncio.Redis，避免在 FastAPI async 请求中阻塞 event loop。
  原实现用同步 redis.Redis，stream 输出会和 chat IO 抢 event loop 造成卡顿。
- 与 rate_limiter.py P1.2 修复保持一致。
"""

from __future__ import annotations

import json
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# redis-py 的 async 客户端（同步客户端 import 仅做依赖检查）
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis 未安装，Chat History 不可用。请运行: pip install redis")


class ChatStore:
    """
    Redis Chat History Store

    使用 Redis List 存储会话消息:
    - key: chat:{session_id}
    - value: JSON 编码的消息列表
    - TTL: 30 天
    """

    _instance: "ChatStore | None" = None

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,  # 用 db=1，与语义缓存 db=0 区分
        ttl_days: int = 30,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("需要安装 redis: pip install redis")

        import os as _os
        self._host = _os.environ.get("REDIS_HOST", host)
        self._port = int(_os.environ.get("REDIS_PORT", port))
        self._db = db
        self._ttl_seconds = ttl_days * 86400

        # P0-4: async client，所有方法必须 await
        self._client = aioredis.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            decode_responses=False,
        )
        logger.info(f"ChatStore initialized (async): redis://{self._host}:{self._port}/db{self._db}")

    @classmethod
    def get_instance(cls, **kwargs) -> "ChatStore":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def _key(self, session_id: str) -> str:
        return f"chat:history:{session_id}"

    async def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """添加一条消息到会话历史"""
        try:
            msg = json.dumps({
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }, ensure_ascii=False)
            key = self._key(session_id)
            # P0-4: async API，需 await
            await self._client.rpush(key, msg)
            await self._client.expire(key, self._ttl_seconds)
            logger.debug(f"Added message to session {session_id}: role={role}, len={len(content)}")
        except Exception as e:
            logger.warning(f"Failed to add message to session {session_id}: {e}")

    async def get_history(
        self,
        session_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """
        获取会话历史。

        Args:
            session_id: 会话 ID
            limit: 返回最近 N 条消息（默认 20）
            offset: 从末尾往前偏移量

        Returns:
            消息列表，格式: [{"role": "...", "content": "...", "metadata": {}}, ...]
        """
        try:
            key = self._key(session_id)
            # P0-4: async API
            raw = await self._client.lrange(key, -(limit + offset), -1 - offset if offset > 0 else -1)
            messages = []
            for item in raw:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                messages.append(json.loads(item))
            return messages
        except Exception as e:
            logger.warning(f"Failed to get history for session {session_id}: {e}")
            return []

    async def get_history_count(self, session_id: str) -> int:
        """获取会话消息总数"""
        try:
            key = self._key(session_id)
            return await self._client.llen(key)
        except Exception:
            return 0

    async def clear_session(self, session_id: str) -> bool:
        """清空指定会话的历史"""
        try:
            key = self._key(session_id)
            await self._client.delete(key)
            logger.info(f"Cleared session: {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear session {session_id}: {e}")
            return False

    async def ping(self) -> bool:
        """检查 Redis 连接是否正常"""
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def get_stats(self, session_id: str) -> dict:
        """获取指定会话的统计信息"""
        try:
            key = self._key(session_id)
            total = await self._client.llen(key)
            return {
                "session_id": session_id,
                "total_messages": total,
                "ttl_seconds": self._ttl_seconds,
            }
        except Exception as e:
            return {"session_id": session_id, "error": str(e)}
