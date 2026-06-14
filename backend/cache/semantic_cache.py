"""
semantic_cache.py — Redis/Valkey 语义缓存
================================================================================
技术决策记录:
- 语义缓存可节省 40-80% 的 LLM 调用成本，同时将响应时间从数秒压缩到 ~50ms。
- cosine similarity ≥ 0.92 是 precision/cost 的经验最优阈值。
- Redis vs Valkey: 协议兼容，性能相当，Valkey 在某些工作负载有内存效率优势。
- 使用 redisvl AsyncSearchIndex 创建真正的 HNSW 向量索引，支持 FT.SEARCH 向量查询。

业务难点:
- 相似度阈值调优: 阈值↑ = 质量↑ = 命中率↓。
  解决方案: 先以 0.92 部署，用真实 query 测 1 周后根据 hit_rate 调优。
- 缓存过期: 数据新鲜度和缓存命中率的权衡。
  解决方案: TTL=7 天 + LRU 淘汰。
"""

from __future__ import annotations

import json
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# redisvl 依赖检查
try:
    from redisvl import RedisVectorStore
    from redisvl.index import AsyncSearchIndex
    from redisvl.schema import Schema
    from redisvl.fields import TextField, VectorField
    REDISVL_AVAILABLE = True
except ImportError:
    REDISVL_AVAILABLE = False
    logger.warning("redisvl 未安装，语义缓存不可用。请运行: pip install redisvl")


class SemanticCache:
    """
    语义缓存抽象基类

    设计模式: 适配器模式
    - 统一接口，支持 Redis/Valkey/Dragonfly 等多种实现
    """

    async def get(self, query: str, query_embedding: list[float]) -> dict | None:
        raise NotImplementedError

    async def set(self, query: str, query_embedding: list[float], response: dict) -> None:
        raise NotImplementedError

    def get_stats(self) -> dict:
        raise NotImplementedError


class RedisSemanticCache(SemanticCache if REDISVL_AVAILABLE else object):
    """
    Redis 语义缓存实现（基于 redisvl）

    技术要点:
    - 使用 redisvl 的 AsyncSearchIndex 创建 HNSW 向量索引
    - cosine similarity ≥ threshold 时视为命中
    - JSON 存储 LLM 响应（含 answer, citations, confidence）
    - TTL=7 天 + LRU 淘汰策略

    性能指标（Percona 生产测试）:
    - 缓存命中延迟: ~27ms (embedding 23ms + FT.SEARCH 2ms + fetch 1ms)
    - 缓存命中后端到端延迟: 50ms（vs LLM 推理 3000-7000ms）
    - 典型命中率: 30-60%（取决于 query 重复率）
    - 成本节省: 40-80% LLM 调用量

    风险考量:
    - Redis 不可用时的降级: 直接走 LLM，不阻塞用户请求。
    - embedding 版本变化: 模型更新后旧 embedding 无法匹配，缓存自然过期。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        similarity_threshold: float = 0.92,
        ttl_days: int = 7,
        index_name: str = "semantic_cache",
        embedding_dim: int = 1024,
    ):
        if not REDISVL_AVAILABLE:
            raise ImportError("需要安装 redisvl: pip install redisvl")

        import redis

        self._host = host
        self._port = port
        self._threshold = similarity_threshold
        self._ttl_seconds = ttl_days * 86400
        self._index_name = index_name
        self._embedding_dim = embedding_dim

        # 统计信息
        self._hits = 0
        self._misses = 0
        self._total_requests = 0

        # Redis async 客户端（P1.2 升级：避免阻塞 event loop）
        self._redis_client: "redis.asyncio.Redis | None" = None
        self._index: AsyncSearchIndex | None = None

    def _get_client(self) -> "redis.asyncio.Redis":
        """延迟初始化 Redis async 客户端

        P1.2: 之前使用同步 redis.Redis 在 async 上下文里会阻塞 event loop，
        改造为 redis.asyncio.Redis，使 set/get 完全非阻塞。
        """
        if self._redis_client is None:
            import redis.asyncio as aioredis
            self._redis_client = aioredis.Redis(
                host=self._host,
                port=self._port,
                decode_responses=False,
            )
        return self._redis_client

    def _ensure_index(self) -> "AsyncSearchIndex":
        """确保 HNSW 向量索引存在，不存在则创建"""
        if self._index is not None:
            return self._index

        client = self._get_client()

        # 检查索引是否已存在
        try:
            from redisvl.index import Index
            existing = Index.from_client(client, name=self._index_name)
            if existing.exists():
                self._index = existing
                logger.info(f"语义缓存索引 '{self._index_name}' 已存在")
                return self._index
        except Exception:
            pass

        # 构建索引 schema
        schema = Schema(
            name=self._index_name,
            fields=[
                TextField(name="query", attrs={"no_stem": True}),
                TextField(name="answer", attrs={"no_stem": True}),
                TextField(name="response_json", attrs={"no_stem": True}),
                VectorField(
                    name="embedding",
                    algorithm="hnsw",
                    attrs={
                        "type": "float32",
                        "dim": self._embedding_dim,
                        "distance_metric": "cosine",
                        "m": 16,
                        "ef_construction": 128,
                    },
                ),
            ],
        )

        # 创建索引
        self._index = AsyncSearchIndex(schema)
        self._index.connect(
            host=self._host,
            port=self._port,
        )
        self._index.create(overwrite=False, drop=False)

        logger.info(
            f"创建语义缓存索引 '{self._index_name}' "
            f"(dim={self._embedding_dim}, threshold={self._threshold})"
        )
        return self._index

    async def get(
        self,
        query: str,
        query_embedding: list[float],
    ) -> dict | None:
        """
        查询语义缓存

        技术要点:
        - 使用 FT.SEARCH 执行向量 KNN 搜索
        - cosine similarity >= threshold 时视为命中
        - 命中时返回缓存的 LLM 响应
        """
        self._total_requests += 1

        try:
            index = self._ensure_index()
        except Exception as e:
            logger.warning(f"语义缓存初始化失败: {e}")
            self._misses += 1
            return None

        try:
            import numpy as np

            # 归一化 embedding（cosine similarity 需要）
            vec = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            # 执行向量 KNN 查询
            results = await index.search(
                query_vector=vec.tolist(),
                return_fields=["query", "answer", "response_json"],
                top_k=1,
            )

            if not results:
                self._misses += 1
                return None

            best = results[0]
            # redisvl 返回的 score 已经是 cosine similarity (0-1)
            score = float(best.get("vector_distance", 0.0))

            # cosine distance → similarity: 1 - distance (redisvl HNSW cosine 返回的是 distance)
            similarity = 1.0 - score

            if similarity >= self._threshold:
                self._hits += 1
                logger.debug(f"语义缓存命中: similarity={similarity:.3f} >= {self._threshold}")
                response_data = json.loads(best["response_json"])
                return response_data
            else:
                self._misses += 1
                logger.debug(
                    f"语义缓存未命中: best_similarity={similarity:.3f} < {self._threshold}"
                )
                return None

        except Exception as e:
            logger.warning(f"语义缓存查询失败: {e}")
            self._misses += 1
            return None

    async def set(
        self,
        query: str,
        query_embedding: list[float],
        response: dict,
    ) -> None:
        """
        写入语义缓存

        技术要点:
        - JSON 存储完整 LLM 响应（含 answer, citations, confidence）
        - TTL 自动过期
        - embedding 向量归一化后存储（cosine similarity 需要归一化）
        """
        try:
            index = self._ensure_index()
        except Exception as e:
            logger.warning(f"语义缓存初始化失败: {e}")
            return

        try:
            import numpy as np

            # 归一化 embedding（cosine similarity 需要）
            vec = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            response_json = json.dumps(response, ensure_ascii=False)
            # 用 query 内容的哈希作为唯一 key
            import hashlib
            doc_id = hashlib.md5(query.encode()).hexdigest()

            record = {
                "id": doc_id,
                "query": query[:500],
                "answer": response.get("answer", "")[:1000],
                "response_json": response_json,
                "embedding": vec.tolist(),
            }

            # 写入索引（带 TTL）
            await index.add(record)

            # 设置 Redis TTL（redisvl key 格式为 {index_name}:{id}）
            redis_client = self._get_client()
            redis_key = f"{self._index_name}:{doc_id}"
            # P1.2: redis_client 是 redis.asyncio.Redis，expire 是 coroutine
            await redis_client.expire(redis_key, self._ttl_seconds)

            logger.debug(f"语义缓存写入: key={doc_id}, ttl={self._ttl_seconds}s")

        except Exception as e:
            logger.warning(f"语义缓存写入失败: {e}")

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.2%}",
            "estimated_cost_savings": f"{self._hits}/{total} queries",
        }
