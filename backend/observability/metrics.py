"""
metrics.py — Prometheus metrics for RAG system
================================================================================
技术决策记录:
- rag_retrieval_latency_seconds: histogram，按 retrieval/complete/total 分类
- rag_cache_hit_total: counter，按 hit/miss 分类
- rag_llm_tokens_total: counter，按 model 分类
- rag_retrieval_chunks_count: histogram，按 query_complexity 分类
- rag_retrieval_scores: histogram，按 retrieval score 分布
- rag_errors_total: counter，按 error_type 分类
- 每条指标都有适当的 buckets 设置
- 使用 prometheus_client 作为指标库（事实标准，2026 年已广泛采用）
- 单例模式：使用 module-level instance 避免重复初始化

业务价值:
- 延迟监控: P50/P95/P99 延迟用于 SLO 告警
- 缓存命中率: 直接关联 LLM 成本节省
- Token 计数: 用于成本核算和 rate limit 监控
- Error rate: 用于告警和根因分析
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from prometheus_client import Counter, Histogram, Gauge, REGISTRY, generate_latest

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Prometheus Histogram Buckets
# =============================================================================

# 检索延迟 buckets: 5ms ~ 10s（覆盖 BM25 10ms 到 LLM 推理数秒的场景）
RETRIEVAL_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
)

# LLM 延迟 buckets: 100ms ~ 30s
LLM_LATENCY_BUCKETS = (
    0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0
)

# Chunk 数量 buckets: 1 ~ 100
CHUNK_COUNT_BUCKETS = (1, 2, 5, 10, 20, 50, 100)

# Retrieval score buckets: 0.0 ~ 1.0
RETRIEVAL_SCORE_BUCKETS = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0
)

# Token count buckets: 1 ~ 8192
TOKEN_COUNT_BUCKETS = (1, 16, 64, 256, 512, 1024, 2048, 4096, 8192)


# =============================================================================
# 2. Metrics Definitions
# =============================================================================

# --- Retrieval Latency ---
rag_retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds",
    "RAG retrieval latency in seconds",
    labelnames=["stage"],
    buckets=RETRIEVAL_LATENCY_BUCKETS,
)

rag_retrieval_latency_stage_bm25 = rag_retrieval_latency.labels(stage="bm25")
rag_retrieval_latency_stage_dense = rag_retrieval_latency.labels(stage="dense")
rag_retrieval_latency_stage_fusion = rag_retrieval_latency.labels(stage="fusion")
rag_retrieval_latency_stage_rerank = rag_retrieval_latency.labels(stage="rerank")
rag_retrieval_latency_stage_embedding = rag_retrieval_latency.labels(stage="embedding")
rag_retrieval_latency_stage_total = rag_retrieval_latency.labels(stage="total")

# --- Cache Hits ---
rag_cache_hit = Counter(
    "rag_cache_hit_total",
    "RAG cache hit counter",
    labelnames=["result"],
)

rag_cache_hit_hit = rag_cache_hit.labels(result="hit")
rag_cache_hit_miss = rag_cache_hit.labels(result="miss")

# --- LLM Tokens ---
rag_llm_tokens = Counter(
    "rag_llm_tokens_total",
    "RAG LLM token consumption",
    labelnames=["model", "type"],
)

rag_llm_latency = Histogram(
    "rag_llm_latency_seconds",
    "RAG LLM generation latency in seconds",
    labelnames=["model"],
    buckets=LLM_LATENCY_BUCKETS,
)

# --- Retrieval Chunks ---
rag_retrieval_chunks = Histogram(
    "rag_retrieval_chunks_count",
    "Number of chunks retrieved per query",
    labelnames=["query_complexity"],
    buckets=CHUNK_COUNT_BUCKETS,
)

rag_retrieval_scores = Histogram(
    "rag_retrieval_scores",
    "Retrieval score distribution",
    buckets=RETRIEVAL_SCORE_BUCKETS,
)

# --- Query Complexity ---
rag_query_complexity = Histogram(
    "rag_query_complexity_score",
    "Query complexity score (0-1)",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)

# --- Errors ---
rag_errors = Counter(
    "rag_errors_total",
    "RAG system errors",
    labelnames=["error_type", "component"],
)

# --- Active Requests (for concurrency monitoring) ---
rag_active_requests = Gauge(
    "rag_active_requests",
    "Number of active RAG requests",
    labelnames=["stage"],
)

# --- System Health ---
rag_up = Gauge(
    "rag_up",
    "RAG system health (1=healthy, 0=unhealthy)",
)


# =============================================================================
# 3. MetricsCollector Singleton
# =============================================================================


@lru_cache(maxsize=1)
def MetricsCollector() -> "MetricsCollectorImpl":
    """
    获取 MetricsCollector 单例实例（使用 lru_cache 保证）。

    用法示例:
        from backend.observability.metrics import MetricsCollector
        mc = MetricsCollector()
        mc.record_retrieval_latency(stage="total", latency_seconds=0.15)
        mc.record_cache_hit(hit=True)
    """
    return MetricsCollectorImpl()


class MetricsCollectorImpl:
    """
    RAG 系统指标收集器

    设计模式: Facade Pattern
    - 统一封装所有 Prometheus 指标调用
    - 提供高层次的业务语义方法
    - 内部直接操作 prometheus_client 指标对象

    线程安全: prometheus_client 指标对象本身是线程安全的，
    不需要额外加锁。
    """

    def record_retrieval_latency(
        self,
        stage: Literal["bm25", "dense", "fusion", "rerank", "embedding", "total"],
        latency_seconds: float,
        query_complexity: float | None = None,
        num_chunks: int | None = None,
    ) -> None:
        """
        记录检索阶段延迟

        Args:
            stage: 检索阶段（bm25/dense/fusion/rerank/embedding/total）
            latency_seconds: 延迟时间（秒）
            query_complexity: 可选，查询复杂度 (0-1)
            num_chunks: 可选，检索到的 chunk 数量
        """
        if stage == "bm25":
            rag_retrieval_latency_stage_bm25.observe(latency_seconds)
        elif stage == "dense":
            rag_retrieval_latency_stage_dense.observe(latency_seconds)
        elif stage == "fusion":
            rag_retrieval_latency_stage_fusion.observe(latency_seconds)
        elif stage == "rerank":
            rag_retrieval_latency_stage_rerank.observe(latency_seconds)
        elif stage == "embedding":
            rag_retrieval_latency_stage_embedding.observe(latency_seconds)
        elif stage == "total":
            rag_retrieval_latency_stage_total.observe(latency_seconds)
            # 记录查询复杂度（仅在 total 阶段记录）
            if query_complexity is not None:
                rag_query_complexity.observe(query_complexity)
            # 记录 chunk 数量（仅在 total 阶段记录）
            if num_chunks is not None:
                complexity_label = self._complexity_to_label(query_complexity)
                rag_retrieval_chunks.labels(query_complexity=complexity_label).observe(num_chunks)

    def record_cache_hit(self, hit: bool, similarity: float | None = None) -> None:
        """
        记录缓存命中结果

        Args:
            hit: 是否命中缓存
            similarity: 可选，相似度分数
        """
        if hit:
            rag_cache_hit_hit.inc()
        else:
            rag_cache_hit_miss.inc()

    def record_llm_tokens(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        记录 LLM Token 消耗

        Args:
            model: 模型名称
            input_tokens: 输入 token 数量
            output_tokens: 输出 token 数量
        """
        if input_tokens > 0:
            rag_llm_tokens.labels(model=model, type="input").inc(input_tokens)
        if output_tokens > 0:
            rag_llm_tokens.labels(model=model, type="output").inc(output_tokens)

    def record_llm_latency(self, model: str, latency_seconds: float) -> None:
        """
        记录 LLM 生成延迟

        Args:
            model: 模型名称
            latency_seconds: 延迟时间（秒）
        """
        rag_llm_latency.labels(model=model).observe(latency_seconds)

    def record_retrieval_scores(self, scores: list[float]) -> None:
        """
        记录检索分数分布

        Args:
            scores: 检索分数列表
        """
        for score in scores:
            rag_retrieval_scores.observe(score)

    def record_error(
        self,
        error_type: Literal[
            "retrieval", "llm", "embedding", "cache", "reranker", "validation", "timeout", "unknown"
        ],
        component: Literal["retrieval", "llm", "embedding", "cache", "reranker", "orchestrator", "agentic"],
        message: str | None = None,
    ) -> None:
        """
        记录错误

        Args:
            error_type: 错误类型
            component: 错误发生的组件
            message: 可选，错误消息（不记录到指标，仅用于日志）
        """
        rag_errors.labels(error_type=error_type, component=component).inc()
        if message:
            logger.warning(f"RAG error recorded: {error_type}/{component} - {message}")

    def record_active_request(self, stage: str, increment: bool = True) -> None:
        """
        记录活跃请求数

        Args:
            stage: 请求阶段
            increment: True=增加, False=减少
        """
        if increment:
            rag_active_requests.labels(stage=stage).inc()
        else:
            rag_active_requests.labels(stage=stage).dec()

    def set_healthy(self, healthy: bool) -> None:
        """
        设置系统健康状态

        Args:
            healthy: 是否健康
        """
        rag_up.set(1 if healthy else 0)

    def _complexity_to_label(self, complexity: float | None) -> str:
        """将复杂度数值转换为标签"""
        if complexity is None:
            return "unknown"
        if complexity < 0.3:
            return "simple"
        elif complexity < 0.6:
            return "moderate"
        else:
            return "complex"


# =============================================================================
# 4. Convenience Functions (for backward compatibility)
# =============================================================================


def record_retrieval_latency(
    stage: Literal["bm25", "dense", "fusion", "rerank", "embedding", "total"],
    latency_seconds: float,
    query_complexity: float | None = None,
    num_chunks: int | None = None,
) -> None:
    """快捷函数：记录检索延迟"""
    MetricsCollector().record_retrieval_latency(stage, latency_seconds, query_complexity, num_chunks)


def record_cache_hit(hit: bool, similarity: float | None = None) -> None:
    """快捷函数：记录缓存命中"""
    MetricsCollector().record_cache_hit(hit, similarity)


def record_llm_tokens(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """快捷函数：记录 LLM Token"""
    MetricsCollector().record_llm_tokens(model, input_tokens, output_tokens)


def record_error(
    error_type: str,
    component: str,
    message: str | None = None,
) -> None:
    """快捷函数：记录错误"""
    MetricsCollector().record_error(error_type, component, message)


def get_metrics() -> bytes:
    """
    获取 Prometheus metrics 输出（用于 /metrics endpoint）

    Returns:
        metrics in Prometheus text format
    """
    return generate_latest(REGISTRY)
