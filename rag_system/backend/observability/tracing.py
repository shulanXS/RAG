"""
tracing.py — OpenTelemetry tracing for RAG system
================================================================================
技术决策记录:
- 每个 RAG 阶段作为独立 span：query_rewrite / routing / retrieval / rerank / generation
- span 记录：duration、attributes（query_complexity, num_chunks, cache_hit 等）
- 支持 W3C TraceContext 传播（HTTP header 注入/提取）
- 使用 OTLP exporter（支持 Jaeger/Zipkin/Console）
- 采样策略: AlwaysOn（生产环境零遗漏）+ Parent-based（保留 trace 树结构）

业务价值:
- 端到端延迟分析：从 query rewrite 到 generation 的完整链路
- 跨服务追踪：在微服务架构中追踪请求流转
- 根因分析：通过 span 层级快速定位慢查询
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Generator, Literal

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace import set_tracer_provider

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Span Names Constants
# =============================================================================

SPAN_NAME_QUERY_REWRITE = "rag.query_rewrite"
SPAN_NAME_ROUTING = "rag.routing"
SPAN_NAME_RETRIEVAL = "rag.retrieval"
SPAN_NAME_RERANK = "rag.rerank"
SPAN_NAME_GENERATION = "rag.generation"
SPAN_NAME_AGENTIC = "rag.agentic"
SPAN_NAME_EMBEDDING = "rag.embedding"
SPAN_NAME_CACHE_LOOKUP = "rag.cache_lookup"
SPAN_NAME_CACHE_STORE = "rag.cache_store"

# =============================================================================
# 2. Tracer Names
# =============================================================================

TRACER_NAME = "enterprise_rag"


# =============================================================================
# 3. TracingManager Singleton
# =============================================================================


@lru_cache(maxsize=1)
def TracingManager() -> "TracingManagerImpl":
    """
    获取 TracingManager 单例实例。

    用法示例:
        from backend.observability.tracing import TracingManager

        tm = TracingManager()
        tm.setup_tracing(service_name="rag-api", otlp_endpoint="http://jaeger:4317")

        # 使用 context manager 创建 span
        with tm.create_span("rag.retrieval") as span:
            span.set_attribute("num_chunks", 5)
            # ... 业务逻辑 ...
    """
    return TracingManagerImpl()


class TracingManagerImpl:
    """
    OpenTelemetry 追踪管理器

    设计模式: Facade + Singleton
    - 统一封装 OpenTelemetry 配置和 span 操作
    - 单例模式确保 tracer 只初始化一次
    - 支持多种 exporter 配置

    技术要点:
    - TracerProvider: 全局追踪提供者，配置采样策略和 exporter
    - SpanProcessor: BatchSpanProcessor 用于生产（异步批处理）
    - Propagator: W3C TraceContext 用于跨服务传播
    """

    def __init__(self):
        self._tracer: trace.Tracer | None = None
        self._provider: TracerProvider | None = None
        self._propagator = TraceContextTextMapPropagator()
        self._initialized = False

    def setup_tracing(
        self,
        service_name: str = "enterprise_rag",
        service_version: str = "1.0.0",
        otlp_endpoint: str | None = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
    ) -> None:
        """
        初始化 OpenTelemetry tracing

        Args:
            service_name: 服务名称（用于 trace grouping）
            service_version: 服务版本
            otlp_endpoint: OTLP exporter 端点（如 "http://jaeger:4317"）
            console_export: 是否启用 console exporter（用于调试）
            sample_rate: 采样率 (0.0-1.0)，默认 1.0 全采样
        """
        if self._initialized:
            logger.warning("Tracing 已初始化，忽略重复调用")
            return

        try:
            # 创建 Resource
            resource = Resource.create({
                "service.name": service_name,
                "service.version": service_version,
                "deployment.environment": "production",
            })

            # 创建 TracerProvider（带采样策略）
            from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
            sampler = TraceIdRatioBased(sample_rate)

            self._provider = TracerProvider(resource=resource, sampler=sampler)
            set_tracer_provider(self._provider)

            # 添加 Console Exporter（调试用）
            if console_export:
                console_processor = BatchSpanProcessor(ConsoleSpanExporter())
                self._provider.add_span_processor(console_processor)
                logger.info("Console span exporter 已启用")

            # 添加 OTLP Exporter（生产用）
            if otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
                    otlp_processor = BatchSpanProcessor(otlp_exporter)
                    self._provider.add_span_processor(otlp_processor)
                    logger.info(f"OTLP exporter 已配置: {otlp_endpoint}")
                except ImportError:
                    logger.warning(
                        "opentelemetry-exporter-otlp-proto-grpc 未安装，"
                        "OTLP tracing 不可用。请运行: pip install opentelemetry-exporter-otlp-proto-grpc"
                    )

            self._tracer = self._provider.get_tracer(TRACER_NAME)
            self._initialized = True
            logger.info(f"Tracing 已初始化: service={service_name}, sample_rate={sample_rate}")

        except Exception as e:
            logger.error(f"Tracing 初始化失败: {e}")
            raise

    def get_tracer(self) -> trace.Tracer:
        """
        获取 tracer 实例

        Returns:
            OpenTelemetry Tracer

        Raises:
            RuntimeError: 如果未初始化
        """
        if not self._initialized:
            # 自动初始化（懒加载）
            self.setup_tracing()
        return self._tracer

    @contextmanager
    def create_span(
        self,
        span_name: str,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    ) -> Generator[Span, None, None]:
        """
        创建 span 的 context manager

        Args:
            span_name: span 名称（如 "rag.retrieval"）
            kind: span 类型（INTERNAL/CLIENT/SERVER/PRODUCER/CONSUMER）

        Yields:
            Span: 当前 span 对象

        用法示例:
            with tm.create_span("rag.retrieval") as span:
                span.set_attribute("num_chunks", 5)
                results = await hybrid_search.search(query)
                span.set_attribute("num_results", len(results))
        """
        tracer = self.get_tracer()
        with tracer.start_as_current_span(span_name, kind=kind) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def create_span_decorator(
        self,
        span_name: str | None = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    ):
        """
        span 装饰器（用于函数级别的自动追踪）

        Args:
            span_name: 可选，span 名称（默认使用函数名）
            kind: span 类型

        用法示例:
            @tracing_manager.create_span_decorator("rag.embedding")
            def embed_text(text: str) -> list[float]:
                # ... 函数逻辑 ...
                pass
        """
        def decorator(func):
            _span_name = span_name or f"{func.__module__}.{func.__name__}"

            def sync_wrapper(*args, **kwargs):
                with self.create_span(_span_name, kind=kind) as span:
                    span.set_attribute("function", func.__name__)
                    return func(*args, **kwargs)

            async def async_wrapper(*args, **kwargs):
                with self.create_span(_span_name, kind=kind) as span:
                    span.set_attribute("function", func.__name__)
                    return await func(*args, **kwargs)

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def inject_context(self, carrier: dict[str, str]) -> dict[str, str]:
        """
        将当前 trace context 注入到 carrier（用于 HTTP header 传播）

        Args:
            carrier: 目标容器（如 request.headers）

        Returns:
            包含 trace context 的 carrier
        """
        self._propagator.inject(carrier)
        return carrier

    def extract_context(self, carrier: dict[str, str]) -> trace.Context:
        """
        从 carrier 提取 trace context（用于接收外部请求）

        Args:
            carrier: 源容器（如 response.headers）

        Returns:
            OpenTelemetry Context
        """
        return self._propagator.extract(carrier)

    def get_current_span(self) -> Span:
        """
        获取当前活跃的 span

        Returns:
            当前 span（如果没有则返回 noop span）
        """
        return trace.get_current_span()

    def add_span_attributes(self, attributes: dict[str, Any]) -> None:
        """
        向当前 span 添加 attributes

        Args:
            attributes: attribute 字典
        """
        span = self.get_current_span()
        for key, value in attributes.items():
            span.set_attribute(key, value)

    def set_span_status(self, status: StatusCode, description: str = "") -> None:
        """
        设置当前 span 的状态

        Args:
            status: 状态码（OK/ERROR）
            description: 状态描述
        """
        span = self.get_current_span()
        span.set_status(Status(status, description))

    def record_exception(self, exception: Exception) -> None:
        """
        记录异常到当前 span

        Args:
            exception: 异常对象
        """
        span = self.get_current_span()
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception)))

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        关闭 tracing provider（优雅退出）

        Args:
            timeout: 超时时间（秒）
        """
        if self._provider:
            self._provider.shutdown(timeout=timeout)
            logger.info("Tracing provider 已关闭")


# =============================================================================
# 4. Convenience Functions
# =============================================================================


@lru_cache(maxsize=1)
def get_tracer() -> trace.Tracer:
    """
    获取全局 tracer 实例

    Returns:
        OpenTelemetry Tracer
    """
    return TracingManager().get_tracer()


@contextmanager
def create_span(
    span_name: str,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Generator[Span, None, None]:
    """
    快捷函数：创建 span

    用法示例:
        from backend.observability.tracing import create_span

        with create_span("rag.retrieval") as span:
            span.set_attribute("num_chunks", 5)
    """
    tm = TracingManager()
    if not tm._initialized:
        tm.setup_tracing()
    with tm.create_span(span_name, kind) as span:
        yield span


def inject_trace_context(carrier: dict[str, str]) -> dict[str, str]:
    """快捷函数：注入 trace context 到 HTTP header"""
    return TracingManager().inject_context(carrier)


def extract_trace_context(carrier: dict[str, str]) -> trace.Context:
    """快捷函数：从 HTTP header 提取 trace context"""
    return TracingManager().extract_context(carrier)
