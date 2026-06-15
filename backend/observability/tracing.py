"""
tracing.py — OpenTelemetry tracing for RAG system
================================================================================
技术决策记录:
- 每个 RAG 阶段作为独立 span：query_rewrite / routing / retrieval / rerank / generation
- span 记录：duration、attributes（query_complexity, num_chunks, cache_hit 等）
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
from typing import Generator

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

# Span Names Constants

SPAN_NAME_QUERY_REWRITE = "rag.query_rewrite"
SPAN_NAME_ROUTING = "rag.routing"
SPAN_NAME_RETRIEVAL = "rag.retrieval"
SPAN_NAME_RERANK = "rag.rerank"
SPAN_NAME_GENERATION = "rag.generation"
SPAN_NAME_AGENTIC = "rag.agentic"
SPAN_NAME_EMBEDDING = "rag.embedding"
SPAN_NAME_CACHE_LOOKUP = "rag.cache_lookup"
SPAN_NAME_CACHE_STORE = "rag.cache_store"

# Tracer Names

TRACER_NAME = "enterprise_rag"


# TracingManager Singleton


@lru_cache(maxsize=1)
def TracingManager() -> "TracingManagerImpl":
    """
    获取 TracingManager 单例实例。

    用法示例:
        from backend.observability.tracing import TracingManager

        tm = TracingManager()
        tm.setup_tracing(service_name="rag-api", otlp_endpoint="http://jaeger:4317")

        with tm.create_span("rag.retrieval") as span:
            span.set_attribute("num_chunks", 5)
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
        """
        tracer = self.get_tracer()
        with tracer.start_as_current_span(span_name, kind=kind) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def get_current_span(self) -> Span:
        """
        获取当前活跃的 span

        Returns:
            当前 span（如果没有则返回 noop span）
        """
        return trace.get_current_span()

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        关闭 tracing provider（优雅退出）

        Args:
            timeout: 超时时间（秒）
        """
        if self._provider:
            self._provider.shutdown(timeout=timeout)
            logger.info("Tracing provider 已关闭")
