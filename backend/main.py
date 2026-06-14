"""main.py — FastAPI 应用入口
================================================================================
技术决策记录:
- 日志轮转: 使用 RotatingFileHandler 避免磁盘被日志撑爆
  （单文件 100MB，最多 5 个备份，总占用上限 600MB）。
- 预热: lifespan 启动时主动初始化 orchestrator / LLM client，
  让模型下载、连接建立发生在启动阶段而非首次请求时。
- CORS: 仅允许开发端口，生产环境应通过 env 注入 allowed origins。
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from backend.api import (
    auth_router,
    chat_router,
    documents_router,
    eval_router,
    health_router,
    search_router,
    stream_router,
)
from backend.config import get_config
from backend.middleware.rate_limiter import RateLimitMiddleware, get_rate_limiter

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """配置日志（轮转文件 + stdout）"""
    cfg = get_config()
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    formatter = logging.Formatter(cfg.logging.format)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_path = Path(cfg.logging.file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=cfg.logging.max_bytes,
            backupCount=cfg.logging.backup_count,
            encoding="utf-8",
        )
        rotating.setFormatter(formatter)
        handlers.append(rotating)
    except Exception as e:
        # 目录不可写时（如容器只读 fs）只回退到 stdout
        print(f"WARN: Failed to create rotating file handler for {log_path}: {e}", file=sys.stderr)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)


_setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动预热 + 优雅关闭"""
    logger.info("RAG API 服务启动中...")
    try:
        # 预热关键组件：触发模型/连接初始化发生在启动阶段
        from backend.api import deps

        deps.get_embedder()
        deps.get_llm_client()
        # P1.2: 预热 Semantic Cache（如果 Redis/redisvl 不可用会降级）
        try:
            deps.get_semantic_cache()
        except Exception as e:
            logger.warning(f"Semantic cache 预热失败（已降级到 no-op）: {e}")

        # P1.5: 启动 OpenTelemetry tracing（OTLP/console 双输出）
        try:
            from backend.observability.tracing import TracingManager

            otlp_endpoint = os.getenv("OTLP_ENDPOINT")  # e.g. http://jaeger:4317
            TracingManager().setup_tracing(otlp_endpoint=otlp_endpoint)
            logger.info(f"Tracing 初始化完成: endpoint={otlp_endpoint or 'console'}")
        except Exception as e:
            logger.warning(f"Tracing 初始化失败: {e}")

        logger.info("RAG API 服务启动完成")
    except Exception as e:
        # 预热失败不阻塞启动：依赖项可惰性按需初始化
        logger.warning(f"启动预热部分失败（将继续按需初始化）: {e}")
    yield
    logger.info("RAG API 服务关闭中...")
    # 关闭 tracing provider，确保 BatchSpanProcessor 缓冲的 spans 落盘
    try:
        from backend.observability.tracing import TracingManager

        TracingManager().shutdown(timeout=5.0)
    except Exception as e:
        logger.debug(f"Tracing shutdown 跳过: {e}")
    logger.info("RAG API 服务关闭完成")


app = FastAPI(
    title="Enterprise RAG System API",
    description="企业级 RAG 系统的 REST API 接口",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件（开发环境允许前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 限流中间件（Redis 滑动窗口）— 启动时硬依赖检查
# P3.2: 之前 try/except: pass 静默吞错，改为启动失败而非降级
try:
    _rate_limiter = get_rate_limiter()
    app.add_middleware(RateLimitMiddleware, limiter=_rate_limiter)
    logger.info("RateLimit 中间件注册成功")
except ImportError:
    # 缺少 redis 库时硬失败（不应静默降级）
    logger.error("RateLimit 中间件注册失败：缺少 redis 依赖")
    raise

# P1.5: 暴露 Prometheus /metrics 端点（与 docs/api 并列）
try:
    app.mount("/metrics", make_asgi_app())
    logger.info("Prometheus /metrics 端点已挂载")
except Exception as e:
    logger.error(f"/metrics 端点挂载失败: {e}")
    raise

# 注册路由
app.include_router(health_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(stream_router, prefix="/api")
app.include_router(eval_router, prefix="/api")  # P2.3 Evaluation dashboard


@app.get("/")
async def root():
    return {
        "name": "Enterprise RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
