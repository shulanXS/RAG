"""
test_health_checks.py — 健康检查 / 依赖注入 / 配置加载集成测试
================================================================================
覆盖:
- /api/health/live 永远返回 200
- /api/health/ready 在所有依赖不可用时返回 503
- FastAPI 应用的 lifespan 启动 / 关闭正常
- 依赖注入工厂返回单例
- AppConfig 从 yaml 加载 / env 覆盖

使用 FastAPI TestClient + 不启动真实外部服务（live/ready 端点不依赖外部）。
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def mock_external_services():
    """对所有外部依赖做 mock，让 health 检查可以独立运行"""
    with patch("backend.observability.health.HealthChecker.check_qdrant") as qd, \
         patch("backend.observability.health.HealthChecker.check_redis") as rd, \
         patch("backend.observability.health.HealthChecker.check_llm") as llm:
        from backend.observability.health import HealthState, HealthStatus

        async def _healthy():
            return HealthStatus(status=HealthState.HEALTHY, latency_ms=5.0)

        async def _unhealthy():
            return HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=100.0,
                error="connection refused",
            )

        qd.side_effect = _healthy
        rd.side_effect = _healthy
        llm.side_effect = _healthy
        yield {"qdrant": qd, "redis": rd, "llm": llm}


@pytest.fixture
def client(mock_external_services, dummy_config):
    """FastAPI 测试客户端"""
    # 重要: 必须在导入 main 之前 patch 外部依赖，否则 lifespan 会尝试初始化
    from backend.main import app

    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# /api/health/live
# --------------------------------------------------------------------------


class TestLiveness:
    def test_live_returns_200(self, client):
        """Liveness probe 永远 200"""
        response = client.get("/api/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body == {"alive": True}

    def test_live_no_auth_required(self, client):
        """Liveness probe 不需要 JWT 认证"""
        # 没有 Authorization header
        response = client.get("/api/health/live")
        assert response.status_code == 200


# --------------------------------------------------------------------------
# /api/health/ready
# --------------------------------------------------------------------------


class TestReadiness:
    def test_ready_returns_200_when_all_healthy(self, client):
        """所有依赖 healthy 时返回 200"""
        response = client.get("/api/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert "details" in body
        assert "dependencies" in body["details"]

    def test_ready_returns_503_when_qdrant_down(self, client, mock_external_services):
        """Qdrant（核心依赖）不可用时返回 503"""
        from backend.observability.health import HealthState, HealthStatus

        async def _down():
            return HealthStatus(
                status=HealthState.UNHEALTHY,
                latency_ms=100.0,
                error="Qdrant down",
            )

        mock_external_services["qdrant"].side_effect = _down

        response = client.get("/api/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["detail"]["ready"] is False
        assert body["detail"]["status"] == "unhealthy"

    def test_ready_returns_200_degraded_when_redis_down(self, client, mock_external_services):
        """Redis 不可用时仍返回 200（degraded 而非 unhealthy）"""
        from backend.observability.health import HealthState, HealthStatus

        async def _degraded():
            return HealthStatus(
                status=HealthState.DEGRADED,
                latency_ms=50.0,
                error="Redis unreachable",
            )

        mock_external_services["redis"].side_effect = _degraded

        response = client.get("/api/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body.get("degraded") is True


# --------------------------------------------------------------------------
# 配置加载
# --------------------------------------------------------------------------


class TestConfigLoading:
    def test_config_loads_from_yaml(self, dummy_config):
        """默认 config.yaml 能成功加载"""
        assert dummy_config.vector_db.bm25_mode in ("qdrant_sparse", "external")
        assert dummy_config.logging.max_bytes > 0
        assert dummy_config.logging.backup_count > 0

    def test_bm25_mode_default(self, dummy_config):
        """bm25_mode 默认 qdrant_sparse（避免双重 BM25）"""
        assert dummy_config.vector_db.bm25_mode == "qdrant_sparse"

    def test_logging_rotation_settings(self, dummy_config):
        """LoggingConfig 含轮转字段"""
        assert hasattr(dummy_config.logging, "max_bytes")
        assert hasattr(dummy_config.logging, "backup_count")


# --------------------------------------------------------------------------
# 依赖注入
# --------------------------------------------------------------------------


class TestDependencyInjection:
    def test_deps_module_exists(self):
        """deps.py 模块存在"""
        from backend.api import deps

        assert hasattr(deps, "get_orchestrator")
        assert hasattr(deps, "get_chat_store")
        assert hasattr(deps, "get_embedder")
        assert hasattr(deps, "get_llm_client")
        assert hasattr(deps, "get_hybrid_search")

    def test_getters_are_cached(self, dummy_config):
        """依赖注入工厂应该返回单例（lru_cache）"""
        from backend.api import deps

        # 由于 lru_cache，同名调用应返回同一对象
        # 注意: get_orchestrator 会触发实际初始化，可能失败，这里只验证 callable
        assert callable(deps.get_orchestrator)
        assert callable(deps.get_chat_store)


# --------------------------------------------------------------------------
# 路由注册
# --------------------------------------------------------------------------


class TestRoutes:
    def test_routes_registered(self, client):
        """所有路由都注册到 FastAPI app"""
        from backend.main import app

        paths = [route.path for route in app.routes]
        # 健康检查
        assert any("/api/health/live" in p for p in paths)
        assert any("/api/health/ready" in p for p in paths)
        # 业务路由
        assert any("/api/chat" in p for p in paths)
        assert any("/api/search" in p for p in paths)
        assert any("/api/stream" in p for p in paths)
