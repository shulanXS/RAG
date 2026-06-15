"""
test_auth_flow.py — 用户注册 / 登录 / Token 刷新集成测试
================================================================================
覆盖:
- 完整注册流程（username/password → 创建用户 → 颁发 token）
- 完整登录流程（verify password → 颁发 access/refresh token）
- Token 刷新流程
- 错误凭证返回 401
- 重复用户名注册返回错误
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# fastapi / TestClient 是集成测试可选依赖 (CI 环境会装, 离线开发可缺)
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient


@pytest.fixture
def isolated_user_db(tmp_path, monkeypatch):
    """重定向 user DB 到临时路径，避免污染真实 users.db

    注意: 必须用 yield 而非 return，否则 pytest 不会撤销 monkeypatch，
    导致后续测试继承被 patch 的路径（测试隔离失效）。
    """
    db_path = tmp_path / "users_test.db"
    monkeypatch.setattr("backend.security.auth._get_db_path", lambda: db_path)
    yield db_path


@pytest.fixture
def client(isolated_user_db):
    """FastAPI 客户端（隔离 user DB）"""
    from backend.app import app

    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# 注册
# --------------------------------------------------------------------------


class TestRegistration:
    def test_register_new_user_returns_tokens(self, client):
        """新用户注册成功，返回 access + refresh token"""
        response = client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "securepass123"},
        )
        assert response.status_code in (200, 201)
        body = response.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    def test_register_duplicate_user_fails(self, client):
        """重复用户名注册失败"""
        client.post("/api/auth/register", json={"username": "bob", "password": "securepass"})
        response = client.post(
            "/api/auth/register",
            json={"username": "bob", "password": "anotherpass"},
        )
        assert response.status_code in (400, 409)

    def test_register_short_password_rejected(self, client):
        """弱密码被拒绝"""
        response = client.post(
            "/api/auth/register",
            json={"username": "charlie", "password": "x"},
        )
        # 短密码应该被拒绝（具体状态码取决于验证逻辑）
        assert response.status_code in (400, 422)


# --------------------------------------------------------------------------
# 登录
# --------------------------------------------------------------------------


class TestLogin:
    def test_login_with_valid_credentials(self, client):
        """正确凭证返回 token"""
        client.post("/api/auth/register", json={"username": "dave", "password": "securepass"})
        response = client.post(
            "/api/auth/login",
            json={"username": "dave", "password": "securepass"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body

    def test_login_with_wrong_password(self, client):
        """错误密码返回 401"""
        client.post("/api/auth/register", json={"username": "eve", "password": "securepass"})
        response = client.post(
            "/api/auth/login",
            json={"username": "eve", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """不存在的用户返回 401"""
        response = client.post(
            "/api/auth/login",
            json={"username": "nobody", "password": "anything"},
        )
        assert response.status_code == 401


# --------------------------------------------------------------------------
# Token 刷新
# --------------------------------------------------------------------------


class TestTokenRefresh:
    def test_refresh_with_valid_token(self, client):
        """有效的 refresh token 可换发新 access token"""
        reg = client.post(
            "/api/auth/register",
            json={"username": "frank", "password": "securepass"},
        ).json()
        response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": reg["refresh_token"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body

    def test_refresh_with_invalid_token(self, client):
        """无效 token 返回 401"""
        response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )
        assert response.status_code == 401


# --------------------------------------------------------------------------
# 保护端点
# --------------------------------------------------------------------------


class TestProtectedEndpoints:
    def test_protected_endpoint_requires_token(self, client):
        """未携带 token 访问保护端点返回 401"""
        response = client.get("/api/chat/history?session_id=test")
        assert response.status_code == 401

    def test_protected_endpoint_with_valid_token(self, client):
        """携带有效 token 访问保护端点返回 200"""
        reg = client.post(
            "/api/auth/register",
            json={"username": "grace", "password": "securepass"},
        ).json()
        response = client.get(
            "/api/chat/history?session_id=test",
            headers={"Authorization": f"Bearer {reg['access_token']}"},
        )
        assert response.status_code == 200

    def test_protected_endpoint_with_invalid_token(self, client):
        """无效 token 返回 401"""
        response = client.get(
            "/api/chat/history?session_id=test",
            headers={"Authorization": "Bearer invalid.token"},
        )
        assert response.status_code == 401
