"""
backend/security/auth.py — JWT 认证核心逻辑
"""

from __future__ import annotations

import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

ALGORITHM = "HS256"
_bearer = HTTPBearer(auto_error=False)

# P1.4: 强制要求 JWT_SECRET_KEY 从 env 注入；未设置即 fail-fast，
# 避免重启后所有 token 失效（之前默认会随机生成，每次启动都换 key）。
_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
if not _SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable is required. "
        "Set it in .env or via 'export JWT_SECRET_KEY=...' before starting the API. "
        "Refusing to start with a random secret because that invalidates tokens on restart."
    )

# P1.4: 强制要求 JWT_PEPPER 显式设置。绝不允许硬编码默认 pepper（默认 pepper 会被
# 任何读到源码的人用于离线爆破）。
_PEPPER = os.environ.get("JWT_PEPPER")
if not _PEPPER:
    raise RuntimeError(
        "JWT_PEPPER environment variable is required. "
        "Set it in .env (use a 32+ char random string). "
        "Refusing to start with the hardcoded default because it leaks via source code."
    )

_ACCESS_TOKEN_EXPIRE_MINUTES = 30
_REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------

def _get_db_path() -> Path:
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "users.db"


def _init_db() -> None:
    """确保 users 表存在"""
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                disabled INTEGER DEFAULT 0
            )
        """)


def _get_user_by_username(username: str) -> dict | None:
    _init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, username, password_hash, disabled FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if row:
            return dict(row)
    return None


def _create_user(username: str, password: str) -> int:
    """创建用户，返回 user_id"""
    _init_db()
    pw_hash = pwd_context.hash(password + _PEPPER)
    now = datetime.now(timezone.utc).isoformat()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, pw_hash, now),
        )
        conn.commit()
        return cur.lastrowid


# --------------------------------------------------------------------------
# Password helpers
# --------------------------------------------------------------------------

def verify_password(plain: str, stored_hash: str) -> bool:
    return pwd_context.verify(plain + _PEPPER, stored_hash)


# --------------------------------------------------------------------------
# JWT helpers
# --------------------------------------------------------------------------

def _make_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, _SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(sub: str) -> str:
    return _make_token({"sub": sub, "type": "access"}, timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES))


def create_refresh_token(sub: str) -> str:
    return _make_token({"sub": sub, "type": "refresh"}, timedelta(days=_REFRESH_TOKEN_EXPIRE_DAYS))


def decode_token(token: str) -> dict:
    return jwt.decode(token, _SECRET_KEY, algorithms=[ALGORITHM])


# --------------------------------------------------------------------------
# FastAPI dependencies
# --------------------------------------------------------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict | None:
    """
    可选的当前用户依赖。
    有 token 时返回 payload，无 token 时返回 None（不抛异常）。
    用于可选认证的端点。
    """
    if credentials is None:
        return None
    try:
        payload = decode_token(credentials.credentials)
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None


def require_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> dict:
    """
    强制认证的依赖。
    无 token 或 token 无效时抛出 401。
    """
    try:
        payload = decode_token(credentials.credentials)
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token invalid: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
