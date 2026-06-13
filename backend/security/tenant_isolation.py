"""
tenant_isolation.py — 多租户隔离层
================================================================================
技术决策记录:
- 向量数据库隔离：每个 tenant 有独立的 namespace（Qdrant collection 前缀）
- 检索时自动注入 tenant_id filter，确保租户只能访问自己的数据
- tenant_id 从请求上下文提取，支持 middleware 注入
- 使用 contextvars 实现线程安全的上下文传递（异步兼容）

权衡记录:
- 为什么不用数据库层面的行级安全（RLS）?
  → RLS 需要数据库特定实现，Qdrant 的 payload filter 更通用，
    且与应用层策略保持一致，减少维护负担。
- 为什么用 contextvars 而非 threading.local?
  → contextvars 原生支持 asyncio，且在 greenlet/Fiber 环境表现更好，
    适合现代 Python 并发模型。
"""

from __future__ import annotations

import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 上下文变量（线程/异步安全）
# =============================================================================

_tenant_context_var: ContextVar["TenantContext | None"] = ContextVar(
    "tenant_context", default=None
)

_thread_local = threading.local()


def set_tenant_context(ctx: TenantContext | None) -> None:
    """设置当前线程/协程的租户上下文"""
    _tenant_context_var.set(ctx)
    _thread_local.tenant_context = ctx


def get_tenant_context() -> TenantContext | None:
    """获取当前线程/协程的租户上下文"""
    ctx = _tenant_context_var.get()
    if ctx is not None:
        return ctx
    return getattr(_thread_local, "tenant_context", None)


def clear_tenant_context() -> None:
    """清除当前线程/协程的租户上下文"""
    _tenant_context_var.set(None)
    _thread_local.tenant_context = None


# =============================================================================
# 2. 租户上下文数据模型
# =============================================================================


@dataclass
class TenantContext:
    """
    租户执行上下文

    字段说明:
    - tenant_id: 租户唯一标识（UUID 或数据库主键）
    - user_id: 用户唯一标识
    - roles: 用户角色列表（对应 RBAC 的 Role）
    - metadata: 扩展元数据（如部门、权限级别等）

    技术要点:
    - 使用 dataclass 保证数据结构不可变性（但 fields 默认可变）
    - 支持 JSON 序列化（用于日志、缓存、审计）
    """
    tenant_id: str
    user_id: str
    roles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于日志和审计）"""
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "roles": self.roles,
            "metadata": self.metadata,
        }

    def has_role(self, role: str) -> bool:
        """检查用户是否拥有指定角色"""
        return role in self.roles

    def has_any_role(self, roles: list[str]) -> bool:
        """检查用户是否拥有任意一个指定角色"""
        return any(r in self.roles for r in roles)


# =============================================================================
# 3. 租户隔离中间件
# =============================================================================


class TenantIsolationMiddleware:
    """
    租户隔离中间件 — 从请求中提取租户上下文

    支持的认证方式:
    1. Header 认证: X-Tenant-ID + X-User-ID
    2. API Key 认证: X-API-Key（需要预先配置 tenant mapping）
    3. JWT 认证: 从 JWT token 中提取（如果配置了 JWT secret）

    技术要点:
    - 中间件是可组合的，可以叠加多个认证方式
    - 认证失败时抛出异常，由上层处理（如返回 401）
    """

    def __init__(
        self,
        api_key_mapping: dict[str, tuple[str, str]] | None = None,
        jwt_secret: str | None = None,
        required_roles: list[str] | None = None,
    ):
        """
        Args:
            api_key_mapping: API Key → (tenant_id, user_id) 的映射
            jwt_secret: JWT 验签密钥（如果使用 JWT 认证）
            required_roles: 允许访问的角色列表（None 表示不限制）
        """
        self._api_key_mapping = api_key_mapping or {}
        self._jwt_secret = jwt_secret
        self._required_roles = required_roles

    def extract_from_headers(self, headers: dict[str, str]) -> TenantContext:
        """
        从 HTTP Headers 提取租户上下文

        Args:
            headers: 请求头字典（键小写）

        Returns:
            TenantContext: 租户上下文

        Raises:
            ValueError: 缺少必需的头信息或认证失败
        """
        tenant_id = headers.get("x-tenant-id")
        user_id = headers.get("x-user-id")

        if not tenant_id:
            raise ValueError("缺少 X-Tenant-ID 请求头")
        if not user_id:
            raise ValueError("缺少 X-User-ID 请求头")

        roles = self._extract_roles_from_header(headers)
        return TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=roles,
        )

    def extract_from_api_key(self, api_key: str) -> TenantContext:
        """
        从 API Key 提取租户上下文

        Args:
            api_key: API Key

        Returns:
            TenantContext: 租户上下文

        Raises:
            ValueError: API Key 无效或未注册
        """
        mapping = self._api_key_mapping.get(api_key)
        if mapping is None:
            raise ValueError(f"无效的 API Key")

        tenant_id, user_id = mapping
        logger.info(f"API Key 认证成功: tenant={tenant_id}, user={user_id}")

        return TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=["analyst"],  # API Key 用户默认角色
        )

    def extract_from_jwt(self, token: str) -> TenantContext:
        """
        从 JWT Token 提取租户上下文

        Args:
            token: JWT token 字符串

        Returns:
            TenantContext: 租户上下文

        Raises:
            ValueError: JWT 无效或已过期
        """
        if not self._jwt_secret:
            raise ValueError("JWT 认证未配置")

        import json
        import time
        import base64
        import hmac

        try:
            # 解析 JWT (无库依赖)
            parts = token.split(".")
            if len(parts) != 3:
                raise ValueError("JWT 格式错误")

            header, payload, signature = parts

            # 验签
            expected = base64.urlsafe_b64encode(
                hmac.new(
                    self._jwt_secret.encode(),
                    f"{header}.{payload}".encode(),
                    "sha256",
                ).digest()
            ).rstrip(b"=")
            if not hmac.compare_digest(signature.replace("-", "+").replace("_", "/"), expected.decode()):
                raise ValueError("JWT 签名验证失败")

            # 解析 payload
            payload_data = json.loads(
                base64.urlsafe_b64decode(payload + "==")
            )

            # 检查过期
            if payload_data.get("exp", 0) < time.time():
                raise ValueError("JWT 已过期")

            tenant_id = payload_data.get("tenant_id") or payload_data.get("tid")
            user_id = payload_data.get("user_id") or payload_data.get("sub")
            roles = payload_data.get("roles", [])

            if not tenant_id or not user_id:
                raise ValueError("JWT 缺少 tenant_id 或 user_id")

            return TenantContext(
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                roles=roles,
                metadata={"jwt_claims": payload_data},
            )

        except Exception as e:
            logger.warning(f"JWT 解析失败: {e}")
            raise ValueError(f"JWT 认证失败: {e}")

    def extract_from_request(
        self,
        headers: dict[str, str] | None = None,
        api_key: str | None = None,
        jwt_token: str | None = None,
    ) -> TenantContext:
        """
        自动选择认证方式提取租户上下文

        优先级: JWT > API Key > Headers

        Args:
            headers: HTTP 请求头
            api_key: API Key（可从 header 或 query 参数获取）
            jwt_token: JWT token（优先从 header Authorization 获取）

        Returns:
            TenantContext: 租户上下文
        """
        # JWT 优先
        if jwt_token:
            return self.extract_from_jwt(jwt_token)

        # API Key 其次
        if api_key:
            return self.extract_from_api_key(api_key)

        # Headers 最后
        if headers:
            ctx = self.extract_from_headers(headers)
            self._validate_roles(ctx)
            return ctx

        raise ValueError("未提供任何认证信息")

    def _extract_roles_from_header(self, headers: dict[str, str]) -> list[str]:
        """从请求头提角色"""
        roles_header = headers.get("x-user-roles", "")
        if roles_header:
            return [r.strip() for r in roles_header.split(",") if r.strip()]
        return []

    def _validate_roles(self, ctx: TenantContext) -> None:
        """验证用户角色是否在允许列表中"""
        if self._required_roles and not ctx.has_any_role(self._required_roles):
            raise ValueError(
                f"用户角色 {ctx.roles} 不在允许列表 {self._required_roles} 中"
            )


# =============================================================================
# 4. 租户感知搜索 Mixin
# =============================================================================


class TenantAwareSearchMixin:
    """
    租户感知搜索混入 — 自动注入 tenant_id filter

    使用方式:
        class MySearchEngine(TenantAwareSearchMixin, VectorRetriever):
            pass

    技术要点:
    - Mixin 通过继承添加功能，避免多重继承的菱形问题
    - _get_tenant_filter() 生成 Qdrant payload filter
    - search() 和 hybrid_search() 自动注入 tenant_id
    """

    def _get_tenant_filter(self) -> dict[str, Any] | None:
        """
        获取当前租户的 filter 条件

        Returns:
            dict: Qdrant payload filter，格式为 {"tenant_id": "xxx"}
            None: 如果没有租户上下文（降级为无隔离）
        """
        ctx = get_tenant_context()
        if ctx is None:
            logger.warning("无租户上下文，禁用租户隔离过滤")
            return None
        return {"tenant_id": ctx.tenant_id}

    def _merge_filter(
        self,
        user_filter: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        合并用户自定义 filter 和租户 filter

        技术要点:
        - 租户 filter 始终优先（AND 关系）
        - 避免 tenant_id 被用户 filter 覆盖
        """
        tenant_filter = self._get_tenant_filter()

        if tenant_filter is None:
            return user_filter

        if user_filter is None:
            return tenant_filter

        # 合并: {tenant_id: xxx} + {doc_type: yyy}
        # 渲染为 Qdrant Filter 时会生成 AND 条件
        merged = tenant_filter.copy()
        merged.update(user_filter)
        return merged

    def tenant_aware_search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        query_filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[Any]:
        """
        租户感知的向量检索

        自动注入 tenant_id filter，确保租户只能访问自己的数据。
        """
        merged_filter = self._merge_filter(query_filter)
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            query_filter=merged_filter,
            score_threshold=score_threshold,
        )

    def tenant_aware_hybrid_search(
        self,
        query_vector: list[float],
        sparse_query: dict,
        top_k: int = 50,
        query_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        租户感知的混合检索
        """
        merged_filter = self._merge_filter(query_filter)
        return self.hybrid_search(
            query_vector=query_vector,
            sparse_query=sparse_query,
            top_k=top_k,
            query_filter=merged_filter,
        )


# =============================================================================
# 5. 租户管理器
# =============================================================================


@dataclass
class Tenant:
    """
    租户元数据

    字段说明:
    - id: 租户唯一标识
    - name: 租户名称（用于展示）
    - collection_prefix: Qdrant collection 前缀
    - created_at: 创建时间戳
    - is_active: 是否启用
    """
    id: str
    name: str
    collection_prefix: str
    created_at: float
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class TenantManager:
    """
    租户生命周期管理器

    功能:
    - 创建/删除/查询租户
    - 管理 Qdrant collection 前缀
    - 提供租户元数据存储（内存或可扩展到数据库）

    技术要点:
    - 使用内存字典存储（生产环境应替换为数据库）
    - collection_prefix 用于数据隔离（tenant_{id}_chunks）
    """

    def __init__(self):
        self._tenants: dict[str, Tenant] = {}
        self._api_keys: dict[str, str] = {}  # api_key -> tenant_id

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        api_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Tenant:
        """
        创建新租户

        Args:
            tenant_id: 租户唯一标识
            name: 租户名称
            api_key: 分配的 API Key（可选，自动生成）
            metadata: 扩展元数据

        Returns:
            Tenant: 新创建的租户对象
        """
        import time
        import secrets

        if tenant_id in self._tenants:
            raise ValueError(f"租户 {tenant_id} 已存在")

        # 生成 API Key
        if api_key is None:
            api_key = secrets.token_urlsafe(32)

        tenant = Tenant(
            id=tenant_id,
            name=name,
            collection_prefix=f"tenant_{tenant_id}",
            created_at=time.time(),
            metadata=metadata or {},
        )

        self._tenants[tenant_id] = tenant
        self._api_keys[api_key] = tenant_id

        logger.info(f"创建租户: {tenant_id} ({name}), API Key: {api_key[:8]}...")

        return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """
        删除租户（软删除，标记 is_active=False）

        技术要点:
        - 软删除保留数据用于审计
        - 硬删除需要单独的 purge 操作

        Args:
            tenant_id: 租户唯一标识

        Returns:
            bool: 是否成功删除
        """
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False

        tenant.is_active = False
        logger.info(f"删除租户: {tenant_id}")

        return True

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        """获取租户信息"""
        return self._tenants.get(tenant_id)

    def list_tenants(
        self,
        include_inactive: bool = False,
    ) -> list[Tenant]:
        """
        列出所有租户

        Args:
            include_inactive: 是否包含已删除的租户

        Returns:
            list[Tenant]: 租户列表
        """
        tenants = self._tenants.values()
        if not include_inactive:
            tenants = [t for t in tenants if t.is_active]
        return list(tenants)

    def get_tenant_by_api_key(self, api_key: str) -> Tenant | None:
        """通过 API Key 获取租户"""
        tenant_id = self._api_keys.get(api_key)
        if tenant_id is None:
            return None
        return self._tenants.get(tenant_id)

    def get_collection_name(self, tenant_id: str, base_name: str = "chunks") -> str:
        """
        获取租户专用的 collection 名称

        Args:
            tenant_id: 租户 ID
            base_name: 基础名称（默认 chunks）

        Returns:
            str: tenant_{id}_{base_name}
        """
        return f"tenant_{tenant_id}_{base_name}"

    def register_api_key(self, tenant_id: str, api_key: str) -> None:
        """为租户注册额外的 API Key"""
        if tenant_id not in self._tenants:
            raise ValueError(f"租户 {tenant_id} 不存在")
        self._api_keys[api_key] = tenant_id

    def revoke_api_key(self, api_key: str) -> bool:
        """撤销 API Key"""
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            return True
        return False
