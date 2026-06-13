"""
rbac.py — 基于角色的访问控制
================================================================================
技术决策记录:
- 角色: admin / analyst / viewer
- admin: 完全权限（读、写、删除）
- analyst: 读 + 搜索权限
- viewer: 只读权限（不能删除索引）
- 权限模型存储在 config 中，生产可迁移到 OPA/ Casbin

角色权限矩阵:
| 资源         | 操作       | admin | analyst | viewer |
|--------------|------------|-------|---------|--------|
| documents    | read       |   ✓   |    ✓    |   ✓    |
| documents    | write      |   ✓   |    ✓    |   ✗    |
| documents    | delete     |   ✓   |    ✗    |   ✗    |
| search       | execute    |   ✓   |    ✓    |   ✓    |
| collections  | create     |   ✓   |    ✗    |   ✗    |
| collections  | delete     |   ✓   |    ✗    |   ✗    |
| config       | read       |   ✓   |    ✗    |   ✗    |
| config       | write      |   ✓   |    ✗    |   ✗    |
| users        | manage     |   ✓   |    ✗    |   ✗    |

权衡记录:
- 为什么不用 ABAC (Attribute-Based)?
  → ABAC 灵活性更高但复杂度也更高。RABC 对于当前场景足够，
    且更易于审计和理解。
- 为什么不用外部策略引擎（OPA/Casbin）?
  → 外部引擎增加部署复杂度。初期用配置驱动足够，
    后续可通过 RBACPolicy 接口替换为 OPA。
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 角色枚举
# =============================================================================


class Role(Enum):
    """
    系统角色定义

    技术要点:
    - Enum 确保类型安全，避免字符串拼写错误
    - 值为小写，便于数据库存储和 API 传输
    """
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"

    @classmethod
    def from_string(cls, value: str) -> "Role":
        """从字符串解析角色（不区分大小写）"""
        try:
            return cls(value.lower())
        except ValueError:
            valid = [r.value for r in cls]
            raise ValueError(f"无效角色 '{value}'，有效值: {valid}")

    @classmethod
    def all_roles(cls) -> list[str]:
        """返回所有角色值"""
        return [r.value for r in cls]


# =============================================================================
# 2. 权限模型
# =============================================================================


class Resource(Enum):
    """资源类型枚举"""
    DOCUMENTS = "documents"
    SEARCH = "search"
    COLLECTIONS = "collections"
    CONFIG = "config"
    USERS = "users"


class Action(Enum):
    """操作类型枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    CREATE = "create"
    MANAGE = "manage"


@dataclass
class Permission:
    """
    权限定义

    字段说明:
    - resource: 资源类型
    - action: 操作类型
    - description: 权限描述（用于文档和日志）
    """
    resource: Resource
    action: Action
    description: str = ""

    def matches(self, resource: str | Resource, action: str | Action) -> bool:
        """检查是否匹配给定的资源和操作"""
        res = resource if isinstance(resource, Resource) else Resource(resource)
        act = action if isinstance(action, Action) else Action(action)
        return self.resource == res and self.action == act

    def __str__(self) -> str:
        return f"{self.resource.value}:{self.action.value}"


# =============================================================================
# 3. RBAC 策略引擎
# =============================================================================


@dataclass
class RolePermissions:
    """角色权限配置"""
    permissions: list[Permission] = field(default_factory=list)


class RBACPolicy:
    """
    基于角色的访问控制策略引擎

    技术要点:
    - 使用配置驱动策略，便于修改和审计
    - 支持角色继承（future扩展）
    - 提供灵活的权限检查接口

    用法示例:
        policy = RBACPolicy()
        policy.check_access(["analyst"], Resource.DOCUMENTS, Action.WRITE)  # True
        policy.check_access(["viewer"], Resource.DOCUMENTS, Action.DELETE)  # False
    """

    def __init__(self):
        self._role_permissions: dict[Role, RolePermissions] = {}
        self._init_default_permissions()

    def _init_default_permissions(self) -> None:
        """初始化默认权限配置"""
        # Admin: 完全权限
        admin_permissions = RolePermissions(permissions=[
            Permission(Resource.DOCUMENTS, Action.READ, "读取文档"),
            Permission(Resource.DOCUMENTS, Action.WRITE, "写入文档"),
            Permission(Resource.DOCUMENTS, Action.DELETE, "删除文档"),
            Permission(Resource.SEARCH, Action.EXECUTE, "执行搜索"),
            Permission(Resource.COLLECTIONS, Action.CREATE, "创建集合"),
            Permission(Resource.COLLECTIONS, Action.DELETE, "删除集合"),
            Permission(Resource.COLLECTIONS, Action.READ, "读取集合"),
            Permission(Resource.CONFIG, Action.READ, "读取配置"),
            Permission(Resource.CONFIG, Action.WRITE, "写入配置"),
            Permission(Resource.USERS, Action.MANAGE, "管理用户"),
        ])

        # Analyst: 读写 + 搜索
        analyst_permissions = RolePermissions(permissions=[
            Permission(Resource.DOCUMENTS, Action.READ, "读取文档"),
            Permission(Resource.DOCUMENTS, Action.WRITE, "写入文档"),
            Permission(Resource.SEARCH, Action.EXECUTE, "执行搜索"),
            Permission(Resource.COLLECTIONS, Action.READ, "读取集合"),
        ])

        # Viewer: 只读
        viewer_permissions = RolePermissions(permissions=[
            Permission(Resource.DOCUMENTS, Action.READ, "读取文档"),
            Permission(Resource.SEARCH, Action.EXECUTE, "执行搜索"),
            Permission(Resource.COLLECTIONS, Action.READ, "读取集合"),
        ])

        self._role_permissions = {
            Role.ADMIN: admin_permissions,
            Role.ANALYST: analyst_permissions,
            Role.VIEWER: viewer_permissions,
        }

    def check_access(
        self,
        user_roles: list[str],
        resource: str | Resource,
        action: str | Action,
    ) -> bool:
        """
        检查用户是否具有指定资源的操作权限

        Args:
            user_roles: 用户角色列表
            resource: 资源类型
            action: 操作类型

        Returns:
            bool: 是否允许访问
        """
        # 解析资源
        res = resource if isinstance(resource, Resource) else Resource(resource)
        act = action if isinstance(action, Action) else Action(action)

        # 遍历用户角色
        for role_str in user_roles:
            try:
                role = Role.from_string(role_str)
            except ValueError:
                logger.warning(f"未知角色: {role_str}")
                continue

            # Admin 角色始终放行
            if role == Role.ADMIN:
                return True

            # 检查权限
            role_perms = self._role_permissions.get(role)
            if role_perms:
                for perm in role_perms.permissions:
                    if perm.matches(res, act):
                        logger.debug(
                            f"权限命中: role={role_str}, "
                            f"resource={res.value}, action={act.value}"
                        )
                        return True

        logger.info(
            f"权限拒绝: roles={user_roles}, "
            f"resource={res.value}, action={act.value}"
        )
        return False

    def get_user_permissions(self, user_roles: list[str]) -> list[Permission]:
        """
        获取用户的所有有效权限

        Args:
            user_roles: 用户角色列表

        Returns:
            list[Permission]: 权限列表（去重）
        """
        permissions: dict[str, Permission] = {}

        for role_str in user_roles:
            try:
                role = Role.from_string(role_str)
            except ValueError:
                continue

            role_perms = self._role_permissions.get(role)
            if role_perms:
                for perm in role_perms.permissions:
                    key = f"{perm.resource.value}:{perm.action.value}"
                    permissions[key] = perm

        return list(permissions.values())

    def can_delete(self, user_roles: list[str]) -> bool:
        """检查用户是否有删除权限"""
        return self.check_access(user_roles, Resource.DOCUMENTS, Action.DELETE)

    def can_write(self, user_roles: list[str]) -> bool:
        """检查用户是否有写入权限"""
        return self.check_access(user_roles, Resource.DOCUMENTS, Action.WRITE)

    def can_manage_users(self, user_roles: list[str]) -> bool:
        """检查用户是否有管理用户的权限"""
        return self.check_access(user_roles, Resource.USERS, Action.MANAGE)


# =============================================================================
# 4. 异常定义
# =============================================================================


class AccessControlError(Exception):
    """
    访问控制异常

    当用户尝试执行超出权限的操作时抛出。
    包含详细的错误信息用于日志和用户反馈。
    """

    def __init__(
        self,
        message: str,
        resource: str | None = None,
        action: str | None = None,
        user_roles: list[str] | None = None,
    ):
        super().__init__(message)
        self.resource = resource
        self.action = action
        self.user_roles = user_roles or []

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于 API 响应）"""
        return {
            "error": "access_denied",
            "message": str(self),
            "resource": self.resource,
            "action": self.action,
            "required_roles": self._get_required_roles(),
        }

    def _get_required_roles(self) -> list[str]:
        """推断需要的角色"""
        if self.resource == "documents" and self.action == "delete":
            return ["admin"]
        if self.resource == "users":
            return ["admin"]
        return ["admin"]


# =============================================================================
# 5. 访问控制装饰器
# =============================================================================


def check_access(
    resource: str | Resource,
    action: str | Action,
    policy: RBACPolicy | None = None,
    get_roles_fn: Callable[[], list[str]] | None = None,
):
    """
    访问控制装饰器

    用法示例:
        # 基本用法（从 TenantContext 获取角色）
        @check_access("documents", "write")
        async def delete_document(doc_id: str):
            ...

        # 自定义角色获取函数
        @check_access("collections", "delete", get_roles_fn=lambda: ["admin"])
        def create_collection(name: str):
            ...

        # 集成到 Orchestrator
        class SecureOrchestrator(AgenticOrchestrator):
            @check_access("documents", "delete")
            async def delete_index(self, doc_id: str):
                ...

    技术要点:
    - 使用 functools.wraps 保留原函数签名
    - 支持同步和异步函数
    - 可集成到现有类中（如 Orchestrator）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 获取角色（优先使用自定义函数）
            if get_roles_fn:
                user_roles = get_roles_fn()
            else:
                # 从 TenantContext 获取
                from backend.security.tenant_isolation import get_tenant_context
                ctx = get_tenant_context()
                if ctx is None:
                    raise AccessControlError(
                        "无法获取用户上下文",
                        resource=resource if isinstance(resource, str) else resource.value,
                        action=action if isinstance(action, str) else action.value,
                    )
                user_roles = ctx.roles

            # 检查权限
            _policy = policy or RBACPolicy()
            if not _policy.check_access(user_roles, resource, action):
                raise AccessControlError(
                    f"权限不足: {user_roles} 无法执行 {action} on {resource}",
                    resource=resource if isinstance(resource, str) else resource.value,
                    action=action if isinstance(action, str) else action.value,
                    user_roles=user_roles,
                )

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 获取角色
            if get_roles_fn:
                user_roles = get_roles_fn()
            else:
                from backend.security.tenant_isolation import get_tenant_context
                ctx = get_tenant_context()
                if ctx is None:
                    raise AccessControlError(
                        "无法获取用户上下文",
                        resource=resource if isinstance(resource, str) else resource.value,
                        action=action if isinstance(action, str) else action.value,
                    )
                user_roles = ctx.roles

            # 检查权限
            _policy = policy or RBACPolicy()
            if not _policy.check_access(user_roles, resource, action):
                raise AccessControlError(
                    f"权限不足: {user_roles} 无法执行 {action} on {resource}",
                    resource=resource if isinstance(resource, str) else resource.value,
                    action=action if isinstance(action, str) else action.value,
                    user_roles=user_roles,
                )

            return func(*args, **kwargs)

        # 根据函数类型选择装饰器
        if functools.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# 6. 全局策略实例（便捷访问）
# =============================================================================

_default_policy: RBACPolicy | None = None


def get_policy() -> RBACPolicy:
    """获取全局 RBAC 策略实例"""
    global _default_policy
    if _default_policy is None:
        _default_policy = RBACPolicy()
    return _default_policy


def require_role(role: str):
    """简化装饰器：要求特定角色"""
    return check_access(
        resource="*",
        action="*",
        get_roles_fn=lambda: [role],
    )
