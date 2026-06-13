"""
security/__init__.py — 企业级安全模块

导出多租户隔离和 RBAC 访问控制的核心组件。
"""

from backend.security.tenant_isolation import (
    TenantContext,
    TenantIsolationMiddleware,
    TenantAwareSearchMixin,
    TenantManager,
)
from backend.security.rbac import (
    Role,
    Permission,
    RBACPolicy,
    AccessControlError,
    check_access,
)

__all__ = [
    # Tenant Isolation
    "TenantContext",
    "TenantIsolationMiddleware",
    "TenantAwareSearchMixin",
    "TenantManager",
    # RBAC
    "Role",
    "Permission",
    "RBACPolicy",
    "AccessControlError",
    "check_access",
]
