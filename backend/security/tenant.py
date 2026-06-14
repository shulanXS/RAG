"""
tenant.py — 多租户隔离层
================================================================================
技术决策记录:
- 隔离粒度: 在 Qdrant payload 中为每个 point 添加 tenant_id 字段，
  检索时通过 Filter 强制约束，确保不同租户的数据物理隔离（应用层强制）。
- 双重保险: 即使业务层忘记传 tenant_id，filter builder 也会注入默认租户，
  避免越权访问。
- 来源: tenant_id 从 JWT sub claim 解析（每个用户唯一对应一个租户）。
  若未来需要支持「用户属于多个租户」，可扩展为 set[str]。
- 权衡: 物理隔离（每个租户独立 collection）vs 逻辑隔离（共享 collection +
  filter）—— 选逻辑隔离，运维更简单，跨租户联邦检索可能性保留。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client.http import models


TENANT_ID_KEY = "tenant_id"
DEFAULT_TENANT_ID = "default"


@dataclass
class TenantContext:
    """租户上下文：从 JWT 解析或显式构造"""

    tenant_id: str

    @classmethod
    def from_token(cls, token_payload: dict | None) -> "TenantContext":
        """从 JWT payload 提取 tenant_id，无 token 时使用默认"""
        if not token_payload:
            return cls(DEFAULT_TENANT_ID)
        sub = token_payload.get("sub", DEFAULT_TENANT_ID)
        # 未来: 显式 tenant claim 优先于 sub
        tenant_id = token_payload.get("tenant_id", sub)
        return cls(tenant_id=str(tenant_id))

    def __post_init__(self) -> None:
        # 防御性归一化：防止空字符串、None 等绕过过滤
        if not self.tenant_id or not str(self.tenant_id).strip():
            self.tenant_id = DEFAULT_TENANT_ID


def build_tenant_filter(
    tenant: TenantContext,
    extra_filter: dict | models.Filter | None = None,
) -> models.Filter:
    """
    构造带 tenant_id 约束的 Qdrant Filter。

    Args:
        tenant: 租户上下文
        extra_filter: 额外的过滤条件（与 tenant 过滤通过 AND 组合）

    Returns:
        models.Filter: 完整的 Qdrant Filter 对象

    注意:
    - 任何调用方都不能绕过此函数构造「裸 filter」而不带 tenant 约束。
    - 单元测试可 mock tenant 来验证隔离行为。
    """
    must_clauses: list[models.Condition] = [
        models.FieldCondition(
            key=TENANT_ID_KEY,
            match=models.MatchValue(value=tenant.tenant_id),
        )
    ]

    if extra_filter is None:
        return models.Filter(must=must_clauses)

    if isinstance(extra_filter, models.Filter):
        # 合并：原 filter 的 must 条件追加到我们的 must 列表
        if extra_filter.must:
            must_clauses.extend(list(extra_filter.must))
        return models.Filter(must=must_clauses)

    if isinstance(extra_filter, dict):
        # 字典格式：与 VectorRetriever._build_filter 兼容
        for key, value in extra_filter.items():
            if isinstance(value, list):
                must_clauses.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=value))
                )
            else:
                must_clauses.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=must_clauses)

    raise TypeError(f"Unsupported extra_filter type: {type(extra_filter)}")


def with_tenant_payload(tenant_id: str, payload: dict) -> dict:
    """
    把 tenant_id 注入到 indexing 时的 payload 中。

    用法:
        payload = with_tenant_payload(tenant.tenant_id, {
            "doc_id": doc_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            ...
        })
        # payload 现在包含 "tenant_id" 字段
    """
    safe_id = (tenant_id or "").strip() or DEFAULT_TENANT_ID
    return {TENANT_ID_KEY: safe_id, **payload}


def ensure_tenant_payload_index(client: Any, collection_name: str) -> None:
    """
    确保 Qdrant collection 在 tenant_id 字段上有 payload 索引，
    让 tenant filter 走索引而非全表扫描。

    应在 collection 创建后或 schema 升级时调用。
    """
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=TENANT_ID_KEY,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info(f"已为 {collection_name}.{TENANT_ID_KEY} 创建 payload 索引")
    except Exception as e:
        # 索引已存在是常见情况，吞掉这种错误
        msg = str(e).lower()
        if "already exists" in msg or "exists" in msg:
            return
        logger.warning(f"创建 tenant payload 索引失败: {e}")


import logging  # noqa: E402

logger = logging.getLogger(__name__)
