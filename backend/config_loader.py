"""
config_loader.py — 配置加载器 (P2-6 拆分)
================================================================================
P2-6 拆分说明:
- ConfigLoader / get_config / _apply_env_overrides 从 config.py 抽出
- 仅依赖 config_schema 中的 Pydantic 模型，不感知字段细节
- 行为与原 config.py 完全一致 — 保持向后兼容
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from backend.config_schema import AppConfig


# =============================================================================
# 配置加载器
# =============================================================================


class ConfigLoader:
    """
    配置加载器 — 支持 YAML + 环境变量覆盖

    技术决策:
    - 为什么不用 BaseSettings 直接加载 env ?
      → 许多配置项（如 model 名称、URL）不适合用 env 命名约定，
        统一用 YAML 管理，通过 env 覆盖关键 secrets 更清晰。
    - _instance: 类级别缓存确保配置只加载一次，符合单例语义但更显式。
    """

    _instance: AppConfig | None = None
    _yaml_path: str | None = None

    @classmethod
    def load(
        cls,
        yaml_path: str | Path | None = None,
        *,
        reload: bool = False,
    ) -> AppConfig:
        """
        加载配置（支持 YAML + env 覆盖）。

        Args:
            yaml_path: YAML 配置文件路径，默认查找项目根目录 config.yaml
            reload: 强制重新加载（用于测试场景）

        Returns:
            AppConfig: 经验证的完整配置对象
        """
        if yaml_path:
            path = str(yaml_path)
        else:
            # 查找项目根目录的 config.yaml
            root = Path(__file__).parent
            path = str(root / "config.yaml")

        if cls._instance is not None and not reload:
            return cls._instance

        # 加载 YAML
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # 构建 Pydantic 模型（自动验证）
        config = AppConfig(**raw)

        # Env 变量覆盖（仅针对敏感的 key 项）
        config = cls._apply_env_overrides(config)

        cls._instance = config
        cls._yaml_path = path
        return config

    @classmethod
    def _apply_env_overrides(cls, config: AppConfig) -> AppConfig:
        """
        用环境变量覆盖配置中的敏感项。

        技术决策:
        - 不做全量 env 覆盖（那样会让 YAML 失去可读性），
          仅覆盖明确需要 env 管理的项：API keys、URLs、secrets。
        - 覆盖规则: ENV_VAR_NAME 格式（如 QDRANT_URL、ANTHROPIC_API_KEY）
        """
        # LLM API Keys
        if os.environ.get("OPENAI_API_KEY"):
            pass  # 由各 SDK 自己读取 env

        # Qdrant URL
        if qdrant_url := os.environ.get("QDRANT_URL"):
            config.vector_db.url = qdrant_url

        # Redis URL
        if redis_host := os.environ.get("REDIS_HOST"):
            config.cache.host = redis_host

        return config

    @classmethod
    def get(cls) -> AppConfig:
        """获取已加载的配置（未加载则先加载）"""
        if cls._instance is None:
            return cls.load()
        return cls._instance

    @classmethod
    def reload(cls) -> AppConfig:
        """强制重新加载配置"""
        return cls.load(reload=True)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    获取全局配置实例（lru_cache 保证单例）。

    用法示例:
        from backend.config import get_config
        cfg = get_config()
        print(cfg.embedding.backend)
    """
    return ConfigLoader.load()
