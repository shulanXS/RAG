"""
config.py — 配置入口 (P2-6 拆分后的 shim)
================================================================================
P2-6 拆分后，配置拆为 3 个文件:
- config_schema.py — Pydantic 数据模型
- config_loader.py — ConfigLoader + get_config + env 覆盖
- config.py (本文件) — 向后兼容 re-export shim

外部代码继续 `from backend.config import ...` 即可，无需感知拆分。
"""
from backend.config_loader import ConfigLoader, get_config
from backend.config_schema import (
    AgenticConfig,
    AppConfig,
    ChunkingConfig,
    DeepSeekApiConfig,
    EmbeddingConfig,
    EvalCIConfig,
    EvaluationConfig,
    EvalTestSet,
    EvalThresholds,
    HybridSearchConfig,
    LLMConfig,
    LLMGeneratorConfig,
    LLMRouterConfig,
    LoggingConfig,
    PathsConfig,
    ReActConfig,
    RerankerConfig,
    SemanticCacheConfig,
    VectorDBConfig,
)

__all__ = [
    "AgenticConfig",
    "AppConfig",
    "ChunkingConfig",
    "ConfigLoader",
    "DeepSeekApiConfig",
    "EmbeddingConfig",
    "EvalCIConfig",
    "EvalTestSet",
    "EvalThresholds",
    "EvaluationConfig",
    "HybridSearchConfig",
    "LLMConfig",
    "LLMGeneratorConfig",
    "LLMRouterConfig",
    "LoggingConfig",
    "PathsConfig",
    "ReActConfig",
    "RerankerConfig",
    "SemanticCacheConfig",
    "VectorDBConfig",
    "get_config",
]
