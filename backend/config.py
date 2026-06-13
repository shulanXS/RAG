"""
Enterprise RAG System — 核心配置加载器
================================================================================
技术决策记录:
- 使用 Pydantic BaseModel 做配置验证: 在启动时捕获配置错误，而非运行时。
  这是 2026 年 Python 项目的标准实践，比 dict.get() + isinstance 检查好得多。
- YAML + Pydantic 组合: YAML 便于非开发人员修改配置，Pydantic 提供
  类型安全和 IDE 自动补全。Env 变量覆盖支持生产环境 secrets 管理。
- 分层配置设计: 将配置分为 embedding/chunking/vector_db/reranker/llm/
  hybrid_search/agentic/cache/evaluation 9 个独立配置块，
  每个模块只依赖自己需要的配置，避免全局 config 单例的隐式耦合。

权衡记录:
- 为什么不使用动态配置 (如动态改写 chunk_size) ?
  → 配置应该在启动时确定，运行时调优通过代码参数传入而非配置文件，
    避免配置文件成为隐性状态来源。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# =============================================================================
# 1. 配置数据模型 (Pydantic)
# =============================================================================


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置"""
    backend: Literal["voyage", "openai", "bge"] = "voyage"
    voyage_model: str = "voyage-3-large"
    openai_model: str = "text-embedding-3-large"
    bge_model: str = "BAAI/bge-m3"
    dimension: int = 1024
    batch_size: int = 100
    normalize: bool = True
    contextual_prefix_tokens: int = 80


class ChunkingConfig(BaseModel):
    """分块策略配置"""
    strategy: Literal["fixed", "hierarchical", "semantic", "recursive"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 150
    heading_levels: list[int] = Field(default_factory=lambda: [1, 2, 3])
    semantic_threshold: float = 0.7

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_not_exceed_size(cls, v: int, info) -> int:
        if v >= info.data.get("chunk_size", 512):
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class VectorDBConfig(BaseModel):
    """Qdrant 向量数据库配置"""
    provider: Literal["qdrant"] = "qdrant"
    url: str = "http://localhost:6333"
    collection_name: str = "enterprise_rag"
    vector_size: int = 1024
    distance: Literal["Cosine", "Dot", "Euclid"] = "Cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    sparse_enabled: bool = True
    sparse_k1: float = 1.5
    sparse_b: float = 0.75
    batch_size: int = 64


class RerankerConfig(BaseModel):
    """Reranker 配置"""
    provider: Literal["cohere", "bge", "voyage"] = "cohere"
    cohere_model: str = "rerank-3.5"
    bge_model: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5
    candidate_size: int = 50


class LLMGeneratorConfig(BaseModel):
    """生成模型配置"""
    provider: Literal["anthropic", "openai", "google", "deepseek"] = "anthropic"
    model: str = "claude-3-7-sonnet-20250620"
    max_tokens: int = 2048
    temperature: float = 0.3


class LLMRouterConfig(BaseModel):
    """路由模型配置（轻量）"""
    provider: Literal["anthropic", "openai", "google", "deepseek"] = "anthropic"
    model: str = "claude-3-5-haiku-20250620"
    max_tokens: int = 256
    temperature: float = 0.1


class DeepSeekApiConfig(BaseModel):
    """DeepSeek API 配置"""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-chat"
    coder_model: str = "deepseek-coder"


class LLMConfig(BaseModel):
    """LLM 完整配置"""
    deepseek: DeepSeekApiConfig = Field(default_factory=DeepSeekApiConfig)
    generator: LLMGeneratorConfig = Field(default_factory=LLMGeneratorConfig)
    router: LLMRouterConfig = Field(default_factory=LLMRouterConfig)


class HybridSearchConfig(BaseModel):
    """混合搜索配置"""
    individual_top_k: int = 50
    rrf_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5


class ReActConfig(BaseModel):
    """ReAct Agent 配置"""
    max_iterations: int = 5
    early_stop_threshold: float = 0.85


class PlanExecuteConfig(BaseModel):
    """Plan-and-Execute 配置"""
    max_steps: int = 8
    max_subqueries_per_step: int = 3


class AgenticConfig(BaseModel):
    """Agentic 编排配置"""
    complexity_threshold: float = 0.6
    react: ReActConfig = Field(default_factory=ReActConfig)
    plan_execute: PlanExecuteConfig = Field(default_factory=PlanExecuteConfig)


class SemanticCacheConfig(BaseModel):
    """语义缓存配置"""
    enabled: bool = True
    provider: Literal["redis", "valkey"] = "redis"
    host: str = "localhost"
    port: int = 6379
    ttl_days: int = 7
    similarity_threshold: float = 0.92
    index_name: str = "semantic_cache"
    max_entries: int = 100_000


class EvalThresholds(BaseModel):
    """评估指标阈值"""
    faithfulness: float = 0.85
    answer_relevancy: float = 0.75
    context_precision: float = 0.70
    context_recall: float = 0.70
    answer_correctness: float = 0.80


class EvalTestSet(BaseModel):
    """测试集配置"""
    min_size: int = 30
    categories: list[str] = Field(default_factory=lambda: ["simple", "moderate", "difficult"])


class EvalCIConfig(BaseModel):
    """CI/CD 集成配置"""
    fail_on_regression: bool = True
    regression_threshold: float = 0.05


class EvaluationConfig(BaseModel):
    """评估配置"""
    thresholds: EvalThresholds = Field(default_factory=EvalThresholds)
    test_set: EvalTestSet = Field(default_factory=EvalTestSet)
    ci: EvalCIConfig = Field(default_factory=EvalCIConfig)


class LoggingConfig(BaseModel):
    """日志配置"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file: str = "logs/rag_system.log"
    trace_enabled: bool = True
    log_retrieval_scores: bool = True
    log_citation_mapping: bool = True


class PathsConfig(BaseModel):
    """路径配置"""
    data_dir: str = "data/sample_docs"
    output_dir: str = "outputs"


class AppConfig(BaseModel):
    """完整配置模型（根节点）"""
    paths: PathsConfig = Field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    agentic: AgenticConfig = Field(default_factory=AgenticConfig)
    cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# =============================================================================
# 2. 配置加载器
# =============================================================================


class ConfigLoader:
    """
    配置加载器 — 支持 YAML + 环境变量覆盖

    技术决策:
    - 为什么不用 BaseSettings 直接加载 env ?
      → 许多配置项（如 model 名称、URL）不适合用 env 命名约定，
        统一用 YAML 管理，通过 env 覆盖关键 secrets 更清晰。
    - _cache: 类级别缓存确保配置只加载一次，符合单例语义但更显式。
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
