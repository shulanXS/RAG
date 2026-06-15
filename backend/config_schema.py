"""
config_schema.py — Pydantic 配置数据模型 (P2-6 拆分)
================================================================================
P2-6 拆分说明:
- 14 个配置类从原 config.py 抽出，单文件 ~190 行
- 与 ConfigLoader / get_config 解耦 — schema 仅关心字段与验证
- 不再包含 YAML 加载与 env 覆盖逻辑
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# 1. 配置数据模型 (Pydantic)
# =============================================================================


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置"""
    backend: Literal["openai", "deepseek"] = "deepseek"
    openai_model: str = "text-embedding-3-small"
    deepseek_model: str = "BAAI/bge-m3"  # DeepSeek 无官方 embedding，fallback 到本地 BGE
    dimension: int = 1024
    batch_size: int = 100
    normalize: bool = True
    contextual_prefix_tokens: int = 80


class ChunkingConfig(BaseModel):
    """分块策略配置"""
    strategy: Literal["recursive"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 150

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
    batch_size: int = 64


class RerankerConfig(BaseModel):
    """Reranker 配置"""
    provider: Literal["cohere", "bge"] = "cohere"
    cohere_model: str = "rerank-3.5"
    bge_model: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5
    candidate_size: int = 50


class LLMGeneratorConfig(BaseModel):
    """生成模型配置"""
    provider: Literal["openai", "deepseek"] = "deepseek"
    model: str = "deepseek-chat"
    max_tokens: int = 2048
    temperature: float = 0.3


class LLMRouterConfig(BaseModel):
    """路由模型配置（轻量）"""
    provider: Literal["openai", "deepseek"] = "deepseek"
    model: str = "deepseek-chat"
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
    use_structured_output: bool = Field(
        default=True,
        description="是否在非流式生成路径中启用 JSON Schema 约束输出",
    )


class HybridSearchConfig(BaseModel):
    """混合搜索配置"""
    individual_top_k: int = 50
    rrf_k: int = 60
    # P2-B5: DynamicRRFFusion 总开关; 关闭后固定 k=rrf_k
    dynamic_k_enabled: bool = True
    # P2-B5: complexity → k 覆盖映射; None 用默认 DEFAULT_K_BY_COMPLEXITY
    k_by_complexity: dict[str, int] | None = None


class ReActConfig(BaseModel):
    """ReAct Agent 配置"""
    max_iterations: int = 5
    early_stop_threshold: float = 0.85


class AgenticConfig(BaseModel):
    """Agentic 编排配置"""
    complexity_threshold: float = 0.6
    react: ReActConfig = Field(default_factory=ReActConfig)


class SemanticCacheConfig(BaseModel):
    """语义缓存配置"""
    enabled: bool = True
    provider: Literal["redis"] = "redis"
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
    # 日志轮转配置（避免日志撑爆磁盘）
    max_bytes: int = 100 * 1024 * 1024  # 单个日志文件 100MB
    backup_count: int = 5  # 保留 5 个历史日志文件


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
