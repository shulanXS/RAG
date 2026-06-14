"""
retrieval 模块 — 混合检索引擎
================================================================================
技术决策记录:
- 两阶段检索: 混合召回 (BM25 + dense, RRF 融合) → Cross-Encoder 重排序。
  这不是两次向量检索，而是两种互补检索范式的组合。
- Hybrid Search 是 2026 FAANG 标准: 纯向量在精确标识符上召回差，
  纯 BM25 无法处理语义 paraphrase。
- RRF 融合: 避免 score 归一化问题，对不同量纲天然鲁棒。
- 移除项 (P0): ColBERT (colbert_retriever.py, 343 行) — 未接入主流程，
  主因是 sentence-transformers ColBERT 变体在长文档上 MaxSim 慢且召回无提升。
- 移除项 (P0): HyDE (hyde.py, 305 行) — 仅在 COMPLEX 路径用，但 99% 流量
  走 SIMPLE，HyDE 多一次 LLM 调用 (200-500ms) 不值得。已在 ARCHITECTURE.md
  的"为什么不做"段解释。
- 移除项 (P0): Parent Document Retrieval (parent_retriever.py, 337 行) —
  Indexer 没有产出 parent chunks，是死代码；如果要做应同时改 chunker + indexer
  + hybrid_search 三处，超出 P0 预算。
"""

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.vector_retriever import VectorRetriever
from backend.retrieval.fusion import RRFFusion, WeightedFusion
from backend.retrieval.reranker import CrossEncoderReranker, CohereReranker, BGEReranker
from backend.retrieval.query_rewriter import QueryRewriter, QueryClassifier, QueryIntent, QueryType
from backend.retrieval.hybrid_search import HybridSearchEngine

__all__ = [
    # Core retrieval
    "BM25Retriever",
    "VectorRetriever",
    "RRFFusion",
    "WeightedFusion",
    "CrossEncoderReranker",
    "CohereReranker",
    "BGEReranker",
    "HybridSearchEngine",
    # Query processing
    "QueryRewriter",
    "QueryClassifier",
    "QueryIntent",
    "QueryType",
]
