"""
retrieval 模块 — 混合检索引擎
================================================================================
技术决策记录:
- 两阶段检索: 混合召回 (BM25 + dense, RRF 融合) → Cross-Encoder 重排序。
  这不是两次向量检索，而是两种互补检索范式的组合。
- Hybrid Search 是 2026 FAANG 标准: 纯向量在精确标识符上召回差，
  纯 BM25 无法处理语义 paraphrase。
- RRF 融合: 避免 score 归一化问题，对不同量纲天然鲁棒。
"""

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.vector_retriever import VectorRetriever
from backend.retrieval.fusion import RRFFusion, WeightedFusion
from backend.retrieval.reranker import CrossEncoderReranker, CohereReranker, BGEReranker
from backend.retrieval.query_rewriter import QueryRewriter, QueryClassifier, QueryIntent, QueryType
from backend.retrieval.hybrid_search import HybridSearchEngine
from backend.retrieval.hyde import HyDEQueryEnhancer, HyDEHypothesis, HyDEResult
from backend.retrieval.query_expander import QueryExpander, ExpandedQuery, ExpansionResult, QueryIntent as ExpansionIntent
from backend.retrieval.colbert_retriever import ColBERTRetriever, ColBERTResult, ColBERTFusion
from backend.retrieval.parent_retriever import ParentChunkRetriever, ParentChunkResult, build_parent_chunks

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
    "HyDEQueryEnhancer",
    "HyDEHypothesis",
    "HyDEResult",
    "QueryExpander",
    "ExpandedQuery",
    "ExpansionResult",
    "ExpansionIntent",
    # Advanced retrieval
    "ColBERTRetriever",
    "ColBERTResult",
    "ColBERTFusion",
    "ParentChunkRetriever",
    "ParentChunkResult",
    "build_parent_chunks",
]
