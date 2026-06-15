"""混合检索引擎 — Dense + Sparse + RRF + Cross-Encoder Rerank

P1-B30: WeightedFusion 已删除。
P1-B2: QueryClassifier 已删除。
P0-7: BM25Retriever 整模块已删除（统一走 Qdrant 服务端 sparse vector 路径）。
"""
from backend.domain.retrieval.vector_retriever import VectorRetriever
from backend.domain.retrieval.fusion import DynamicRRFFusion, DEFAULT_K_BY_COMPLEXITY
from backend.domain.retrieval.reranker import CrossEncoderReranker, CohereReranker, BGEReranker
from backend.domain.retrieval.query_rewriter import QueryRewriter
from backend.domain.retrieval.hybrid_search import HybridSearchEngine
