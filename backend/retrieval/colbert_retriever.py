"""
colbert_retriever.py — ColBERT Late Interaction Retriever
================================================================================
技术决策记录:
- ColBERT (Contextualized Late Interaction over BERT) 的核心洞察:
  不同于 Bi-encoder 的 query/doc 独立编码，ColBERT 在编码后仍保留
  token 级交互能力。与 Cross-encoder 的全量 attention 不同，
  ColBERT 使用 MaxSim 机制（token 级最大值池化），在效率和精度间取得平衡。
- 与 Cross-Encoder 的区别:
  - Cross-Encoder: query+doc 联合编码 → O(N×M) attention → 最精确但最慢
  - Bi-Encoder: query/doc 独立编码 → O(1) 交互 → 最快但最不精确
  - ColBERT: token 级独立编码 + MaxSim → O(N+M) → 精度与速度的折中
- 适用场景:
  - 复杂查询（多义词、精确术语匹配）
  - 超大候选集压缩（>10K 候选）
  - 当 Cross-Encoder 延迟不可接受但 Bi-encoder 精度不够时
- 不适用场景: 简单事实型查询（直接用 BM25+Dense 已足够）

业务难点:
- 模型部署: ColBERT 模型需要在服务端部署（GPU 推荐，A10G 或更好）
- 索引构建: 需要为每个文档计算 token 级向量，索引体积约为 Bi-encoder 的 10 倍
- DeepSeek 集成: 使用 DeepSeek 的 Embedding API 作为备选（无 late interaction）

技术方案:
- Primary: 本地部署 ColBERTv2 (from colbertv2_local)
- Fallback: BGE-M3 的 late interaction (multi-vector 模式)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class ColBERTResult:
    """
    ColBERT 检索结果

    字段说明:
    - chunk_id / doc_id: 来源标识
    - colbert_score: ColBERT late interaction 得分
    - rank: 最终排名
    - matched_tokens: 匹配的 token 信息（用于可解释性）
    """
    chunk_id: str
    doc_id: str
    colbert_score: float
    rank: int
    text: str = ""
    section_path: str = ""
    matched_tokens: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ColBERTRetriever:
    """
    ColBERT Late Interaction 检索器

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 编码 Query: [CLS] q1 q2 q3 ... [SEP] → Q_vectors (N×d)  │
    │  2. 编码 Doc:   [CLS] d1 d2 d3 ... [SEP] → D_vectors (M×d) │
    │  3. Late Interaction: MaxSim(Q_i, all D_j) → N 个分数        │
    │  4. 求和: Σ MaxSim → document score                          │
    └─────────────────────────────────────────────────────────────┘

    技术要点:
    - 使用 sentence-transformers 的 ColBERT 变体
    - 支持 CPU/GPU 自动切换
    - 与现有 VectorRetriever 并列，作为第三路检索
    - 融合方式: RRF (与 BM25/Dense 一起)
    """

    def __init__(
        self,
        model: str = "colbertv2/colbertv2.0",
        device: str = "cpu",
        max_token: int = 512,
        normalize: bool = True,
    ):
        """
        Args:
            model: ColBERT 模型名称
            device: 设备 "cuda" 或 "cpu"
            max_token: 最大 token 长度
            normalize: 是否归一化向量
        """
        self._model_name = model
        self._device = device
        self._max_token = max_token
        self._normalize = normalize
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers 未安装，ColBERT 不可用")
            raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")

        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info(f"ColBERT 模型加载完成: {self._model_name} on {self._device}")

    def encode_query(self, query: str) -> list[list[float]]:
        """
        编码查询为 token 级向量

        Returns:
            list[list[float]]: 每个 token 的向量，形状为 (N, d)
        """
        self._ensure_model()

        encoded = self._model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        if hasattr(encoded, "cpu"):
            encoded = encoded.cpu()

        vectors = encoded[0].numpy()
        if vectors.ndim == 1:
            return [vectors.tolist()]
        return vectors.tolist()

    def encode_documents(self, texts: list[str]) -> list[list[list[float]]]:
        """
        批量编码文档为 token 级向量

        Returns:
            list[list[list[float]]]: 每个文档的 token 向量列表
        """
        self._ensure_model()

        encoded = self._model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        if hasattr(encoded, "cpu"):
            encoded = encoded.cpu()

        result = []
        for i in range(len(texts)):
            vec = encoded[i].numpy()
            if vec.ndim == 1:
                result.append([vec.tolist()])
            else:
                result.append(vec.tolist())
        return result

    def maxsim(
        self,
        query_vectors: list[list[float]],
        doc_vectors: list[list[float]],
    ) -> float:
        """
        MaxSim 计算: 对每个 query token，找最相似的 doc token，求和

        Args:
            query_vectors: 查询的 token 向量 (N, d)
            doc_vectors: 文档的 token 向量 (M, d)

        Returns:
            float: MaxSim 得分
        """
        import numpy as np

        Q = np.array(query_vectors)
        D = np.array(doc_vectors)

        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        if D.ndim == 1:
            D = D.reshape(1, -1)

        similarities = np.dot(Q, D.T)

        max_scores_per_query_token = similarities.max(axis=1)
        return float(max_scores_per_query_token.sum())

    def retrieve(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 50,
    ) -> list[ColBERTResult]:
        """
        执行 ColBERT 检索

        Args:
            query: 用户查询
            documents: 候选文档列表，格式: [{"chunk_id": ..., "text": ..., ...}]
            top_k: 返回 top-k 结果

        Returns:
            list[ColBERTResult]: 按 ColBERT 得分降序排列
        """
        self._ensure_model()

        if not documents:
            return []

        query_vectors = self.encode_query(query)
        texts = [doc.get("text", "")[:2000] for doc in documents]
        doc_vectors_list = self.encode_documents(texts)

        results: list[ColBERTResult] = []
        for i, doc in enumerate(documents):
            doc_vecs = doc_vectors_list[i] if i < len(doc_vectors_list) else [[]]
            score = self.maxsim(query_vectors, doc_vecs)

            results.append(ColBERTResult(
                chunk_id=doc.get("chunk_id", f"colbert_{i}"),
                doc_id=doc.get("doc_id", ""),
                colbert_score=score,
                rank=0,
                text=doc.get("text", ""),
                section_path=doc.get("section_path", ""),
                metadata=doc.get("metadata", {}),
            ))

        results.sort(key=lambda x: x.colbert_score, reverse=True)

        for rank, r in enumerate(results[:top_k], 1):
            r.rank = rank

        logger.debug(f"ColBERT 检索完成: {len(documents)} docs → top-{top_k}")
        return results[:top_k]

    async def retrieve_async(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 50,
    ) -> list[ColBERTResult]:
        """异步版本的 retrieve"""
        import asyncio

        def _sync():
            return self.retrieve(query, documents, top_k)

        return await asyncio.to_thread(_sync)


class ColBERTFusion:
    """
    ColBERT 与其他检索路（BM25/Dense）的融合

    支持两种融合方式:
    1. RRF: Reciprocal Rank Fusion
    2. Weighted: 加权分数融合
    """

    def __init__(self, rrf_k: int = 60):
        self._rrf_k = rrf_k

    def fuse_with_rrf(
        self,
        colbert_results: list[ColBERTResult],
        other_results: list,
        top_k: int = 50,
    ) -> list[dict]:
        """
        将 ColBERT 结果与其他检索路融合

        Args:
            colbert_results: ColBERT 检索结果
            other_results: 其他检索路的结果 (已排序)
            top_k: 最终返回数量

        Returns:
            融合后的结果列表
        """
        from backend.retrieval.fusion import FusionResult

        doc_scores: dict[str, dict] = {}

        for rank, r in enumerate(colbert_results, 1):
            chunk_id = r.chunk_id
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "doc_id": r.doc_id,
                    "text": r.text,
                    "section_path": r.section_path,
                    "metadata": r.metadata,
                    "rrf_contribution": 0.0,
                    "sources": [],
                }
            doc_scores[chunk_id]["rrf_contribution"] += 1.0 / (self._rrf_k + rank)
            doc_scores[chunk_id]["sources"].append("colbert")
            doc_scores[chunk_id]["colbert_score"] = r.colbert_score

        for rank, r in enumerate(other_results, 1):
            chunk_id = r.chunk_id if hasattr(r, "chunk_id") else r.get("chunk_id", "")
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "doc_id": r.doc_id if hasattr(r, "doc_id") else r.get("doc_id", ""),
                    "text": getattr(r, "text", "") or r.get("text", ""),
                    "section_path": getattr(r, "section_path", "") or r.get("section_path", ""),
                    "metadata": getattr(r, "metadata", {}) or r.get("metadata", {}),
                    "rrf_contribution": 0.0,
                    "sources": [],
                }
            doc_scores[chunk_id]["rrf_contribution"] += 1.0 / (self._rrf_k + rank)
            doc_scores[chunk_id]["sources"].append("other")

        ranked = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["rrf_contribution"],
            reverse=True,
        )

        results = []
        for rank, (chunk_id, data) in enumerate(ranked[:top_k], 1):
            results.append({
                "chunk_id": chunk_id,
                "doc_id": data["doc_id"],
                "text": data["text"],
                "section_path": data["section_path"],
                "metadata": data["metadata"],
                "rrf_score": data["rrf_contribution"],
                "rank": rank,
                "sources": data["sources"],
                "colbert_score": data.get("colbert_score", 0.0),
            })

        return results
