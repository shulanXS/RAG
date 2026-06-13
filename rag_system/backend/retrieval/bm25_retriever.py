"""
bm25_retriever.py — BM25 关键词检索
================================================================================
技术决策记录:
- BM25 vs TF-IDF: BM25 是 TF-IDF 的改进版本，通过引入 k1 和 b 参数
  解决了词频饱和和文档长度归一化问题，是 2026 年关键词检索的事实标准。
- 为什么需要 BM25: 向量检索在精确标识符（SKU、合同号、政策编号）上表现极差，
  因为这些词在训练语料中很少出现，embedding 质量差。BM25 通过精确字符串匹配
  弥补了这一缺陷。
- 实现选择: 使用 rank_bm25（纯 Python，轻量），生产级可迁移到
  Qdrant native sparse vector 或 ParadeDB。

业务难点:
- 中文分词: BM25 依赖词边界，中文需要先分词。
  解决方案: 使用 jieba 分词（最成熟的中文 NLP 库）。
- 停用词: "的"、"是"、"在" 等高频无意义词需要过滤。
- 短语匹配: BM25 对短语（如「企业级RAG」）的匹配不如 n-gram。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# 依赖检查
try:
    from rank_bm25 import BM25Okapi
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 或 jieba 未安装，BM25 检索不可用。请运行: pip install rank-bm25 jieba")


@dataclass
class BM25Result:
    """
    BM25 检索结果

    字段说明:
    - chunk_id: 对应的 chunk 唯一标识
    - score: BM25 相关性得分（未归一化）
    - rank: 在 BM25 结果中的排名
    - text: chunk 文本（用于展示和 Reranker 输入）
    """
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    text: str
    metadata: dict


class BM25Retriever:
    """
    BM25 关键词检索器

    技术要点:
    - 使用 rank_bm25 库实现 BM25Okapi 算法
    - 中文使用 jieba 分词，英文使用空格分词
    - 支持增量构建索引（适合流式文档更新）
    - 返回 top-k 结果，按 BM25 得分降序排列

    风险考量:
    - 索引构建时间: O(N * L)，N 为文档数，L 为平均文档长度。
      对于 10 万 chunk 的语料，构建时间约 5-10 秒，可接受。
    - 内存占用: BM25 索引需要存储所有 token 到文档的映射，
      约 3-5 倍于原始文本大小。对企业级语料需注意内存规划。
    """

    def __init__(
        self,
        language: Literal["en", "zh", "mixed"] = "mixed",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Args:
            language: 分词语言，"en" 空格分词，"zh" jieba 分词，"mixed" 自动检测
            k1: BM25 term 频率饱和参数。k1↑ → term 频率影响减弱。
            b: BM25 文档长度归一化参数。b↑ → 对短文档更友好。
        """
        if not BM25_AVAILABLE:
            raise ImportError("需要安装 rank_bm25 和 jieba: pip install rank-bm25 jieba")

        self._k1 = k1
        self._b = b
        self._language = language
        self._corpus: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._metadatas: dict[str, dict] = {}
        self._bm25: BM25Okapi | None = None
        # 脏标记：追加数据后需重建索引，但不立即重建（延迟到第一次 search）
        self._dirty = True

    def add_corpus(
        self,
        chunks: list,
        texts: dict[str, str],
        metadatas: dict[str, dict],
    ) -> None:
        """
        将 chunk 语料添加到 BM25 索引中（延迟构建模式）。

        技术要点:
        - 增量追加：追加后标记 dirty，不立即重建索引
        - 延迟构建：在第一次 search() 时检查 dirty 标志，有脏数据则一次性重建
        - 优势：避免 O(N) 全量重建，适合批量追加场景
        """
        for chunk_id, text in texts.items():
            if chunk_id in self._texts:
                # 已存在的 chunk 跳过（幂等性）
                continue
            tokenized = self._tokenize(text)
            self._tokenized_corpus.append(tokenized)
            self._chunk_ids.append(chunk_id)
            self._doc_ids.append(metadatas.get(chunk_id, {}).get("doc_id", ""))
            self._texts[chunk_id] = text
            self._metadatas[chunk_id] = metadatas.get(chunk_id, {})

        # 标记脏，下次 search 时重建
        self._dirty = True
        logger.debug(
            f"BM25 追加 {len(texts)} 个文档，标记 dirty，"
            f"下次 search 时构建索引（共 {len(self._tokenized_corpus)} 个）"
        )

    def build(self) -> None:
        """
        手动触发索引构建（索引脚本结束时调用）。

        用于批量追加完成后一次性构建，避免 search 时突然重建。
        """
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self._k1,
                b=self._b,
            )
            self._dirty = False
            logger.info(f"BM25 索引构建完成: {len(self._tokenized_corpus)} 个文档")

    def search(self, query: str, top_k: int = 50) -> list[BM25Result]:
        """
        执行 BM25 检索。

        Args:
            query: 查询文本
            top_k: 返回 top-k 结果

        Returns:
            按 BM25 得分降序排列的检索结果
        """
        # 延迟构建：追加数据后第一次 search 时重建索引
        if self._dirty and self._tokenized_corpus:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self._k1,
                b=self._b,
            )
            self._dirty = False
            logger.info(f"BM25 索引构建完成（lazy）: {len(self._tokenized_corpus)} 个文档")

        if self._bm25 is None:
            logger.warning("BM25 索引未构建，返回空结果")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 获取 top-k
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk_id = self._chunk_ids[idx]
            results.append(BM25Result(
                chunk_id=chunk_id,
                doc_id=self._doc_ids[idx],
                score=float(scores[idx]),
                rank=rank,
                text=self._texts.get(chunk_id, ""),
                metadata=self._metadatas.get(chunk_id, {}),
            ))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """
        文本分词

        技术决策:
        - 英文: 转为小写 + 空格分词
        - 中文: jieba 精确模式
        - 停用词过滤: 移除常见无意义词（"的"、"是"、标点等）
        """
        import re

        if self._language == "en":
            tokens = text.lower().split()
            return [t for t in tokens if len(t) > 1]

        tokens: list[str] = []
        # 混合语言：先按非中文字符分割，再对每段分词
        segments = re.split(r"([^\u4e00-\u9fff]+)", text)
        for seg in segments:
            if not seg.strip():
                continue
            if re.search(r"[\u4e00-\u9fff]", seg):
                # 中文段落：jieba 分词
                tokens.extend(jieba.cut(seg, cut_all=False))
            else:
                # 英文段落：空格分词
                tokens.extend(w for w in seg.lower().split() if len(w) > 1)

        # 停用词过滤
        stopwords = {
            "的", "了", "是", "在", "和", "与", "或", "及", "等", "对", "为",
            "了", "着", "过", "但", "却", "而", "这", "那", "有", "没有",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]
