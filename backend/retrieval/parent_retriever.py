"""
parent_retriever.py — Parent Document Retrieval (父子文档两级检索)
================================================================================
技术决策记录:
- 为什么需要 Parent Document Retrieval: 传统的 chunk 检索容易丢失文档级上下文。
  当一个 chunk 被召回时，可能只包含部分信息，缺少「这个 chunk 在整篇文档中的位置」
  以及「同文档其他相关 chunk」的信息。
- 两级检索流程:
  - Level 1: 召回父文档（整篇文档的汇总）
  - Level 2: 在父文档的子 chunks 中精确检索
  - 融合: 父文档分数 × 子 chunk 分数
- 适用场景:
  - 需要文档级上下文的分析型查询
  - 「第三章讨论的X观点」类型的问题
  - 跨段落理解的查询
- 不适用场景:
  - 简单的事实型查询（直接用 chunk 检索即可）
  - 超大文档（父文档可能包含数千个 chunks）

业务难点:
- 父文档构建: 索引阶段需要为每个文档生成父 chunk（合并所有 chunks 的摘要）
- 分数融合: 父文档分数与子 chunk 分数的乘积可能导致分数被过度稀释
- 配置: parent_merge_tokens 决定了父 chunk 的粒度

技术方案:
- 索引阶段: 在 chunk 中标记 parent_doc_id，在检索时回溯
- 检索阶段: 先检索 chunks，再按 doc_id 聚合，找到父文档
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParentChunkResult:
    """
    父子文档检索结果

    字段说明:
    - child_chunk: 子 chunk 检索结果
    - parent_doc_id: 父文档 ID
    - parent_text: 父文档文本（用于上下文扩展）
    - parent_score: 父文档相关性分数
    - sibling_chunks: 同父文档的其他相关 chunks
    - fusion_score: 融合后的最终分数
    """
    child_chunk: dict
    parent_doc_id: str
    parent_text: str
    parent_score: float
    sibling_chunks: list[dict] = field(default_factory=list)
    fusion_score: float = 0.0


class ParentChunkRetriever:
    """
    父子文档两级检索器

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 初步检索: 使用基础向量检索找到候选 chunks                  │
    │  2. 父文档召回: 按 doc_id 聚合，找到父文档                    │
    │  3. 子 chunk 精确召回: 在同父文档的 chunks 中精细检索          │
    │  4. 分数融合: parent_score × child_score                    │
    │  5. 结果重排: 按 fusion_score 排序                         │
    └─────────────────────────────────────────────────────────────┘

    设计要点:
    - 与现有 VectorRetriever 互补，不替代
    - 通过配置控制是否启用
    - 支持与 HybridSearchEngine 集成
    """

    def __init__(
        self,
        vector_retriever,
        parent_doc_id_field: str = "parent_doc_id",
        parent_text_field: str = "parent_doc_summary",
        fusion_mode: Literal["multiply", "add", "rrf"] = "multiply",
    ):
        """
        Args:
            vector_retriever: 底层向量检索器实例
            parent_doc_id_field: payload 中父文档 ID 的字段名
            parent_text_field: payload 中父文档文本的字段名
            fusion_mode: 分数融合模式
                - multiply: parent_score × child_score
                - add: parent_score + child_score
                - rrf: 将两路视为独立检索路进行 RRF 融合
        """
        self._vector_retriever = vector_retriever
        self._parent_field = parent_doc_id_field
        self._parent_text_field = parent_text_field
        self._fusion_mode = fusion_mode

    def retrieve(
        self,
        query_vector: list[float],
        top_k: int = 50,
        query_filter: dict | None = None,
        sibling_count: int = 3,
    ) -> list[dict]:
        """
        执行父子文档两级检索

        Args:
            query_vector: 查询 embedding
            top_k: 最终返回的 chunk 数量
            query_filter: Qdrant filter 条件
            sibling_count: 每个召回 chunk 的同文档相关 chunks 数量

        Returns:
            融合后的检索结果列表
        """
        initial_results = self._vector_retriever.search(
            query_vector=query_vector,
            top_k=top_k * 3,
            query_filter=query_filter,
        )

        if not initial_results:
            return []

        grouped = self._group_by_doc(initial_results)
        parent_info = self._fetch_parent_docs(list(grouped.keys()))

        fused_results: list[dict] = []

        for doc_id, chunks in grouped.items():
            parent = parent_info.get(doc_id, {})

            parent_score = parent.get("parent_score", 1.0)
            parent_text = parent.get("text", "")

            for chunk in chunks:
                child_score = chunk.score if hasattr(chunk, "score") else 0.0

                if self._fusion_mode == "multiply":
                    fusion = parent_score * child_score
                elif self._fusion_mode == "add":
                    fusion = parent_score + child_score
                else:
                    fusion = child_score

                sibling_chunks = self._get_sibling_chunks(
                    chunks, chunk, sibling_count
                )

                result = {
                    "chunk_id": chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id"),
                    "doc_id": doc_id,
                    "text": chunk.text if hasattr(chunk, "text") else chunk.get("text", ""),
                    "section_path": chunk.section_path if hasattr(chunk, "section_path") else chunk.get("section_path", ""),
                    "score": child_score,
                    "parent_score": parent_score,
                    "fusion_score": fusion,
                    "parent_text": parent_text,
                    "parent_doc_id": doc_id,
                    "sibling_chunks": sibling_chunks,
                    "metadata": chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {}),
                }
                fused_results.append(result)

        fused_results.sort(key=lambda x: x["fusion_score"], reverse=True)

        for rank, r in enumerate(fused_results[:top_k], 1):
            r["rank"] = rank

        logger.debug(
            f"Parent retrieval: {len(initial_results)} initial → "
            f"{len(grouped)} docs → {len(fused_results)} fused"
        )

        return fused_results[:top_k]

    async def retrieve_async(
        self,
        query_vector: list[float],
        top_k: int = 50,
        query_filter: dict | None = None,
        sibling_count: int = 3,
    ) -> list[dict]:
        """异步版本的 retrieve"""
        import asyncio

        def _sync():
            return self.retrieve(query_vector, top_k, query_filter, sibling_count)

        return await asyncio.to_thread(_sync)

    def _group_by_doc(
        self,
        results: list,
    ) -> dict[str, list]:
        """按 doc_id 分组"""
        grouped: dict[str, list] = {}
        for r in results:
            doc_id = r.doc_id if hasattr(r, "doc_id") else r.get("doc_id", "")
            if doc_id not in grouped:
                grouped[doc_id] = []
            grouped[doc_id].append(r)
        return grouped

    def _fetch_parent_docs(self, doc_ids: list[str]) -> dict[str, dict]:
        """获取父文档信息（通过 Qdrant scroll）"""
        parent_info: dict[str, dict] = {}

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            client = self._vector_retriever._client
            collection = self._vector_retriever._collection_name

            for doc_id in doc_ids:
                results, _ = client.scroll(
                    collection_name=collection,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id),
                            ),
                            models.FieldCondition(
                                key="chunk_index",
                                match=models.MatchValue(value=0),
                            ),
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                )

                if results:
                    p = results[0].payload
                    parent_text = p.get("parent_doc_summary", "") or p.get("text", "")[:500]
                    parent_info[doc_id] = {
                        "text": parent_text,
                        "parent_score": 0.8,
                        "metadata": p.get("metadata", {}),
                    }
                else:
                    parent_info[doc_id] = {"text": "", "parent_score": 0.5}

        except Exception as e:
            logger.warning(f"Failed to fetch parent docs: {e}")

        return parent_info

    def _get_sibling_chunks(
        self,
        doc_chunks: list,
        target_chunk,
        count: int,
    ) -> list[dict]:
        """获取同文档的相关 chunks"""
        if len(doc_chunks) <= 1:
            return []

        target_id = target_chunk.chunk_id if hasattr(target_chunk, "chunk_id") else target_chunk.get("chunk_id")
        siblings = [c for c in doc_chunks if (c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id")) != target_id]

        siblings.sort(
            key=lambda c: (c.score if hasattr(c, "score") else c.get("score", 0)),
            reverse=True,
        )

        return [
            {
                "chunk_id": c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id"),
                "text": c.text if hasattr(c, "text") else c.get("text", ""),
                "section_path": c.section_path if hasattr(c, "section_path") else c.get("section_path", ""),
                "score": c.score if hasattr(c, "score") else c.get("score", 0),
            }
            for c in siblings[:count]
        ]


def build_parent_chunks(chunks: list, merge_tokens: int = 2000) -> list[dict]:
    """
    为索引阶段构建父 chunk

    Args:
        chunks: 文档的所有子 chunks
        merge_tokens: 父 chunk 的最大 token 数

    Returns:
        父 chunk 列表（每个 doc 一个父 chunk）
    """
    from backend.ingestion.chunker import count_tokens

    by_doc: dict[str, list] = {}
    for chunk in chunks:
        doc_id = chunk.doc_id if hasattr(chunk, "doc_id") else chunk.get("doc_id", "")
        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(chunk)

    parent_chunks: list[dict] = []

    for doc_id, doc_chunks in by_doc.items():
        doc_chunks.sort(key=lambda c: c.chunk_index if hasattr(c, "chunk_index") else c.get("chunk_index", 0))

        parent_text_parts = []
        current_tokens = 0

        for chunk in doc_chunks:
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
            tokens = count_tokens(text)

            if current_tokens + tokens > merge_tokens:
                break

            parent_text_parts.append(text)
            current_tokens += tokens

        parent_text = "\n\n".join(parent_text_parts)

        parent_chunks.append({
            "chunk_id": f"{doc_id}_parent",
            "doc_id": doc_id,
            "text": parent_text,
            "parent_doc_summary": parent_text[:500],
            "chunk_index": -1,
            "metadata": {
                "is_parent": True,
                "num_children": len(doc_chunks),
                "parent_merge_tokens": merge_tokens,
            },
        })

    return parent_chunks
