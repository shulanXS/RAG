"""
embedder.py — 多后端 Embedding 模型封装
================================================================================
技术决策记录:
- 统一接口设计: 不管底层是 OpenAI/voyage/BGE，对外暴露统一的 embed() 和
  embed_batch() 接口。这是典型的适配器模式 (Adapter Pattern)。
- 批量优化: 一次 API 调用处理多个文本，减少网络开销和 API 限制。
  batch_size=100 是 API 限制和延迟的平衡点。
- Contextual Retrieval 实现: 在 embed() 内部，为每个 chunk prepend 文档上下文
  摘要。这是 2026 年最高 ROI 的单点改进 (Anthropic 2024: -49% 检索失败)。

业务难点:
- API 速率限制: OpenAI/voyage 都有 RPM (Requests Per Minute) 限制。
  解决方案: 使用 exponential backoff + batch_size 限制。
- Embedding 维度对齐: 不同模型的默认维度不同（1024 vs 1536 vs 3072），
  必须与 vector_db.vector_size 和 reranker 配置严格一致。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """
    Embedding 后端抽象基类

    设计模式: 适配器模式 (Adapter Pattern)
    - 将不同 provider 的 API 封装为统一的 EmbeddingBackend 接口
    - 新增 provider 只需继承此类实现 embed_batch()
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """单个文本的 embedding"""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本的 embedding"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度（必须与 vector_db 配置一致）"""
        ...


class VoyageEmbedder(EmbeddingBackend):
    """
    Voyage AI Embedding 后端

    技术决策:
    - voyage-3-large: 2026 年 MTEB Retrieval 英文场景最高分 (~62)
    - 专用变体: voyage-3-legal / voyage-3-finance / voyage-code-3
    - 优势: API 简单，无速率限制说明文档友好
    - API key: 从 VOYAGE_API_KEY 环境变量读取
    """

    def __init__(
        self,
        model: str = "voyage-3-large",
        batch_size: int = 100,
        truncate_input: bool = True,
    ):
        try:
            import voyageai
        except ImportError:
            raise ImportError("需要安装 voyageai: pip install voyageai")

        self._client = voyageai
        self._model = model
        self._batch_size = batch_size
        self._truncate = truncate_input

    def embed(self, text: str) -> list[float]:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            # Voyage AI 支持自动长截断，无需手动处理
            response = self._client.TextEmbedding.create(
                input=batch,
                model=self._model,
                truncation=self._truncate,
            )
            # 处理 single 和 batch 两种响应格式
            if isinstance(response, list):
                embeddings = [r.embedding for r in response]
            else:
                embeddings = [response.embedding]
            all_embeddings.extend(embeddings)
            time.sleep(0.05)  # 简单速率保护
        return all_embeddings

    @property
    def dimension(self) -> int:
        return 1024  # voyage-3-large 固定 1024 维


class OpenAIEmbedder(EmbeddingBackend):
    """
    OpenAI Embedding 后端

    技术决策:
    - text-embedding-3-large: 通用场景默认值，MTEB ~59，支持 Matryoshka 截断
    - text-embedding-3-small: 成本敏感场景，$0.02/1M tokens，MTEB ~55
    - Matryoshka Representation Learning (MRL): 可将 3072 维截断到 1536 维，
      存储成本减半，检索精度仅下降 1-2%。这是 2026 年标配优化。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        dimensions: int | None = None,  # Matryoshka 截断维度
        normalize: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要安装 openai: pip install openai")

        self._client = OpenAI()
        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions  # None = 使用模型原始维度
        self._normalize = normalize

        # 根据模型确定基础维度
        self._base_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-3-medium": 1024,
        }.get(model, 1024)

    def embed(self, text: str) -> list[float]:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            kwargs: dict = {
                "model": self._model,
                "input": batch,
                "normalization": self._normalize,
            }
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions

            response = self._client.embeddings.create(**kwargs)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            time.sleep(0.05)
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimensions or self._base_dimensions


class BGEEmbedder(EmbeddingBackend):
    """
    BGE (BAAI General Embedding) 开源自托管后端

    技术决策:
    - BGE-M3: Apache 2.0 开源，568M 参数，支持 100+ 语言
    - 独特优势: 单次 forward 同时输出 dense + sparse + multi-vector (ColBERT 风格)
    - MTEB Retrieval ~58，开源自托管首选
    - 适用场景: 多语言语料、合规要求不允许 API 调用、>500K 查询/月成本优化
    """

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: str = "cpu",  # "cuda" 或 "cpu"
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")

        self._model_name = model
        # device_map="auto" 自动选择 GPU/CPU
        self._model = SentenceTransformer(model, device=device)
        self._batch_size = batch_size
        self._normalize = normalize

    def embed(self, text: str) -> list[float]:
        result = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return result.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return results.tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class Embedder:
    """
    Embedding 统一入口 — 支持多后端切换

    设计模式: 门面模式 (Facade Pattern) + 工厂模式
    - 对外提供统一的 embed() 和 embed_with_context() 接口
    - 内部委托给具体后端实现
    - 通过 config.backend 选择后端，支持热切换

    Contextual Retrieval 核心实现:
    - embed_with_context(): 在 embedding 前，用轻量 LLM 为整个文档生成摘要，
      然后 prepend 到每个 chunk 前。
    - Anthropic 2024 研究: 此方法减少 49% 检索失败，叠加 Reranker 后达 67%。

    权衡取舍:
    - 成本: 每个文档额外调用一次轻量 LLM（Haiku 成本 $0.8/1M tokens，可忽略）
    - 延迟: 索引时间略有增加，但检索质量提升远超成本
    - 决策: 高价值文档必须使用 contextual embedding；临时测试可用普通 embedding
    """

    def __init__(
        self,
        backend: Literal["voyage", "openai", "bge"] = "voyage",
        contextual_llm_client=None,
        contextual_prefix_tokens: int = 80,
    ):
        self._backend = self._create_backend(backend)
        self._contextual_llm = contextual_llm_client
        self._prefix_tokens = contextual_prefix_tokens
        self._doc_summaries: dict[str, str] = {}

    def _create_backend(self, backend: str) -> EmbeddingBackend:
        if backend == "voyage":
            return VoyageEmbedder()
        elif backend == "openai":
            return OpenAIEmbedder()
        elif backend == "bge":
            return BGEEmbedder()
        else:
            raise ValueError(f"不支持的 embedding backend: {backend}")

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    def embed(self, text: str) -> list[float]:
        """普通 embedding（无上下文增强）"""
        return self._backend.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding"""
        return self._backend.embed_batch(texts)

    def embed_with_context(
        self,
        chunk_text: str,
        doc_summary: str,
        doc_id: str | None = None,
    ) -> list[float]:
        """
        Contextual Retrieval — 为 chunk 添加文档级上下文后再 embedding

        技术细节:
        - Anthropic 2024 contextual retrieval 的核心方法
        - 步骤: (1) 取 doc_summary (2) 构建 "文档摘要 + chunk 文本" (3) embedding
        - doc_summary 需要在索引阶段由 generate_doc_summary() 生成并缓存

        Args:
            chunk_text: 原始 chunk 文本
            doc_summary: 文档级上下文摘要（50-100 tokens）
            doc_id: 可选，用于缓存

        Returns:
            带上下文的 embedding 向量
        """
        # 构建带上下文的文本
        # 格式: "[Document Summary] {doc_summary}\n\n[Chunk] {chunk_text}"
        context_text = (
            f"[Document Context]\n{doc_summary}\n\n"
            f"[Content]\n{chunk_text}"
        )
        return self._backend.embed(context_text)

    def embed_chunks_with_context(
        self,
        chunks: list,
        doc_summary: str,
    ) -> list[list[float]]:
        """
        批量 contextual embedding

        技术要点:
        - 每个 chunk 都 prepend 同一个 doc_summary
        - 批量调用底层 embed_batch，节省 API 开销
        """
        texts = [
            (
                f"[Document Context]\n{doc_summary}\n\n"
                f"[Content]\n{chunk.text}"
            )
            for chunk in chunks
        ]
        return self.embed_batch(texts)

    async def generate_doc_summary(self, doc_text: str, doc_id: str) -> str:
        """
        为文档生成摘要（用于 Contextual Retrieval）

        技术决策:
        - 使用 Haiku 4.5 生成摘要: $0.8/1M tokens，成本可忽略
        - 摘要长度: 50-100 tokens（约 200-400 字符），足够提供上下文又不喧宾夺主
        - doc_id 作为 key 缓存摘要，避免重复生成

        提示词设计:
        - 核心原则: 生成一个中立、描述性的摘要，
          不要引入 chunk 中不存在的信息（避免幻觉）
        """
        if doc_id in self._doc_summaries:
            return self._doc_summaries[doc_id]

        if self._contextual_llm is None:
            # 没有 LLM client 时，返回空字符串（降级到无上下文 embedding）
            logger.warning("Contextual LLM 未配置，跳过文档摘要生成")
            return ""

        prompt = (
            "You are a document summarizer. Generate a concise summary of the following document.\n"
            "The summary should be 3-5 sentences and capture the main topic and purpose.\n"
            "Do not invent information not present in the document.\n\n"
            f"Document:\n{doc_text[:8000]}"  # 限制输入长度
        )

        try:
            response = await self._contextual_llm.generate(
                prompt,
                max_tokens=self._prefix_tokens,
                temperature=0.1,
            )
            summary = response.strip()
            self._doc_summaries[doc_id] = summary
            logger.debug(f"生成文档摘要 (doc_id={doc_id}, tokens={len(summary.split())})")
            return summary
        except Exception as e:
            logger.warning(f"文档摘要生成失败: {e}")
            return ""

    def get_cached_summary(self, doc_id: str) -> str:
        """获取已缓存的文档摘要"""
        return self._doc_summaries.get(doc_id, "")
