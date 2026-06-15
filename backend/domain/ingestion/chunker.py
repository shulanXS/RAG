"""
chunker.py — 智能分块策略
================================================================================
技术决策记录:
- 为什么需要分块: LLM 的上下文窗口虽然越来越长，但向量化检索必须依赖固定
  维度的向量。一个长文档无法直接 embedding，需要切分为语义完整的块。
- 分块策略选择:
  (1) 固定分块 (fixed): 最简单但质量最差，会在句子中间断开
  (2) 递归字符分块 (recursive): 按 "\n\n" → "\n" → ". " 逐级递归，
      接近语义边界，工业界最常用
  (3) 层级分块 (hierarchical): 保留文档结构（SOP/政策文档首选）
  (4) 语义分块 (semantic): embedding 相似度检测断点，精度最高但成本也最高

业务难点:
- 块太小: 上下文不足，无法回答需要跨段落理解的问题
- 块太大: 语义稀释，检索精度下降，且可能超过 LLM 窗口
- 跨块引用: 「第三章讨论的X观点」的检索，跨越了块边界

解决方案:
- 重叠分块 (overlap): chunk_overlap=64 在块之间创建上下文桥接
- 父子块关系: metadata 记录 parent_doc_id，支持回溯到更大上下文
- 最小块阈值: min_chunk_size=150 过滤碎片化块

权衡取舍:
- 语义分块精度最高，但需要为每个句子调用 embedding 模型，成本极高。
  决策: 仅对高价值文档（合同/协议/核心政策）使用语义分块，普通文档用递归分块。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# =============================================================================
# 1. 数据结构
# =============================================================================


@dataclass
class Chunk:
    """
    分块结果 — 对应向量数据库中的一个向量记录

    字段说明:
    - chunk_id: 全局唯一标识符，格式: {doc_id}_chunk_{index}
    - doc_id: 父文档 ID
    - text: 块文本内容（未拼接上下文前缀）
    - token_count: 估算 token 数（用于质量监控）
    - section_path: 文档中的层级路径，如 "第一章 / 第一节 / 核心概念"
    - chunk_index: 在文档中的顺序（用于父子块关系）
    - metadata: 扩展信息（来源、ACL 标签等）
    """
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    section_path: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_size_category(self) -> Literal["tiny", "small", "medium", "large"]:
        """按 token 数量分类，用于监控"""
        if self.token_count < 150:
            return "tiny"
        elif self.token_count < 350:
            return "small"
        elif self.token_count < 600:
            return "medium"
        return "large"


@dataclass
class ChunkResult:
    """
    文档分块结果

    统计字段用于监控:
    - num_chunks: 总块数
    - num_kept: 有效块数（≥ min_chunk_size）
    - num_dropped: 被丢弃的碎片块数
    - avg_token_per_chunk: 平均块大小
    """
    doc_id: str
    chunks: list[Chunk]
    num_kept: int = 0
    num_dropped: int = 0
    avg_token_per_chunk: float = 0.0

    def __post_init__(self):
        self.num_kept = len(self.chunks)
        self.avg_token_per_chunk = (
            sum(c.token_count for c in self.chunks) / len(self.chunks)
            if self.chunks else 0.0
        )


# =============================================================================
# 2. Token 计数器
# =============================================================================

def count_tokens(text: str) -> int:
    """
    估算 token 数量

    技术决策:
    - 使用 tiktoken (OpenAI BPE) 估算: 比 4 * char 规则更准确，
      与 embedding 模型的实际 tokenizer 更接近。
    - 对中文: tiktoken 对中文的估算略偏低，但误差在可接受范围内。
    - 备选方案: 如果 tiktoken 不可用，回退到 0.25 * char（对中文较准）。
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception:
        # Fallback: 经验公式（中文约 0.5 char/token，英文约 4 char/token）
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 0.5 + other_chars * 0.25)


# =============================================================================
# 3. 分块策略基类
# =============================================================================

class ChunkingStrategy(ABC):
    """
    分块策略抽象基类

    设计模式: 策略模式 (Strategy Pattern)
    - 不同的分块算法封装为独立策略类
    - 客户端代码通过 get_chunker() 工厂函数获取策略，不直接依赖具体类
    - 新增分块策略只需实现 split 方法，不影响现有代码

    这是 2026 年生产代码的必备设计思维: 策略模式 + 工厂模式的组合。
    """

    @abstractmethod
    def split(
        self,
        text_units: list[str],
        doc_id: str,
        metadata: dict,
    ) -> list[Chunk]:
        """对文档文本进行分块"""
        ...

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        index: int,
        section_path: str = "",
        extra_metadata: dict | None = None,
    ) -> Chunk:
        """创建 Chunk 实例的工厂方法

        P3.1: 删 parent_doc_summary 参数(plan §3.3),运行时无消费方,
        原值来自 metadata["doc_summary"],由 context-aware embedder 注入 text。
        """
        token_count = count_tokens(text)
        chunk_id = f"{doc_id}_chunk_{index}"
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            token_count=token_count,
            section_path=section_path,
            chunk_index=index,
            metadata=extra_metadata or {},
        )


class RecursiveChunker(ChunkingStrategy):
    """
    递归字符分块 — 工业界最常用的默认方案

    技术细节:
    - 按分隔符列表逐级尝试拆分: ["\n\n", "\n", ". ", " "]
    - 每次尝试用当前分隔符拆分，如果块仍然太大，递归到下一个分隔符
    - 重叠: 在块边界处保留 overlap tokens 的上下文，确保语义连续性

    权衡:
    - vs 固定分块: 不会在句子中间断开（大多数情况）
    - vs 语义分块: 无需逐句 embedding，成本低 10-100 倍
    - 缺点: 边界仍然是启发式的，不是真正的语义边界
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 150,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        # 分隔符优先级: 空行 > 换行 > 句号 > 空格
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split(
        self,
        text_units: list[str],
        doc_id: str,
        metadata: dict,
    ) -> list[Chunk]:
        full_text = "\n\n".join(text_units)

        raw_chunks = self._recursive_split(full_text)
        chunks: list[Chunk] = []

        for i, text in enumerate(raw_chunks):
            token_count = count_tokens(text)
            if token_count < self.min_chunk_size:
                # 尝试合并到前一个块
                if chunks and (token_count + chunks[-1].token_count) < self.chunk_size * 1.5:
                    chunks[-1].text += "\n\n" + text
                    chunks[-1].token_count = count_tokens(chunks[-1].text)
                continue

            section_path = metadata.get("section_path", "")
            chunk = self._make_chunk(
                text=text,
                doc_id=doc_id,
                index=i,
                section_path=section_path,
                extra_metadata={"chunking_strategy": "recursive"},
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str) -> list[str]:
        """递归地按分隔符拆分文本"""
        if count_tokens(text) <= self.chunk_size:
            return [text]

        for separator in self.separators:
            if separator not in text:
                continue

            splits = text.split(separator)
            parts: list[str] = []
            current = ""

            for part in splits:
                test = current + separator + part if current else part
                if count_tokens(test) <= self.chunk_size:
                    current = test
                else:
                    if current:
                        parts.append(current.strip())
                    # 如果单个 part 本身就超过 chunk_size，递归处理
                    if count_tokens(part) > self.chunk_size:
                        nested = self._recursive_split(part)
                        parts.extend(nested[:-1])
                        current = nested[-1] if nested else ""
                    else:
                        current = part

            if current:
                parts.append(current.strip())

            # 如果拆分有效（产生了多个块），返回结果
            if len(parts) > 1:
                return [p for p in parts if p.strip()]
            # 否则尝试下一个分隔符

        # 无法拆分，直接截断
        tokens = text.split()
        return [" ".join(tokens[: self.chunk_size * 2])]


# =============================================================================
# 4. 工厂函数
# =============================================================================

def get_chunker(
    strategy: Literal["recursive"],
    config: dict,
    embedder=None,
) -> ChunkingStrategy:
    """
    分块策略工厂函数

    P1-3: 移除 HierarchicalChunker / SemanticChunker 分支。
    config.yaml 默认 strategy=recursive；其他值显式抛错。
    """
    chunk_size = config.get("chunk_size", 512)
    overlap = config.get("chunk_overlap", 64)
    min_size = config.get("min_chunk_size", 150)

    if strategy == "recursive":
        return RecursiveChunker(chunk_size, overlap, min_size)
    else:
        # 显式报错，避免静默退化
        raise ValueError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Expected 'recursive' (HierarchicalChunker / SemanticChunker 已删除)."
        )
