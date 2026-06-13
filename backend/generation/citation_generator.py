"""
citation_generator.py — Sentence-level Citation Generation (句级引用生成)
================================================================================
技术决策记录:
- 从粗粒度（chunk 级引用）升级到句级引用，显著提升答案的可信度和可追溯性。
- 核心能力:
  1. 句级切分: 将答案按句子拆分，每个句子独立标注来源
  2. 引用提取: 为每个句子匹配最相关的 chunk
  3. 验证: 检查引用的 chunk 是否真的包含对应信息
  4. 渲染: 输出适合前端展示的结构化数据

技术方案:
- 使用 DeepSeek 进行句级引用提取
- CitationValidator 验证引用准确性
- 支持多种输出格式（JSON、前端组件 props）

业务价值:
- 用户可以精确看到每个句子背后的来源
- 提升答案可信度
- 支持点击引用跳转
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SentenceSource:
    """
    句子的引用来源

    字段说明:
    - chunk_id: 来源 chunk ID
    - doc_id: 来源文档 ID
    - quote: 引用的原文
    - relevance_score: 引用相关性分数
    - char_start: 引用在原文中的起始位置
    - char_end: 引用在原文中的结束位置
    """
    chunk_id: str
    doc_id: str
    quote: str
    relevance_score: float = 1.0
    char_start: int = 0
    char_end: int = 0


@dataclass
class SentencedAnswer:
    """
    句级答案

    字段说明:
    - text: 句子文本
    - sources: 该句子的所有引用来源
    - is_citable: 是否可引用（有意义的内容 vs 连接词）
    - is_verified: 引用是否已验证
    """
    text: str
    sources: list[SentenceSource] = field(default_factory=list)
    is_citable: bool = True
    is_verified: bool = False


@dataclass
class CitationGenerationResult:
    """
    Citation Generation 结果

    字段说明:
    - answer: 原始答案
    - sentenced_answer: 句级拆分的答案
    - citations: 所有引用（去重）
    - citation_map: chunk_id → 引用信息
    - verification_results: 验证结果
    """
    answer: str
    sentenced_answer: list[SentencedAnswer]
    citations: list[dict]
    citation_map: dict[str, dict]
    verification_results: dict[str, bool]


class SentenceLevelCitationExtractor:
    """
    句级引用提取器

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 句子切分: 将答案按句子拆分                              │
    │  2. 引用提取: 为每个句子匹配最相关的 chunk                  │
    │  3. 原文匹配: 在 chunk 中找到对应文本片段                  │
    │  4. 验证: 检查引用的 chunk 是否真的包含对应信息             │
    └─────────────────────────────────────────────────────────────┘

    设计要点:
    - 使用正则切分句子（处理中英文）
    - 使用 embedding 相似度匹配 chunk
    - 使用关键词匹配在 chunk 中定位原文
    """

    SENTENCE_SPLIT_PATTERN = re.compile(
        r"(?<=[。！？；\?!])\s*(?=[A-Z\u4e00-\u9fff])|(?<=[.!?;])\s+"
    )

    def __init__(
        self,
        llm_client=None,
        embedder=None,
        min_sentence_len: int = 5,
    ):
        """
        Args:
            llm_client: DeepSeek LLM client（用于智能引用提取）
            embedder: Embedder 实例（用于语义匹配）
            min_sentence_len: 最小句子长度（短句如连接词不引用）
        """
        self._llm = llm_client
        self._embedder = embedder
        self._min_len = min_sentence_len

    async def extract(
        self,
        answer: str,
        contexts: list[dict],
    ) -> CitationGenerationResult:
        """
        从答案中提取句级引用

        Args:
            answer: 原始答案文本
            contexts: 检索到的上下文列表

        Returns:
            CitationGenerationResult: 包含句级答案和引用信息
        """
        if not answer or not contexts:
            return CitationGenerationResult(
                answer=answer,
                sentenced_answer=[],
                citations=[],
                citation_map={},
                verification_results={},
            )

        sentences = self._split_sentences(answer)
        sentenced_answer = self._create_sentenced_answer(sentences)

        citation_map: dict[str, dict] = {}
        verification_results: dict[str, bool] = {}

        for sa in sentenced_answer:
            if not sa.is_citable:
                continue

            sources = await self._find_sources(sa.text, contexts)
            sa.sources = sources

            for src in sources:
                if src.chunk_id not in citation_map:
                    citation_map[src.chunk_id] = {
                        "chunk_id": src.chunk_id,
                        "doc_id": src.doc_id,
                        "quote": src.quote,
                        "appears_in": [],
                    }
                citation_map[src.chunk_id]["appears_in"].append(sa.text[:50])

                if self._llm and src.chunk_id not in verification_results:
                    verified = await self._verify_citation(sa.text, src, contexts)
                    verification_results[src.chunk_id] = verified
                    sa.is_verified = verified

        all_citations = [
            {"chunk_id": cid, "doc_id": v["doc_id"], "quote": v["quote"]}
            for cid, v in citation_map.items()
        ]

        return CitationGenerationResult(
            answer=answer,
            sentenced_answer=sentenced_answer,
            citations=all_citations,
            citation_map=citation_map,
            verification_results=verification_results,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """将文本按句子拆分"""
        sentences = self.SENTENCE_SPLIT_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _create_sentenced_answer(self, sentences: list[str]) -> list[SentencedAnswer]:
        """创建 SentencedAnswer 列表"""
        non_citable_patterns = [
            r"^但是",
            r"^而且",
            r"^因此",
            r"^总之",
            r"^综上所述",
            r"^(然而|不过|同时|另外)$",
            r"^(In summary|However|Moreover|Therefore|Additionally|Also|Finally)$",
        ]

        result = []
        for s in sentences:
            is_citable = len(s) >= self._min_len

            for pattern in non_citable_patterns:
                if re.match(pattern, s, re.IGNORECASE):
                    is_citable = False
                    break

            result.append(SentencedAnswer(text=s, is_citable=is_citable))

        return result

    async def _find_sources(
        self,
        sentence: str,
        contexts: list[dict],
    ) -> list[SentenceSource]:
        """为句子找到最相关的来源"""
        if not contexts or not sentence:
            return []

        if self._llm is None:
            return self._keyword_match(sentence, contexts)

        return await self._llm_match(sentence, contexts)

    async def _llm_match(
        self,
        sentence: str,
        contexts: list[dict],
    ) -> list[SentenceSource]:
        """使用 LLM 进行智能匹配"""
        import json

        context_text = self._format_contexts(contexts)

        prompt = f"""请为以下句子找到最相关的文档引用。

句子: {sentence}

上下文:
{context_text}

请找出:
1. 句子中的关键信息点
2. 每个信息点最匹配的文档片段
3. 匹配度分数 (0.0-1.0)

请以 JSON 格式输出:
{{
  "sources": [
    {{
      "chunk_id": "chunk_xxx",
      "quote": "匹配的原文片段（尽量完整，30字以内）",
      "score": 0.8
    }},
    ...
  ]
}}
如果句子没有涉及具体信息，返回空数组。"""

        try:
            response = await self._llm.generate(
                prompt,
                max_tokens=512,
                temperature=0.1,
            )

            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                data = json.loads(text)

            sources = []
            for item in data.get("sources", [])[:3]:
                chunk_id = item.get("chunk_id", "")
                for ctx in contexts:
                    if ctx.get("chunk_id") == chunk_id or ctx.get("chunk_id", "").startswith(chunk_id.replace("chunk_", "")):
                        sources.append(SentenceSource(
                            chunk_id=chunk_id,
                            doc_id=ctx.get("doc_id", ""),
                            quote=item.get("quote", ""),
                            relevance_score=float(item.get("score", 0.5)),
                        ))
                        break
                else:
                    sources.append(SentenceSource(
                        chunk_id=chunk_id,
                        doc_id="",
                        quote=item.get("quote", ""),
                        relevance_score=float(item.get("score", 0.5)),
                    ))

            return sources

        except Exception as e:
            logger.warning(f"LLM citation matching failed: {e}")
            return self._keyword_match(sentence, contexts)

    def _keyword_match(
        self,
        sentence: str,
        contexts: list[dict],
    ) -> list[SentenceSource]:
        """基于关键词的简单匹配（降级方案）"""
        keywords = re.findall(r"\b\w{2,}\b", sentence.lower())
        keywords = [k for k in keywords if k not in {"的", "是", "在", "了", "和", "与", "the", "a", "an", "is", "are", "and", "or"}]

        best_context = None
        best_score = 0

        for ctx in contexts:
            text = ctx.get("text", "").lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_context = ctx

        if best_context and best_score > 0:
            return [SentenceSource(
                chunk_id=best_context.get("chunk_id", ""),
                doc_id=best_context.get("doc_id", ""),
                quote=best_context.get("text", "")[:100],
                relevance_score=min(1.0, best_score / max(1, len(keywords))),
            )]

        return []

    async def _verify_citation(
        self,
        sentence: str,
        source: SentenceSource,
        contexts: list[dict],
    ) -> bool:
        """验证引用是否准确"""
        import json

        ctx = next((c for c in contexts if c.get("chunk_id") == source.chunk_id), None)
        if not ctx:
            return False

        chunk_text = ctx.get("text", "")[:500]

        prompt = f"""验证以下引用是否准确。

答案句子: {sentence}

引用来源: {source.quote}

文档片段: {chunk_text}

请判断引用的来源是否真的包含句子中的关键信息。回答 "true" 或 "false"。

回答:"""

        try:
            response = await self._llm.generate(
                prompt,
                max_tokens=32,
                temperature=0.0,
            )
            result = response.strip().lower()
            return "true" in result or "是" in result
        except Exception as e:
            logger.warning(f"Citation verification failed: {e}")
            return False

    def _format_contexts(self, contexts: list[dict]) -> str:
        """格式化上下文"""
        parts = []
        for i, ctx in enumerate(contexts[:8], 1):
            chunk_id = ctx.get("chunk_id", f"chunk_{i}")
            text = ctx.get("text", "")[:300]
            parts.append(f"[{chunk_id}] {text}")
        return "\n\n".join(parts)

    def to_json(self, result: CitationGenerationResult) -> dict:
        """将结果转换为适合前端渲染的 JSON 格式"""
        return {
            "answer": result.answer,
            "sentences": [
                {
                    "text": sa.text,
                    "sources": [
                        {
                            "chunk_id": src.chunk_id,
                            "doc_id": src.doc_id,
                            "quote": src.quote,
                            "relevance_score": src.relevance_score,
                            "verified": src.is_verified,
                        }
                        for src in sa.sources
                    ],
                    "is_citable": sa.is_citable,
                }
                for sa in result.sentenced_answer
            ],
            "citations": result.citations,
        }
