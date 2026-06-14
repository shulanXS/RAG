"""
backend/generation/citation_verifier.py — Answer-Citation Grounding Verification

将答案中的每个声明与检索到的 chunk 进行对齐验证，
确保答案中的每个关键陈述都有对应引用支撑。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class VerifiedCitation:
    """经过答实对齐验证的引用"""
    doc_id: str
    chunk_id: str | None
    quote: str
    score: float
    supported_claims: list[str]
    unsupported_claims: list[str]
    is_grounded: bool
    reason: str


_VERIFICATION_PROMPT = """你是一个答案质量审查专家。请验证答案中的每个关键陈述是否被检索上下文支撑。

用户问题: {query}

检索到的上下文:
{context}

AI 生成的答案:
{answer}

任务:
1. 从答案中提取 3-5 个关键陈述（claims）
2. 对每个陈述，判断它是否可以从上下文中找到直接支持
3. 对于有支撑的陈述，标注对应的上下文编号

请以 JSON 格式输出:
{{
  "claims": [
    {{
      "text": "陈述内容",
      "supported": true或false,
      "supporting_chunk_index": 对应上下文编号（从1开始，如果不支持则为null）,
      "reason": "判断理由"
    }}
  ]
}}"""


@dataclass
class CitationVerificationResult:
    """引用验证结果"""
    verified_citations: list[VerifiedCitation]
    overall_groundedness_score: float
    unsupported_claims: list[str]
    num_supported: int
    num_total: int


class CitationVerifier:
    """
    答实对齐验证器

    工作流程:
    1. 将 answer 拆解为独立的声明（claims）
    2. 对每个声明，用 LLM 判断是否被对应 chunk 支撑
    3. 返回声明级别的引用映射 + 总体 groundness score
    """

    def __init__(self, llm_client=None, fallback_citations: list[dict] | None = None):
        """
        Args:
            llm_client: LLM client（用于验证）
            fallback_citations: 无法验证时的降级引用
        """
        self._llm = llm_client
        self._fallback = fallback_citations or []

    async def verify(
        self,
        query: str,
        answer: str,
        chunks: list[dict],
    ) -> CitationVerificationResult:
        """
        验证答案中的声明是否被 chunk 支撑。

        Args:
            query: 用户问题
            answer: 生成的答案
            chunks: 检索到的 chunks

        Returns:
            CitationVerificationResult: 验证结果
        """
        if not self._llm or not answer or not chunks:
            return self._fallback_result(chunks)

        ctx_text = self._format_chunks(chunks)

        prompt = _VERIFICATION_PROMPT.format(
            query=query,
            context=ctx_text,
            answer=answer,
        )

        try:
            response = await self._llm.generate_async(
                prompt,
                max_tokens=1024,
                temperature=0.1,
            )
            data = json.loads(response.strip())
            return self._parse_result(data, chunks)
        except Exception as e:
            logger.warning(f"Citation verification failed: {e}")
            return self._fallback_result(chunks)

    def _format_chunks(self, chunks: list[dict], max_chars: int = 500) -> str:
        """将 chunks 格式化为上下文字符串"""
        lines = []
        for i, chunk in enumerate(chunks[:10], 1):
            text = chunk.get("text", "")[:max_chars]
            lines.append(f"[{i}] {text}")
        return "\n\n".join(lines)

    def _parse_result(
        self,
        data: dict,
        chunks: list[dict],
    ) -> CitationVerificationResult:
        """解析 LLM 返回的 JSON 结果"""
        claims = data.get("claims", [])
        verified = []
        supported_count = 0

        for claim in claims:
            text = claim.get("text", "")
            supported = claim.get("supported", False)
            chunk_idx = claim.get("supporting_chunk_index")
            reason = claim.get("reason", "")

            if supported and chunk_idx is not None:
                idx = chunk_idx - 1
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    verified.append(VerifiedCitation(
                        doc_id=chunk.get("doc_id", ""),
                        chunk_id=chunk.get("chunk_id"),
                        quote=chunk.get("text", "")[:200],
                        score=chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
                        supported_claims=[text],
                        unsupported_claims=[],
                        is_grounded=True,
                        reason=reason,
                    ))
                    supported_count += 1
                else:
                    verified.append(self._unsupported_citation(text, reason))
            else:
                verified.append(self._unsupported_citation(text, reason))

        score = supported_count / len(claims) if claims else 0.0
        unsupported = [c.text for c in verified if not c.is_grounded]

        return CitationVerificationResult(
            verified_citations=verified,
            overall_groundedness_score=score,
            unsupported_claims=unsupported,
            num_supported=supported_count,
            num_total=len(claims),
        )

    def _unsupported_citation(self, claim: str, reason: str) -> VerifiedCitation:
        """创建未支撑的引用对象"""
        return VerifiedCitation(
            doc_id="",
            chunk_id=None,
            quote="",
            score=0.0,
            supported_claims=[],
            unsupported_claims=[claim],
            is_grounded=False,
            reason=reason,
        )

    def _fallback_result(self, chunks: list[dict]) -> CitationVerificationResult:
        """无法验证时的降级：使用简单截取"""
        verified = []
        for chunk in chunks[:5]:
            verified.append(VerifiedCitation(
                doc_id=chunk.get("doc_id", ""),
                chunk_id=chunk.get("chunk_id"),
                quote=chunk.get("text", "")[:200],
                score=chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
                supported_claims=[],
                unsupported_claims=[],
                is_grounded=False,
                reason="fallback: simple extraction",
            ))
        return CitationVerificationResult(
            verified_citations=verified,
            overall_groundedness_score=0.0,
            unsupported_claims=[],
            num_supported=0,
            num_total=len(chunks),
        )

    def to_citations(self, result: CitationVerificationResult) -> list[dict]:
        """将验证结果转换为 API 响应格式"""
        return [
            {
                "doc_id": v.doc_id,
                "chunk_id": v.chunk_id,
                "quote": v.quote,
                "score": v.score,
                "is_grounded": v.is_grounded,
                "supported_claims": v.supported_claims,
                "reason": v.reason,
            }
            for v in result.verified_citations
        ]
