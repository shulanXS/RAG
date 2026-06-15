"""
self_rag.py — Self-RAG / CRAG 质量兜底(plan §2.2)
================================================================================
2026 FAANG 标准:
- Self-RAG (Asai et al., 2023 ACL): 生成后用 LLM 自评 confidence + 是否需要 fallback
- CRAG (Yan et al., 2024): 三分支(CORRECT / INCORRECT / AMBIGUOUS)
  替代"LLM 说什么就信什么"的脆弱路径

P3.1(plan §2.2) 关键决策:
- 只对 SIMPLE / MODERATE 触发,COMPLEX 不触发(已经 5 步 ReAct,再 retry 成本太高)
- 不接 Web search(避免引入新外部依赖,作品集 demo 跑不通风险大)
- 硬编码 max_refine_attempts=1(避免无限 retry)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    """CRAG 三分支判定"""
    CORRECT = "correct"      # 高置信度 + context 支撑 → 直接返回
    INCORRECT = "incorrect"  # 低置信度 → 触发 refine(改写 query 再检索)
    AMBIGUOUS = "ambiguous"  # 信息不足 → 加引用 + 显式提示


@dataclass
class SelfRAGResult:
    """Self-RAG 评估结果"""
    verdict: Verdict
    confidence: float
    answer: str
    citations: list[dict]
    refined: bool = False
    refine_attempts: int = 0


class SelfRAGJudge:
    """Self-RAG 评估器 + 改写重检索兜底

    Usage:
        judge = SelfRAGJudge(
            llm_client=client,
            retrieval_fn=hybrid_search,
            rewrite_fn=rewriter.rewrite_async,
            confidence_threshold=0.85,
        )

        result = await judge.evaluate(
            query="RAG 的核心组件?",
            answer="...",
            chunks=[...],
        )
        if result.verdict == Verdict.CORRECT:
            return result.answer
        else:
            # 已经 retry 过一次
            return result.answer
    """

    def __init__(
        self,
        llm_client,
        retrieval_fn: Callable[[str], Awaitable[list[dict]]] | None = None,
        rewrite_fn: Callable[[str], Awaitable[Any]] | None = None,
        confidence_threshold: float = 0.85,
        max_refine_attempts: int = 1,
    ):
        self._llm = llm_client
        self._retrieve = retrieval_fn
        self._rewrite = rewrite_fn
        self._threshold = confidence_threshold
        self._max_refine = max_refine_attempts

    async def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[dict],
    ) -> SelfRAGResult:
        """评估生成质量,必要时 refine(改写 query 重检索)

        流程(plan §2.2):
        1. LLM 自评: confidence + verdict(CORRECT/INCORRECT/AMBIGUOUS)
        2. CORRECT: 直接返回
        3. INCORRECT: refine(改写 query 再检索一次) — 但 max_refine_attempts=1
        4. AMBIGUOUS: 加"信息不足"显式提示
        """
        confidence, verdict = await self._judge(query, answer, chunks)

        if verdict == Verdict.CORRECT and confidence >= self._threshold:
            return SelfRAGResult(
                verdict=verdict,
                confidence=confidence,
                answer=answer,
                citations=self._extract_citations(chunks),
            )

        if verdict == Verdict.AMBIGUOUS:
            return SelfRAGResult(
                verdict=verdict,
                confidence=confidence,
                answer=f"{answer}\n\n[提示] 信息可能不足,建议补充问题细节。",
                citations=self._extract_citations(chunks),
            )

        # INCORRECT: 尝试 refine
        if self._retrieve is not None and self._rewrite is not None and self._max_refine > 0:
            try:
                new_query = await self._rewrite(query)
                if hasattr(new_query, "rewritten"):
                    new_query = new_query.rewritten
                new_chunks = await self._retrieve(new_query)
                if new_chunks:
                    new_confidence, new_verdict = await self._judge(query, answer, new_chunks)
                    if new_confidence > confidence:
                        return SelfRAGResult(
                            verdict=new_verdict,
                            confidence=new_confidence,
                            answer=answer,
                            citations=self._extract_citations(new_chunks),
                            refined=True,
                            refine_attempts=1,
                        )
            except Exception as e:
                logger.warning(f"Self-RAG refine 失败: {e}")

        # refine 失败或不可用: 返回原结果(降级)
        return SelfRAGResult(
            verdict=verdict,
            confidence=confidence,
            answer=answer,
            citations=self._extract_citations(chunks),
            refined=False,
        )

    async def _judge(
        self,
        query: str,
        answer: str,
        chunks: list[dict],
    ) -> tuple[float, Verdict]:
        """LLM 自评 — 返回 (confidence, verdict)

        LLM 输出 JSON: {"confidence": 0.0-1.0, "verdict": "correct|incorrect|ambiguous",
                        "reason": "..."}
        """
        if self._llm is None:
            return 0.5, Verdict.AMBIGUOUS

        context_brief = "\n".join(
            c.get("text", "")[:200] for c in chunks[:3]
        ) or "(无上下文)"

        prompt = f"""你是一个 RAG 答案质量评估器。请判断下面答案是否充分回答了用户问题。

用户问题: {query}

候选答案: {answer}

检索到的上下文(摘要):
{context_brief}

评估维度:
1. 答案是否直接回应了问题?
2. 答案中的关键事实是否有上下文支撑?
3. 是否需要更多上下文才能给出准确答案?

请以 JSON 格式输出:
{{"confidence": 0.0-1.0, "verdict": "correct|incorrect|ambiguous", "reason": "简短原因"}}

判定规则:
- correct: confidence >= 0.85 且上下文支撑
- incorrect: confidence < 0.5 或答案与上下文冲突
- ambiguous: 其余情况"""

        try:
            import json
            response = await self._llm.generate_async(prompt, max_tokens=256, temperature=0.1)
            parsed = json.loads(response.strip())
            confidence = float(parsed.get("confidence", 0.5))
            verdict_str = str(parsed.get("verdict", "ambiguous")).lower()

            if verdict_str == "correct":
                verdict = Verdict.CORRECT
            elif verdict_str == "incorrect":
                verdict = Verdict.INCORRECT
            else:
                verdict = Verdict.AMBIGUOUS

            return confidence, verdict

        except Exception as e:
            logger.warning(f"Self-RAG judge 异常: {e}")
            return 0.5, Verdict.AMBIGUOUS

    @staticmethod
    def _extract_citations(chunks: list[dict]) -> list[dict]:
        return [
            {
                "doc_id": c.get("doc_id", ""),
                "chunk_id": c.get("chunk_id", ""),
                "section_path": c.get("section_path", ""),
            }
            for c in chunks[:5]
        ]
