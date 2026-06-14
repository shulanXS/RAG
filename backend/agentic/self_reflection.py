"""
self_reflection.py — 答案自我反思

P1-1: 从 orchestrator.py 头部提取。
P0-3: 仅对 MODERATE / COMPLEX 路径触发，SIMPLE 路径不再调用（节省 200-500ms LLM 调用）。
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


_REFLECTION_PROMPT = """你是一个答案质量审查专家。请对以下答案进行严格检查。

用户问题: {query}

检索到的上下文:
{context}

初始答案:
{answer}

请以 JSON 格式输出:
{{
  "overall_score": 0.0-1.0,
  "needs_more_retrieval": true|false,
  "requires_correction": true|false,
  "gaps": ["缺口1", "缺口2"],
  "hallucinated_claims": ["幻觉陈述1"]
}}"""


async def do_reflection(
    llm_client,
    query: str,
    answer: str,
    contexts: list[dict],
) -> tuple[float, list[str], str, bool]:
    """
    自我反思：评估答案质量，返回 (score, gaps, revised_answer, needs_correction)。

    Returns:
        (overall_score 0-1, gaps list, revised_answer, needs_correction)
    """
    if not contexts or not answer:
        return 0.5, [], answer, False

    ctx_text = "\n".join(
        f"[{i+1}] {c.get('text', '')[:300]}"
        for i, c in enumerate(contexts[:10])
    )

    prompt = _REFLECTION_PROMPT.format(
        query=query,
        context=ctx_text,
        answer=answer,
    )

    try:
        response = await llm_client.generate_async(prompt, max_tokens=512, temperature=0.1)
        data = json.loads(response.strip())
        score = float(data.get("overall_score", 0.5))
        needs_correction = data.get("requires_correction", False)
        gaps = data.get("gaps", [])

        revised = answer
        if needs_correction and score < 0.5:
            revise_prompt = f"""基于以下审查意见修正答案。

原始问题: {query}
原答案: {answer}

审查发现的问题: {', '.join(gaps[:3])}

上下文: {ctx_text}

要求: 只基于上下文修正，不要引入外部知识。如有无法回答的方面，明确标注。

修正后的答案:"""
            revised = await llm_client.generate_async(revise_prompt, max_tokens=1024, temperature=0.2)

        return score, gaps, revised, needs_correction
    except Exception as e:
        logger.warning(f"Self-reflection failed: {e}")
        return 0.5, [], answer, False
