"""
self_reflection.py — Self-Reflection (答案自我反思与修正)
================================================================================
技术决策记录:
- Self-Reflection 是 FAANG RAG 减少幻觉的核心机制。
- 核心思想: LLM 在生成答案后，不直接返回，而是对自己生成的答案进行反思，
  检查是否与检索上下文一致，是否存在信息缺失。
- 与 Grounded Generation 的关系: Grounded 是生成时的约束，
  Self-Reflection 是生成后的验证，两者互补。

反思检查项:
1. 检索覆盖: 上下文是否充分覆盖问题？
2. 幻觉检测: 答案中哪些陈述无法被上下文支撑？
3. 完整性: 答案是否完整回答了问题的所有方面？
4. 一致性: 多个检索来源是否一致？
5. 修正建议: 需要补充检索什么？

技术方案:
- 使用 DeepSeek 的 deepseek-chat 进行反思分析
- 反思结果用于决定是否需要补充检索
- ReflectionResult 包含所有反思维度的评估

业务价值:
- 显著减少幻觉（LLM 生成的不准确信息）
- 提高答案的可信度
- 识别知识缺口，主动告知用户
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class ReflectionCheck:
    """
    单项反思检查结果

    字段说明:
    - check_type: 检查类型
    - passed: 是否通过
    - score: 通过分数 (0-1)
    - issues: 发现的问题列表
    - suggestions: 改进建议
    """
    check_type: str
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ReflectionResult:
    """
    自我反思结果

    字段说明:
    - needs_more_retrieval: 是否需要补充检索
    - requires_correction: 答案是否需要修正
    - overall_score: 整体质量分数 (0-1)
    - checks: 各维度检查结果
    - gaps: 知识缺口列表
    - revised_answer: 修正后的答案（如果有）
    - additional_queries: 需要补充检索的查询
    """
    needs_more_retrieval: bool = False
    requires_correction: bool = False
    overall_score: float = 0.0
    checks: list[ReflectionCheck] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    revised_answer: str = ""
    additional_queries: list[str] = field(default_factory=list)
    hallucinated_claims: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)


class SelfReflection:
    """
    答案自我反思机制

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  Initial Answer + Context + Query                           │
    │                          ↓                                   │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  5-Dimension Self-Reflection                         │    │
    │  │  1. Coverage Check (上下文覆盖度)                     │    │
    │  │  2. Hallucination Detection (幻觉检测)                │    │
    │  │  3. Completeness Check (完整性检查)                   │    │
    │  │  4. Consistency Check (一致性检查)                     │    │
    │  │  5. Citation Check (引用检查)                          │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                          ↓                                   │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  Decision Gate                                          │    │
    │  │  Score ≥ 0.85 → 直接返回                             │    │
    │  │  0.6 ≤ Score < 0.85 → 返回 + 警告                   │    │
    │  │  Score < 0.6 → 补充检索 + 修正答案                   │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘

    设计要点:
    - 每个检查维度独立运行，最后汇总
    - 使用 DeepSeek 的 deepseek-chat 进行分析
    - 反思结果不影响性能：只有在需要修正时才重新生成
    """

    def __init__(
        self,
        llm_client=None,
        score_threshold: float = 0.6,
        reflection_rounds: int = 2,
    ):
        """
        Args:
            llm_client: DeepSeek LLM client
            score_threshold: 低于此阈值需要修正
            reflection_rounds: 最大反思轮数
        """
        self._llm = llm_client
        self._threshold = score_threshold
        self._max_rounds = reflection_rounds

    async def reflect(
        self,
        query: str,
        initial_answer: str,
        contexts: list[dict],
    ) -> ReflectionResult:
        """
        对初始答案进行自我反思

        Args:
            query: 用户原始查询
            initial_answer: LLM 生成的初始答案
            contexts: 检索到的上下文列表

        Returns:
            ReflectionResult: 包含反思结果和修正建议
        """
        if not initial_answer or not initial_answer.strip():
            return ReflectionResult(
                needs_more_retrieval=True,
                requires_correction=True,
                overall_score=0.0,
                gaps=["无答案内容"],
            )

        if self._llm is None:
            return ReflectionResult(
                overall_score=0.5,
                revised_answer=initial_answer,
            )

        try:
            result = await self._perform_reflection(query, initial_answer, contexts)
            return result
        except Exception as e:
            logger.warning(f"Self-reflection failed: {e}")
            return ReflectionResult(
                overall_score=0.5,
                revised_answer=initial_answer,
            )

    async def _perform_reflection(
        self,
        query: str,
        answer: str,
        contexts: list[dict],
    ) -> ReflectionResult:
        """执行多维度反思"""
        import json

        context_text = self._format_contexts(contexts)

        prompt = f"""你是一个答案质量审查专家。请对以下答案进行严格的自我反思检查。

用户问题: {query}

检索到的上下文:
{context_text}

初始答案:
{answer}

请从以下五个维度进行严格检查:

1. **Coverage (覆盖度)**: 检索上下文是否充分覆盖了问题的所有方面？
   - 问题有几个方面？
   - 每个方面是否都有对应的上下文支撑？

2. **Hallucination (幻觉)**: 答案中是否有无法被上下文支撑的陈述？
   - 列出所有「听起来合理但上下文无法证实」的陈述
   - 注意：通用常识（如「太阳从东边升起」）不算幻觉

3. **Completeness (完整性)**: 答案是否完整回答了问题的所有方面？
   - 是否有遗漏的问题点？
   - 是否存在「半截」的回答？

4. **Consistency (一致性)**: 多个检索来源之间是否一致？
   - 是否有矛盾的信息？
   - 哪个来源更可靠？

5. **Citation (引用)**: 答案中的关键陈述是否标注了来源？
   - 每个关键数据/事实是否都能找到对应来源？

请以 JSON 格式输出:
{{
  "coverage": {{
    "passed": true|false,
    "score": 0.0-1.0,
    "issues": ["问题1", "问题2"],
    "missing_aspects": ["遗漏方面1"]
  }},
  "hallucination": {{
    "passed": true|false,
    "score": 0.0-1.0,
    "issues": ["幻觉陈述1"],
    "hallucinated_claims": ["无法证实的陈述"]
  }},
  "completeness": {{
    "passed": true|false,
    "score": 0.0-1.0,
    "issues": ["不完整的地方"]
  }},
  "consistency": {{
    "passed": true|false,
    "score": 0.0-1.0,
    "issues": ["矛盾点"]
  }},
  "citation": {{
    "passed": true|false,
    "score": 0.0-1.0,
    "issues": ["未标注来源的陈述"]
  }},
  "needs_more_retrieval": true|false,
  "requires_correction": true|false,
  "additional_queries": ["补充检索查询1", "补充检索查询2"],
  "gaps": ["知识缺口描述"],
  "overall_score": 0.0-1.0
}}"""

        response = await self._llm.generate(
            prompt,
            max_tokens=1024,
            temperature=0.1,
        )

        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reflection JSON: {e}")
            return ReflectionResult(overall_score=0.5, revised_answer=answer)

        checks = []
        for check_name in ["coverage", "hallucination", "completeness", "consistency", "citation"]:
            check_data = data.get(check_name, {})
            checks.append(ReflectionCheck(
                check_type=check_name,
                passed=check_data.get("passed", False),
                score=float(check_data.get("score", 0.5)),
                issues=check_data.get("issues", []),
                suggestions=check_data.get("issues", []),
            ))

        needs_more = data.get("needs_more_retrieval", False)
        requires_correction = data.get("requires_correction", False)
        overall_score = float(data.get("overall_score", 0.5))

        gaps = data.get("gaps", [])
        additional_queries = data.get("additional_queries", [])
        hallucinated = data.get("hallucination", {}).get("hallucinated_claims", [])
        missing_aspects = data.get("coverage", {}).get("missing_aspects", [])

        revised = answer
        if requires_correction and overall_score < 0.5:
            revised = await self._revise_answer(query, answer, contexts, checks)

        return ReflectionResult(
            needs_more_retrieval=needs_more,
            requires_correction=requires_correction,
            overall_score=overall_score,
            checks=checks,
            gaps=gaps,
            revised_answer=revised,
            additional_queries=additional_queries,
            hallucinated_claims=hallucinated,
            missing_aspects=missing_aspects,
        )

    async def _revise_answer(
        self,
        query: str,
        original_answer: str,
        contexts: list[dict],
        checks: list[ReflectionCheck],
    ) -> str:
        """基于反思结果修正答案"""
        import json

        context_text = self._format_contexts(contexts)
        issues_text = "\n".join(
            f"- {check.check_type}: {', '.join(check.issues[:2])}"
            for check in checks if not check.passed and check.issues
        )

        prompt = f"""你是一个答案修正专家。请根据以下审查意见修正答案。

用户问题: {query}

原始答案:
{original_answer}

审查发现的问题:
{issues_text}

检索到的上下文:
{context_text}

要求:
1. 只基于上下文中的信息修正答案，不要引入外部知识
2. 移除/修正所有被指出为幻觉的陈述
3. 如果某些方面上下文无法回答，明确标注「此信息在提供的文档中未找到」
4. 保持答案结构清晰

请输出修正后的答案:"""

        try:
            revised = await self._llm.generate(
                prompt,
                max_tokens=1024,
                temperature=0.2,
            )
            return revised.strip()
        except Exception as e:
            logger.warning(f"Answer revision failed: {e}")
            return original_answer

    def _format_contexts(self, contexts: list[dict]) -> str:
        """格式化上下文文本"""
        if not contexts:
            return "（无检索上下文）"

        lines = []
        for i, ctx in enumerate(contexts[:10], 1):
            text = ctx.get("text", "")[:300] if isinstance(ctx, dict) else str(ctx)[:300]
            lines.append(f"[{i}] {text}...")
        return "\n".join(lines)
