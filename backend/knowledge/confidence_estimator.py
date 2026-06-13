"""
confidence_estimator.py — Knowledge Confidence Estimation (知识置信度量化)
================================================================================
技术决策记录:
- 将简单的 "Beyond KB" 二元分类升级为多维置信度量化。
- 置信度分层:
  - high (≥0.9): 强知识自信 → 直接回答
  - medium (0.6-0.9): 中等自信 → 回答 + 标注来源
  - low (0.3-0.6): 低自信 → 回答 + 说明局限性
  - insufficient (<0.3): 知识不足 → 建议外部搜索

多维度置信度评估:
- coverage: 上下文对问题的覆盖程度
- consistency: 多来源一致性
- freshness: 知识时效性
- authority: 来源权威性

技术方案:
- 使用 DeepSeek 分析检索覆盖度
- 检查 chunk 的时效性标签
- 评估来源一致性
- 返回多维置信度
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeConfidence:
    """
    知识置信度

    字段说明:
    - overall: 综合置信度 (0-1)
    - coverage: 上下文对问题的覆盖程度
    - consistency: 多来源一致性
    - freshness: 知识时效性
    - authority: 来源权威性
    - level: 置信度级别 (high/medium/low/insufficient)
    - recommendations: 置信度不足时的建议
    """
    overall: float
    coverage: float
    consistency: float
    freshness: float
    authority: float
    level: str = "medium"
    recommendations: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 3),
            "coverage": round(self.coverage, 3),
            "consistency": round(self.consistency, 3),
            "freshness": round(self.freshness, 3),
            "authority": round(self.authority, 3),
            "level": self.level,
            "recommendations": self.recommendations,
            "gaps": self.gaps,
        }


class KnowledgeConfidenceEstimator:
    """
    多维度知识置信度评估器

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Coverage Analysis (DeepSeek)                             │
    │     分析检索上下文是否充分覆盖问题                            │
    │  2. Consistency Check                                          │
    │     检查多个检索来源是否一致                                  │
    │  3. Freshness Evaluation                                       │
    │     检查文档时效性（通过 metadata 或内容）                   │
    │  4. Authority Assessment                                       │
    │     评估来源权威性                                            │
    │  5. Overall Confidence Computation                            │
    │     综合各维度得分                                            │
    └─────────────────────────────────────────────────────────────┘

    设计要点:
    - 使用 DeepSeek 的 deepseek-chat 进行分析
    - 每个维度独立评估，最后加权汇总
    - 置信度分层用于决定回答策略
    """

    def __init__(
        self,
        llm_client=None,
        coverage_weight: float = 0.4,
        consistency_weight: float = 0.2,
        freshness_weight: float = 0.2,
        authority_weight: float = 0.2,
    ):
        """
        Args:
            llm_client: DeepSeek LLM client
            coverage_weight: 覆盖度权重
            consistency_weight: 一致性权重
            freshness_weight: 时效性权重
            authority_weight: 权威性权重
        """
        self._llm = llm_client
        self._weights = {
            "coverage": coverage_weight,
            "consistency": consistency_weight,
            "freshness": freshness_weight,
            "authority": authority_weight,
        }

    async def estimate(
        self,
        query: str,
        chunks: list[dict],
        answer: str | None = None,
    ) -> KnowledgeConfidence:
        """
        评估知识置信度

        Args:
            query: 用户查询
            chunks: 检索到的 chunks
            answer: 生成的答案（可选，用于辅助评估）

        Returns:
            KnowledgeConfidence: 多维置信度评估结果
        """
        if not chunks:
            return KnowledgeConfidence(
                overall=0.0,
                coverage=0.0,
                consistency=0.0,
                freshness=0.0,
                authority=0.0,
                level="insufficient",
                recommendations=["检索结果为空，建议尝试其他查询词或扩大搜索范围"],
                gaps=["无相关文档"],
            )

        coverage = await self._estimate_coverage(query, chunks)
        consistency = self._estimate_consistency(chunks)
        freshness = self._estimate_freshness(chunks)
        authority = self._estimate_authority(chunks)

        overall = (
            coverage * self._weights["coverage"] +
            consistency * self._weights["consistency"] +
            freshness * self._weights["freshness"] +
            authority * self._weights["authority"]
        )

        level, recommendations = self._determine_level(overall, coverage, consistency)

        return KnowledgeConfidence(
            overall=min(1.0, max(0.0, overall)),
            coverage=coverage,
            consistency=consistency,
            freshness=freshness,
            authority=authority,
            level=level,
            recommendations=recommendations,
            gaps=self._identify_gaps(query, chunks),
        )

    async def _estimate_coverage(
        self,
        query: str,
        chunks: list[dict],
    ) -> float:
        """评估上下文对问题的覆盖程度"""
        if not chunks:
            return 0.0

        if self._llm is None:
            return self._rule_based_coverage(query, chunks)

        import json

        context_text = self._format_chunks(chunks)

        prompt = f"""请评估检索上下文对用户问题的覆盖程度。

用户问题: {query}

检索上下文:
{context_text}

请分析:
1. 问题有几个主要方面？
2. 每个方面是否都有对应的上下文支撑？
3. 上下文是否直接回答了问题，还是只是部分相关？

请以 JSON 格式输出:
{{
  "coverage_score": 0.0-1.0,
  "covered_aspects": ["方面1", "方面2"],
  "uncovered_aspects": ["未覆盖的方面"],
  "reasoning": "分析理由"
}}"""

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

            return float(data.get("coverage_score", 0.5))

        except Exception as e:
            logger.warning(f"Coverage estimation failed: {e}")
            return self._rule_based_coverage(query, chunks)

    def _rule_based_coverage(self, query: str, chunks: list[dict]) -> float:
        """基于规则的覆盖度评估（降级方案）"""
        query_keywords = set(query.lower().split())
        covered_keywords = 0

        for chunk in chunks:
            text = chunk.get("text", "").lower()
            for kw in query_keywords:
                if len(kw) > 2 and kw in text:
                    covered_keywords += 1

        if not query_keywords:
            return 0.5

        return min(1.0, covered_keywords / len(query_keywords) * 0.5 + 0.3)

    def _estimate_consistency(self, chunks: list[dict]) -> float:
        """评估多来源一致性"""
        if len(chunks) <= 1:
            return 0.8

        texts = [c.get("text", "")[:200] for c in chunks[:5]]

        contradiction_keywords = [
            "但是", "然而", "相反", "however", "but", "conversely",
            "不一致", "矛盾", "different from",
        ]

        doc_ids = [c.get("doc_id", "") for c in chunks]
        if len(set(doc_ids)) == 1:
            return 0.7

        contradiction_count = sum(
            1 for text in texts
            if any(kw in text for kw in contradiction_keywords)
        )

        if contradiction_count > 2:
            return 0.3
        elif contradiction_count > 0:
            return 0.5

        return 0.8

    def _estimate_freshness(self, chunks: list[dict]) -> float:
        """评估知识时效性"""
        import re
        from datetime import datetime

        current_year = datetime.now().year

        freshness_scores = []

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            text = chunk.get("text", "")[:500]

            score = 0.7

            created_at = metadata.get("created_at", "")
            if created_at:
                try:
                    year = int(re.search(r"\d{4}", str(created_at)).group())
                    if abs(year - current_year) <= 1:
                        score = 0.9
                    elif abs(year - current_year) <= 3:
                        score = 0.7
                    else:
                        score = 0.4
                except (ValueError, AttributeError):
                    pass

            date_patterns = [
                r"(20\d{2})年", r"(20\d{2})-(0[1-9]|1[0-2])",
                r"Q[1-4]\s*(20\d{2})",
            ]
            for pattern in date_patterns:
                dates = re.findall(pattern, text)
                if dates:
                    try:
                        year = int(dates[0]) if len(dates[0]) == 4 else int(dates[0][-4:])
                        if abs(year - current_year) <= 1:
                            score = max(score, 0.9)
                        elif abs(year - current_year) <= 3:
                            score = max(score, 0.7)
                    except (ValueError, IndexError):
                        pass

            freshness_scores.append(score)

        if not freshness_scores:
            return 0.5

        return sum(freshness_scores) / len(freshness_scores)

    def _estimate_authority(self, chunks: list[dict]) -> float:
        """评估来源权威性"""
        authority_keywords = {
            "high": [
                "官方", "政府", "法律", "regulation", "policy", "official",
                "年度报告", "公告", "白皮书",
            ],
            "low": [
                "论坛", "博客", "社交媒体", "forum", "blog", "social media",
            ],
        }

        scores = []

        for chunk in chunks:
            text = chunk.get("text", "")[:300].lower()
            metadata = chunk.get("metadata", {})

            score = 0.6

            source = metadata.get("source", "").lower()
            if any(kw in source for kw in authority_keywords["high"]):
                score = 0.9
            elif any(kw in source for kw in authority_keywords["low"]):
                score = 0.3

            if metadata.get("is_parent"):
                score = max(score, 0.7)

            doc_type = metadata.get("doc_type", "").lower()
            if any(t in doc_type for t in ["report", "policy", "contract", "agreement"]):
                score = max(score, 0.85)

            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.5

    def _determine_level(
        self,
        overall: float,
        coverage: float,
        consistency: float,
    ) -> tuple[str, list[str]]:
        """根据综合分数确定置信度级别"""
        recommendations = []

        if overall >= 0.9:
            level = "high"
            recommendations.append("置信度高，可直接回答")
        elif overall >= 0.6:
            level = "medium"
            if coverage < 0.7:
                recommendations.append("部分问题方面未充分覆盖，回答时标注来源")
            if consistency < 0.5:
                recommendations.append("存在不一致信息，请综合考虑多个来源")
        elif overall >= 0.3:
            level = "low"
            recommendations.append("置信度较低，回答需说明局限性")
            recommendations.append("建议用户提供更多背景信息以缩小范围")
        else:
            level = "insufficient"
            recommendations.append("知识不足，建议使用外部搜索")
            recommendations.append("或请用户重新描述问题")

        return level, recommendations

    def _identify_gaps(self, query: str, chunks: list[dict]) -> list[str]:
        """识别知识缺口"""
        gaps = []

        if len(chunks) < 2:
            gaps.append("检索结果数量过少，可能遗漏相关信息")

        doc_ids = [c.get("doc_id", "") for c in chunks]
        if len(set(doc_ids)) == 1 and len(doc_ids) > 3:
            gaps.append("检索结果集中在单一文档，可能视角单一")

        total_text = " ".join(c.get("text", "") for c in chunks[:3])
        if len(total_text) < 300:
            gaps.append("检索文本长度较短，信息量可能不足")

        return gaps

    def _format_chunks(self, chunks: list[dict]) -> str:
        """格式化 chunks 文本"""
        parts = []
        for i, chunk in enumerate(chunks[:8], 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            text = chunk.get("text", "")[:250]
            doc_id = chunk.get("doc_id", "")
            parts.append(f"[{chunk_id}] (from {doc_id})\n{text}")
        return "\n\n".join(parts)
