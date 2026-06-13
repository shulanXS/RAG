"""
memory_bank.py — Memory Bank（证据追踪记忆库）
================================================================================
技术决策记录:
- 为什么需要 Memory Bank: 传统 RAG 的检索和生成是分开的，无法追踪「答案的
  每个 claim 是从哪个 chunk 推导出来的」。这在监管行业（金融/医疗/法律）
  是致命问题——需要向审计员展示完整的推理链路。
- claim-evidence 链接: 每个 LLM 生成的 claim（主张）必须链接到具体证据。
  证据来自检索结果，Memory Bank 负责存储和管理这个映射关系。
- Section-level admissible evidence: 证据粒度精确到 section 级别，
  而非 document 级别。这是 ADORE 论文的核心贡献。

业务难点:
- claim 提取: 从 LLM 输出中自动提取可验证的主张。
  解决: 使用 LLM 的 JSON Schema 约束输出，强制结构化。
- 证据覆盖度: 如何判断「所有 claim 都有 evidence 支持」？
  解决: Memory Bank 跟踪 claim-evidence 覆盖率，覆盖率 < 100% 时触发告警。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """
    证据单元 — 来自检索结果的具体文本片段

    字段说明:
    - source_id: 证据来源（chunk_id 或 doc_id）
    - text: 具体文本内容
    - doc_title: 来源文档标题（用于展示）
    - section_path: 在文档中的层级路径
    - retrieval_score: 检索相关性得分
    - used_by_claims: 此证据支持的所有 claim IDs
    """
    source_id: str
    text: str
    doc_title: str = ""
    section_path: str = ""
    retrieval_score: float = 0.0
    used_by_claims: list[str] = field(default_factory=list)
    retrieved_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Claim:
    """
    主张单元 — LLM 生成答案中的可验证主张

    字段说明:
    - claim_id: 主张唯一 ID
    - text: 主张文本内容
    - evidence_ids: 支持此主张的证据 IDs
    - verified: 此主张是否已被验证（有证据支撑）
    - confidence: 主张置信度（0-1）
    """
    claim_id: str
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    verified: bool = False
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


class MemoryBank:
    """
    Memory Bank — 可追溯的证据存储库

    设计模式: Repository Pattern
    - 提供统一的 claim 和 evidence 存储/查询接口
    - 与具体存储后端解耦（当前为内存存储，生产可迁移到 Redis/Postgres）

    核心功能:
    1. 添加证据: add_evidence(chunks) — 从检索结果添加
    2. 添加主张: add_claim(claims) — 从 LLM 输出提取
    3. 建立链接: link_claim_evidence(claim_id, evidence_ids) — 建立 claim-evidence 映射
    4. 验证覆盖率: verify_coverage() — 检查所有 claim 是否有 evidence 支撑
    5. 清理过期数据: gc() — 定期清理过期 claim 和 evidence

    业务难点:
    - claim-evidence 匹配质量: 如果 LLM 提取的 claim 与 evidence 语义不匹配，
      会产生错误的映射关系。缓解: 在 link 阶段做语义相似度检查。
    - 多会话隔离: 不同用户的 Memory Bank 需要隔离。
      解决: session_id 作为 namespace key。
    """

    def __init__(
        self,
        session_id: str,
        max_claims: int = 50,
        ttl_hours: int = 24,
    ):
        """
        Args:
            session_id: 会话 ID（用于多会话隔离）
            max_claims: 单个会话最大 claim 数量
            ttl_hours: 数据的生存时间
        """
        self._session_id = session_id
        self._max_claims = max_claims
        self._ttl = timedelta(hours=ttl_hours)

        # 存储结构: claim_id → Claim
        self._claims: dict[str, Claim] = {}
        # evidence_id → Evidence
        self._evidence: dict[str, Evidence] = {}
        # session_id → 创建时间
        self._created_at = datetime.utcnow()

    def add_evidence(self, chunks: list[dict]) -> list[str]:
        """
        从检索结果添加证据

        Args:
            chunks: 检索到的 chunks，格式:
              [{"chunk_id": ..., "text": ..., "doc_id": ..., "rerank_score": ..., ...}]

        Returns:
            添加的 evidence_ids 列表
        """
        evidence_ids = []
        for chunk in chunks:
            evidence_id = f"ev_{chunk.get('chunk_id', '')}"

            if evidence_id in self._evidence:
                # 已存在：更新使用信息
                pass
            else:
                self._evidence[evidence_id] = Evidence(
                    source_id=chunk.get("chunk_id", ""),
                    text=chunk.get("text", "")[:500],  # 截断存储
                    doc_title=chunk.get("doc_title", ""),
                    section_path=chunk.get("section_path", ""),
                    retrieval_score=chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
                )
            evidence_ids.append(evidence_id)

        logger.debug(f"MemoryBank: 添加 {len(evidence_ids)} 个 evidence (session={self._session_id})")
        return evidence_ids

    def add_claims(self, claims: list[dict]) -> list[str]:
        """
        添加主张（从 LLM 输出提取）

        Args:
            claims: 主张列表，格式:
              [{"claim_id": ..., "text": ..., "confidence": ...}]

        Returns:
            添加的 claim_ids 列表
        """
        # 检查 claim 数量限制
        if len(self._claims) >= self._max_claims:
            # 移除最老的 claim
            oldest = min(self._claims.items(), key=lambda x: x[1].created_at)
            del self._claims[oldest[0]]
            logger.debug(f"MemoryBank: claim 数量超限，移除最老的 claim {oldest[0]}")

        claim_ids = []
        for claim_data in claims:
            claim_id = claim_data.get("claim_id", f"claim_{len(self._claims)}")
            claim = Claim(
                claim_id=claim_id,
                text=claim_data.get("text", ""),
                confidence=claim_data.get("confidence", 0.8),
            )
            self._claims[claim_id] = claim
            claim_ids.append(claim_id)

        return claim_ids

    def link_claim_evidence(
        self,
        claim_id: str,
        evidence_ids: list[str],
    ) -> bool:
        """
        建立 claim-evidence 链接

        Args:
            claim_id: 主张 ID
            evidence_ids: 证据 IDs 列表

        Returns:
            是否成功建立链接
        """
        if claim_id not in self._claims:
            logger.warning(f"Claim {claim_id} 不存在，无法建立链接")
            return False

        claim = self._claims[claim_id]
        for ev_id in evidence_ids:
            if ev_id in self._evidence:
                claim.evidence_ids.append(ev_id)
                self._evidence[ev_id].used_by_claims.append(claim_id)

        claim.verified = len(claim.evidence_ids) > 0
        return True

    def verify_coverage(self) -> dict:
        """
        验证 claim-evidence 覆盖率

        Returns:
            覆盖率报告:
            {
                "total_claims": 10,
                "verified_claims": 8,
                "coverage": 0.8,
                "unverified_claims": ["claim_2", "claim_5", ...],
                "unused_evidence": ["ev_xxx", ...],  # 被检索但未被任何 claim 引用的证据
            }
        """
        total = len(self._claims)
        verified = sum(1 for c in self._claims.values() if c.verified)
        coverage = verified / total if total > 0 else 1.0

        unverified = [c.claim_id for c in self._claims.values() if not c.verified]

        # 找出未被任何 claim 引用的 evidence
        used_evidence = set()
        for claim in self._claims.values():
            used_evidence.update(claim.evidence_ids)
        unused_evidence = [eid for eid in self._evidence if eid not in used_evidence]

        return {
            "total_claims": total,
            "verified_claims": verified,
            "coverage": coverage,
            "unverified_claims": unverified,
            "unused_evidence": unused_evidence,
        }

    def get_context_for_generation(self) -> str:
        """
        将 Memory Bank 转换为 LLM 上下文文本

        用于: 将 Memory Bank 内容注入到最终生成的 prompt 中

        格式:
        [Evidence]
        - [doc_title / section_path] (score: x.xx)
          {text}

        [Verified Claims]
        - claim_1: {text}
          ← ev_xxx, ev_yyy

        [Unverified Claims]
        - claim_2: {text}
          ← (无证据支撑)
        """
        lines = ["[Evidence Bank]\n"]

        for ev_id, ev in sorted(self._evidence.items(), key=lambda x: x[1].retrieval_score, reverse=True):
            source = f"{ev.doc_title} / {ev.section_path}" if ev.section_path else ev.doc_title
            lines.append(f"- [{source}] (score: {ev.retrieval_score:.3f})")
            lines.append(f"  {ev.text[:200]}...")

        lines.append("\n[Claim-Evidence Mapping]")
        for claim in sorted(self._claims.values(), key=lambda c: c.confidence, reverse=True):
            status = "✓" if claim.verified else "✗"
            evidence_refs = ", ".join(claim.evidence_ids) if claim.evidence_ids else "(无证据)"
            lines.append(f"- [{status}] {claim.text}")
            lines.append(f"  ← {evidence_refs}")

        return "\n".join(lines)

    def gc(self) -> int:
        """
        垃圾回收：清理过期数据

        Returns:
            清理的条目数量
        """
        now = datetime.utcnow()
        cutoff = now - self._ttl

        removed = 0

        # 清理过期 claim
        expired_claims = [
            cid for cid, c in self._claims.items() if c.created_at < cutoff
        ]
        for cid in expired_claims:
            del self._claims[cid]
            removed += 1

        # 清理无引用的 evidence
        used_evidence = set()
        for claim in self._claims.values():
            used_evidence.update(claim.evidence_ids)

        unused = [eid for eid in self._evidence if eid not in used_evidence]
        for eid in unused:
            del self._evidence[eid]
            removed += 1

        if removed > 0:
            logger.debug(f"MemoryBank GC: 清理 {removed} 个过期条目")
        return removed


@dataclass
class ClaimEvidenceReport:
    """
    Claim-Evidence 链路报告

    字段说明:
    - total_claims: 总 claim 数量
    - verified_claims: 已验证 claim 数量
    - coverage: 覆盖率
    - unverified_claims: 未验证 claim 列表
    - cross_session_claims: 跨会话关联的 claims
    - alerts: 告警信息
    """
    total_claims: int
    verified_claims: int
    coverage: float
    unverified_claims: list[str]
    cross_session_claims: list[dict]
    alerts: list[str]


class AutomatedMemoryBank(MemoryBank):
    """
    自动化 Claim-Evidence 链路

    新增能力:
    1. 自动 claim 提取: 从答案中自动提取可验证主张
    2. 自动证据链接: 语义匹配 claim → evidence
    3. 覆盖率告警: claim 无 evidence 时自动标记
    4. 跨会话追踪: 相同 claim 在多会话中出现时自动关联

    工作流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. process_answer() → 全自动流程入口                      │
    │  2. extract_claims() → 从答案中提取 claims (DeepSeek)      │
    │  3. match_evidence() → 为每个 claim 匹配 evidence           │
    │  4. verify_coverage() → 计算覆盖率                         │
    │  5. cross_session_track() → 跨会话关联                     │
    │  6. generate_report() → 生成报告                          │
    └─────────────────────────────────────────────────────────────┘

    设计要点:
    - 继承 MemoryBank，保持向后兼容
    - 使用 DeepSeek 进行 claim 提取和 evidence 匹配
    - 覆盖率低于阈值时自动告警
    """

    def __init__(
        self,
        session_id: str,
        llm_client=None,
        coverage_threshold: float = 0.8,
        max_claims: int = 50,
        ttl_hours: int = 24,
    ):
        """
        Args:
            session_id: 会话 ID
            llm_client: DeepSeek LLM client
            coverage_threshold: 覆盖率阈值，低于此值触发告警
            max_claims: 最大 claim 数量
            ttl_hours: 数据生存时间
        """
        super().__init__(session_id, max_claims, ttl_hours)
        self._llm = llm_client
        self._coverage_threshold = coverage_threshold
        self._cross_session_store: dict[str, list[str]] = {}

    async def process_answer(
        self,
        answer: str,
        contexts: list[dict],
        query: str = "",
    ) -> ClaimEvidenceReport:
        """
        全自动流程: 从答案处理到报告生成

        Args:
            answer: LLM 生成的答案
            contexts: 检索到的上下文
            query: 原始用户查询（用于辅助 claim 提取）

        Returns:
            ClaimEvidenceReport: 完整的 claim-evidence 链路报告
        """
        if not answer:
            return ClaimEvidenceReport(
                total_claims=0,
                verified_claims=0,
                coverage=1.0,
                unverified_claims=[],
                cross_session_claims=[],
                alerts=[],
            )

        self.add_evidence(contexts)

        claims = await self._extract_claims(answer, query)
        if claims:
            self.add_claims(claims)

        if self._llm:
            await self._auto_link_claims_evidence(claims, contexts)

        coverage_result = self.verify_coverage()

        alerts = self._generate_alerts(coverage_result)

        cross_session = self._track_cross_session(claims)

        return ClaimEvidenceReport(
            total_claims=len(self._claims),
            verified_claims=coverage_result["verified_claims"],
            coverage=coverage_result["coverage"],
            unverified_claims=coverage_result["unverified_claims"],
            cross_session_claims=cross_session,
            alerts=alerts,
        )

    async def _extract_claims(
        self,
        answer: str,
        query: str = "",
    ) -> list[dict]:
        """
        从答案中自动提取可验证主张

        Args:
            answer: LLM 生成的答案
            query: 原始查询

        Returns:
            claim 列表
        """
        if self._llm is None:
            return self._rule_based_claim_extract(answer)

        import json

        prompt = f"""请从以下答案中提取所有可验证的主张（claims）。

要求:
1. 每个主张应该是一个独立的、可验证的陈述
2. 包含具体数据/日期/事实的陈述优先提取
3. 不要提取连接词、过渡句等不可验证的内容
4. 每个主张尽量简洁（20字以内）

答案:
{answer}

请以 JSON 格式输出:
{{
  "claims": [
    {{
      "claim_id": "claim_1",
      "text": "主张内容",
      "confidence": 0.8,
      "has_data": true
    }},
    ...
  ]
}}

如果答案中无可验证主张，返回空数组。"""

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

            claims = data.get("claims", [])
            logger.debug(f"Extracted {len(claims)} claims from answer")
            return claims

        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            return self._rule_based_claim_extract(answer)

    def _rule_based_claim_extract(self, answer: str) -> list[dict]:
        """基于规则的 claim 提取（降级方案）"""
        import re

        claim_patterns = [
            r"([A-Z][^。！？]+?[的是在有个为与和][^。！？]+)",
            r"(\d+[\.．]\d+[^。！？\n]+)",
            r"([^.。！？\n]{10,50}[的是在有与和可应能][^。！？\n]{1,50})",
        ]

        sentences = re.split(r"[。！？\n]", answer)
        claims = []

        for i, s in enumerate(sentences):
            s = s.strip()
            if len(s) < 8:
                continue

            has_number = bool(re.search(r"\d+", s))
            has_keyword = any(kw in s for kw in ["是", "为", "有", "占", "比", "%", "$"])

            if has_number and has_keyword:
                claims.append({
                    "claim_id": f"claim_{i}",
                    "text": s[:100],
                    "confidence": 0.5,
                    "has_data": True,
                })

        return claims[:10]

    async def _auto_link_claims_evidence(
        self,
        claims: list[dict],
        contexts: list[dict],
    ):
        """自动为 claims 匹配 evidence"""
        if not claims or not contexts:
            return

        for claim_data in claims:
            claim_id = claim_data.get("claim_id", "")
            claim_text = claim_data.get("text", "")

            best_evidence_ids = await self._find_best_evidence(claim_text, contexts)

            if best_evidence_ids:
                self.link_claim_evidence(claim_id, best_evidence_ids)

    async def _find_best_evidence(
        self,
        claim_text: str,
        contexts: list[dict],
    ) -> list[str]:
        """
        为 claim 找到最相关的 evidence

        Args:
            claim_text: claim 文本
            contexts: 可用的 evidence 列表

        Returns:
            evidence_ids 列表
        """
        if not contexts:
            return []

        if self._llm is None:
            return self._keyword_evidence_match(claim_text, contexts)

        import json

        context_text = "\n".join(
            f"[{i}] {c.get('text', '')[:200]}"
            for i, c in enumerate(contexts[:8])
        )

        prompt = f"""请判断以下主张最可能来自哪个文档片段。

主张: {claim_text}

文档片段:
{context_text}

请找出最可能支持这个主张的文档片段（1-3个）。
只选择真正包含主张中信息的片段。

请以 JSON 格式输出:
{{
  "evidence_ids": ["0", "2"],
  "reasoning": "简短理由"
}}

如果没有找到相关片段，返回空数组。"""

        try:
            response = await self._llm.generate(
                prompt,
                max_tokens=256,
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

            evidence_ids = data.get("evidence_ids", [])
            if isinstance(evidence_ids, list):
                result = []
                for eid in evidence_ids[:3]:
                    idx = int(eid) if str(eid).isdigit() else -1
                    if 0 <= idx < len(contexts):
                        chunk_id = contexts[idx].get("chunk_id", f"chunk_{idx}")
                        result.append(f"ev_{chunk_id}")
                return result

        except Exception as e:
            logger.warning(f"Evidence matching failed: {e}")

        return self._keyword_evidence_match(claim_text, contexts)

    def _keyword_evidence_match(
        self,
        claim_text: str,
        contexts: list[dict],
    ) -> list[str]:
        """基于关键词的 evidence 匹配（降级方案）"""
        import re

        claim_words = set(re.findall(r"\b\w{2,}\b", claim_text.lower()))
        stop_words = {"的", "是", "在", "了", "和", "与", "有", "为", "对", "于", "a", "an", "the", "is", "are", "and", "or", "in", "on", "at"}
        claim_words -= stop_words

        if not claim_words:
            return []

        best_idx = -1
        best_score = 0

        for i, ctx in enumerate(contexts):
            text = ctx.get("text", "").lower()
            score = sum(1 for word in claim_words if word in text)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0 and best_score > 0:
            chunk_id = contexts[best_idx].get("chunk_id", f"chunk_{best_idx}")
            return [f"ev_{chunk_id}"]

        return []

    def _track_cross_session(self, claims: list[dict]) -> list[dict]:
        """跨会话追踪：相同 claim 在多会话中出现时自动关联"""
        if not claims:
            return []

        cross_session = []

        for claim_data in claims:
            claim_text = claim_data.get("text", "")
            if len(claim_text) < 10:
                continue

            claim_key = claim_text[:50].lower()

            if claim_key in self._cross_session_store:
                self._cross_session_store[claim_key].append(self._session_id)
                cross_session.append({
                    "claim_text": claim_text,
                    "sessions": self._cross_session_store[claim_key],
                    "count": len(self._cross_session_store[claim_key]),
                })
            else:
                self._cross_session_store[claim_key] = [self._session_id]

        return cross_session

    def _generate_alerts(self, coverage_result: dict) -> list[str]:
        """生成覆盖率告警"""
        alerts = []

        coverage = coverage_result.get("coverage", 0)
        if coverage < self._coverage_threshold:
            alerts.append(
                f"覆盖率告警: {coverage:.1%} < {self._coverage_threshold:.1%} 阈值，"
                f"有 {coverage_result.get('total_claims', 0) - coverage_result.get('verified_claims', 0)} 个 claim 无 evidence"
            )

        unverified = coverage_result.get("unverified_claims", [])
        if len(unverified) > 3:
            alerts.append(
                f"无证据 claim 过多 ({len(unverified)} 个)，建议检查检索质量或生成准确性"
            )

        unused = coverage_result.get("unused_evidence", [])
        if len(unused) > len(self._evidence) * 0.5:
            alerts.append(
                f"检索到的 evidence 未被利用 ({len(unused)}/{len(self._evidence)})，"
                f"可能存在检索-生成语义鸿沟"
            )

        return alerts
