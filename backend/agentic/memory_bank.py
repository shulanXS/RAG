"""
memory_bank.py — 轻量级引用追踪

P1-1: 从 orchestrator.py 头部提取。功能不变，仅文件位置变化。
原 orchestrator.py 内联实现（ARCHITECTURE.md § Why Not 风格：
"Memory Bank 简化为轻量级内联实现，移除 714 行的外部模块"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvidenceUnit:
    source_id: str
    text: str
    doc_title: str = ""
    section_path: str = ""
    retrieval_score: float = 0.0
    used_by_claims: list[str] = field(default_factory=list)


class SimpleMemoryBank:
    """
    轻量级 Memory Bank — 仅保留核心引用追踪，移除 claim-evidence 链路。
    """

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._evidence: dict[str, EvidenceUnit] = {}

    def add_evidence(self, chunks: list[dict]) -> list[str]:
        evidence_ids = []
        for chunk in chunks:
            eid = f"ev_{chunk.get('chunk_id', '')}"
            if eid not in self._evidence:
                self._evidence[eid] = EvidenceUnit(
                    source_id=chunk.get("chunk_id", ""),
                    text=chunk.get("text", "")[:500],
                    doc_title=chunk.get("doc_title", ""),
                    section_path=chunk.get("section_path", ""),
                    retrieval_score=chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
                )
            evidence_ids.append(eid)
        return evidence_ids

    def verify_coverage(self) -> dict:
        used = set()
        for ev in self._evidence.values():
            used.update(ev.used_by_claims)
        total = len(self._evidence)
        return {
            "total_chunks": total,
            "used_chunks": len(used),
            "coverage": len(used) / total if total > 0 else 1.0,
        }
