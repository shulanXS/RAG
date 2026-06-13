"""
orchestrator.py — 中央编排器
================================================================================
技术决策记录:
- 中央编排器统一调度 Router + ReAct + Plan-and-Execute 三层模式。
- 从 Query Router 开始，根据复杂度决定走哪条路径。
- 所有路径最终汇合到生成层（LLM Generation）。
- Self-reflection 作为内部步骤内联，减少模块间依赖。
- Memory Bank 简化为轻量级内联实现，移除 714 行的外部模块。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from backend.agentic.query_router import QueryComplexity, QueryRouter
from backend.agentic.react_agent import ReActAgent
from backend.agentic.plan_execute import PlanExecuteAgent
from backend.generation.llm_client import LLMClient
from backend.generation.prompt_builder import PromptBuilder
from backend.generation.structured_output import StructuredOutputGenerator
from backend.observability.metrics import MetricsCollector
from backend.observability.tracing import TracingManager
from backend.retrieval.hyde import HyDEQueryEnhancer

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """
    编排器执行结果

    字段说明:
    - answer: 最终生成的答案
    - citations: 引用列表
    - confidence: 置信度
    - complexity: 查询复杂度（路由决策）
    - routing_confidence: 路由分类置信度
    - latency_ms: 总执行时间
    - trace: 各阶段执行信息
    - gaps: 信息缺口描述
    """
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: Literal["high", "medium", "low", "insufficient"] = "medium"
    complexity: QueryComplexity = QueryComplexity.MODERATE
    routing_confidence: float = 0.0
    latency_ms: float = 0.0
    trace: dict = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    was_rewritten: bool = False
    cache_hit: bool = False


# =============================================================================
# 内联轻量级 Memory Bank（替代 714 行的外部模块）
# =============================================================================


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
    轻量级 Memory Bank — 仅保留核心引用追踪，移除 claim-evidence 链路
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


# =============================================================================
# 内联 Self-Reflection（替代独立的 364 行模块）
# =============================================================================


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


async def _do_reflection(
    llm_client,
    query: str,
    answer: str,
    contexts: list[dict],
) -> tuple[float, list[str], str, bool]:
    """
    内联自我反思：评估答案质量，返回 (score, gaps, revised_answer, needs_correction)
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


# =============================================================================
# 中央编排器
# =============================================================================


class AgenticOrchestrator:
    """
    中央编排器 — 统一调度 Agentic RAG 全流程

    执行流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. 语义缓存检查                                           │
    │  2. Query Rewrite (多轮对话改写)                           │
    │  3. Query Router (复杂度分类 → 决定路径)                     │
    │  4a. Simple   → HybridSearchEngine → 生成                    │
    │  4b. Moderate  → ReAct Agent      → 生成                    │
    │  4c. Complex   → Plan-and-Execute → 生成                    │
    │  4d. Beyond KB → Direct LLM       → 生成                    │
    │  5. Self-Reflection (内联)                                │
    │  6. 引用提取                                                │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        hybrid_search_engine=None,
        router: QueryRouter | None = None,
        llm_client: LLMClient | None = None,
        session_id: str = "default",
    ):
        self._hybrid_search = hybrid_search_engine
        self._router = router or QueryRouter()
        self._llm = llm_client
        self._prompt_builder = PromptBuilder()
        self._structured_output = StructuredOutputGenerator()
        self._memory_bank = SimpleMemoryBank(session_id=session_id)
        self._metrics = MetricsCollector()
        self._tracing = TracingManager()

        self._react_agent: ReActAgent | None = None
        self._plan_agent: PlanExecuteAgent | None = None

    def _get_react_agent(self) -> ReActAgent:
        if self._react_agent is None:
            self._react_agent = ReActAgent(
                llm_client=self._llm.router_client if self._llm else None,
                retrieval_fn=self._run_retrieval,
            )
        return self._react_agent

    def _get_plan_agent(self) -> PlanExecuteAgent:
        if self._plan_agent is None:
            self._plan_agent = PlanExecuteAgent(
                llm_client=self._llm.generator_client if self._llm else None,
                retrieval_fn=self._run_retrieval,
            )
        return self._plan_agent

    async def _run_retrieval(self, query: str) -> list[dict]:
        """检索函数（供 Agent 调用）"""
        if self._hybrid_search is None:
            return []
        chunks, _ = await self._hybrid_search.search(query)
        return chunks

    async def run(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        semantic_cache_fn=None,
    ) -> OrchestratorResult:
        start = time.perf_counter()
        trace: dict = {}
        cache_hit = False
        was_rewritten = False
        mc = self._metrics
        tm = self._tracing

        # ---- cache_check ----
        with tm.create_span("rag.cache_lookup") as span:
            span.set_attribute("query_length", len(query))
            if semantic_cache_fn:
                cached = await semantic_cache_fn(query)
                if cached:
                    cache_hit = True
                    mc.record_cache_hit(hit=True)
                    latency = (time.perf_counter() - start) * 1000
                    return OrchestratorResult(
                        answer=cached.get("answer", ""),
                        citations=cached.get("citations", []),
                        confidence=cached.get("confidence", "medium"),
                        latency_ms=latency,
                        trace=trace,
                        cache_hit=True,
                    )
            mc.record_cache_hit(hit=False)
            span.set_attribute("cache_hit", False)

        # ---- query_rewrite ----
        from backend.retrieval.query_rewriter import QueryRewriter
        rewriter = QueryRewriter(llm_client=self._llm.generator_client if self._llm else None)
        rewritten = rewriter.rewrite(query, conversation_history)
        was_rewritten = rewritten.was_rewritten
        display_query = rewritten.rewritten

        with tm.create_span("rag.query_rewrite") as span:
            span.set_attribute("was_rewritten", was_rewritten)
            span.set_attribute("original_query_length", len(query))
            span.set_attribute("rewritten_query_length", len(display_query))

        trace["query_rewrite"] = {
            "was_rewritten": was_rewritten,
            "confidence": rewritten.confidence,
            "original": query,
            "rewritten": display_query,
        }

        # ---- routing ----
        routing = self._router.route(display_query, conversation_history)
        complexity = routing.complexity

        with tm.create_span("rag.routing") as span:
            span.set_attribute("complexity", complexity.value)
            span.set_attribute("routing_confidence", routing.confidence)
            span.set_attribute("recommended_approach", routing.recommended_approach)

        trace["routing"] = {
            "complexity": complexity.value,
            "confidence": routing.confidence,
            "reasoning": routing.reasoning,
            "approach": routing.recommended_approach,
        }

        # ---- HyDE (仅 COMPLEX 复杂度) ----
        search_query = display_query
        if complexity == QueryComplexity.COMPLEX and self._llm:
            if self._llm.generator_client:
                with tm.create_span("rag.hyde") as span:
                    hyde = HyDEQueryEnhancer(llm_client=self._llm.generator_client)
                    search_query = await hyde.enhance(display_query)
                    span.set_attribute("enhanced_query_length", len(search_query))

        # ---- execute by complexity ----
        retrieved_chunks: list[dict] = []
        answer: str = ""
        confidence: float = routing.confidence
        citations: list[dict] = []

        if complexity == QueryComplexity.SIMPLE:
            if self._hybrid_search:
                chunks, retrieval_context = await self._hybrid_search.search(search_query)
                retrieved_chunks = chunks
                mc.record_retrieval_latency(
                    "total",
                    retrieval_context.total_latency_ms / 1000,
                    routing.confidence,
                    len(chunks),
                )
                trace["retrieval"] = {
                    "strategy": "hybrid",
                    "latency_ms": retrieval_context.total_latency_ms,
                    "num_chunks": len(chunks),
                    "stages": retrieval_context.stage_breakdown,
                }
                answer, citations = await self._generate_answer(search_query, chunks)
            else:
                answer = "检索引擎未配置"
                citations = []

        elif complexity == QueryComplexity.MODERATE:
            with tm.create_span("rag.agentic") as span:
                span.set_attribute("agent_type", "ReAct")
                react = self._get_react_agent()
                answer, conf, chunks = await react.run(query, search_query)
                retrieved_chunks = chunks
                confidence = conf
                span.set_attribute("num_iterations", len(react._trace))
                span.set_attribute("confidence", conf)
            trace["agent"] = {
                "type": "ReAct",
                "confidence": conf,
                "num_iterations": len(react._trace),
            }
            citations = self._extract_citations(chunks)

        elif complexity == QueryComplexity.COMPLEX:
            with tm.create_span("rag.agentic") as span:
                span.set_attribute("agent_type", "PlanAndExecute")
                plan_agent = self._get_plan_agent()
                answer, conf, chunks = await plan_agent.run(search_query)
                retrieved_chunks = chunks
                confidence = conf
                span.set_attribute("num_steps", len(plan_agent._trace))
                span.set_attribute("confidence", conf)
            trace["agent"] = {
                "type": "Plan-and-Execute",
                "confidence": conf,
                "num_steps": len(plan_agent._trace),
            }
            citations = self._extract_citations(chunks)

        else:
            trace["retrieval"] = {"strategy": "skip", "reason": "beyond_kb"}
            answer, citations = await self._direct_generate(search_query)
            confidence = 0.9

        # ---- self-reflection (内联，仅在有检索结果且非 BEYOND_KB 时触发) ----
        gaps: list[str] = []
        if retrieved_chunks and answer and self._llm and self._llm.generator_client:
            with tm.create_span("rag.self_reflection") as span:
                score, gaps, revised, needs_correction = await _do_reflection(
                    self._llm.generator_client, query, answer, retrieved_chunks
                )
                if needs_correction and score < 0.5:
                    answer = revised
                confidence = score
                span.set_attribute("score", score)
                span.set_attribute("needs_correction", needs_correction)
            trace["reflection"] = {
                "score": score,
                "needs_correction": needs_correction,
                "gaps": gaps,
            }

        # ---- memory_bank ----
        if retrieved_chunks:
            self._memory_bank.add_evidence(retrieved_chunks)
            coverage = self._memory_bank.verify_coverage()
            trace["memory_bank"] = coverage

        # ---- confidence mapping ----
        conf_level = self._map_confidence(confidence)

        # ---- write cache ----
        if semantic_cache_fn and answer:
            await semantic_cache_fn(query, {
                "answer": answer,
                "citations": citations,
                "confidence": conf_level,
            })

        latency = (time.perf_counter() - start) * 1000
        trace["total_latency_ms"] = latency

        with tm.create_span("rag.generation") as span:
            span.set_attribute("llm_model", self._llm.generator_model if self._llm else "unknown")
            span.set_attribute("answer_length", len(answer))

        return OrchestratorResult(
            answer=answer,
            citations=citations,
            confidence=conf_level,
            complexity=complexity,
            routing_confidence=routing.confidence,
            latency_ms=latency,
            trace=trace,
            was_rewritten=was_rewritten,
            cache_hit=cache_hit,
            gaps=gaps,
        )

    async def _generate_answer(
        self,
        query: str,
        chunks: list[dict],
    ) -> tuple[str, list[dict]]:
        """基于检索结果生成答案"""
        if self._llm is None:
            return "LLM 不可用", []

        llm_start = time.perf_counter()
        context = self._prompt_builder.build_context(chunks)
        prompt = self._prompt_builder.build_prompt(query, context)
        response = await self._llm.generate_async(prompt)
        llm_latency = time.perf_counter() - llm_start

        self._metrics.record_llm_latency(
            self._llm.generator_model if self._llm else "unknown",
            llm_latency,
        )

        citations = self._extract_citations(chunks)
        return response, citations

    async def _direct_generate(self, query: str) -> tuple[str, list[dict]]:
        """直接生成（无需检索）"""
        if self._llm is None:
            return "LLM 不可用", []

        prompt = f"请回答以下问题：{query}"
        response = await self._llm.generate_async(prompt)
        return response, []

    @staticmethod
    def _extract_citations(chunks: list[dict]) -> list[dict]:
        """从检索结果中提取引用"""
        citations = []
        for chunk in chunks:
            citations.append({
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "quote": chunk.get("text", "")[:200],
                "score": chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
            })
        return citations

    @staticmethod
    def _map_confidence(confidence: float) -> Literal["high", "medium", "low", "insufficient"]:
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        return "insufficient"
