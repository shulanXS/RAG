"""
orchestrator.py — 中央编排器
================================================================================
技术决策记录:
- 中央编排器统一调度 Router + ReAct + Plan-and-Execute 三层模式。
- 从 Query Router 开始，根据复杂度决定走哪条路径。
- Memory Bank 贯穿整个流程，累积 claim-evidence 链路。
- 所有路径最终汇合到生成层（LLM Generation）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from backend.agentic.query_router import QueryComplexity, QueryRouter
from backend.agentic.react_agent import ReActAgent
from backend.agentic.plan_execute import PlanExecuteAgent
from backend.agentic.memory_bank import MemoryBank
from backend.agentic.self_reflection import SelfReflection
from backend.generation.llm_client import LLMClient
from backend.generation.prompt_builder import PromptBuilder
from backend.generation.structured_output import StructuredOutputGenerator
from backend.observability.metrics import MetricsCollector
from backend.observability.tracing import TracingManager
from backend.retrieval.hyde import HyDEQueryEnhancer
from backend.retrieval.query_expander import QueryExpander

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


class AgenticOrchestrator:
    """
    中央编排器 — 统一调度 Agentic RAG 全流程

    执行流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  1. Query Rewrite (多轮对话改写)                           │
    │  2. Query Router (复杂度分类 → 决定路径)                     │
    │  3a. Simple → HybridSearchEngine → 生成                      │
    │  3b. Moderate → ReAct Agent → 生成                          │
    │  3c. Complex → Plan-and-Execute → 生成                      │
    │  3d. Beyond KB → Direct LLM → 生成                          │
    │  4. Memory Bank → Claim-Evidence 链接                       │
    │  5. Structured Output → JSON 约束输出                       │
    └─────────────────────────────────────────────────────────────┘

    设计模式: Facade Pattern
    - 对外提供单一入口（run()），隐藏内部复杂度
    - 内部委托给 Router、Agent、MemoryBank 等组件

    技术要点:
    - 所有复杂度级别共用 Memory Bank（累积证据）
    - 语义缓存检查在路由之前执行（最快路径）
    - 路由决策记录在 trace 中（用于评估和 debug）
    """

    def __init__(
        self,
        hybrid_search_engine=None,
        router: QueryRouter | None = None,
        llm_client: LLMClient | None = None,
        memory_bank_session_id: str = "default",
    ):
        self._hybrid_search = hybrid_search_engine
        self._router = router or QueryRouter()
        self._llm = llm_client
        self._prompt_builder = PromptBuilder()
        self._structured_output = StructuredOutputGenerator()
        self._memory_bank = MemoryBank(session_id=memory_bank_session_id)
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
        """
        执行完整的 Agentic RAG 流程

        Args:
            query: 用户查询
            conversation_history: 对话历史（用于 query rewriting）
            semantic_cache_fn: 语义缓存查询函数

        Returns:
            OrchestratorResult: 包含答案、引用、置信度等
        """
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

        # ---- HyDE + retrieval ----
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

        # ---- self-reflection (仅在 COMPLEX 且低置信度时触发) ----
        gaps: list[str] = []
        if (complexity == QueryComplexity.COMPLEX and routing.confidence < 0.7 and retrieved_chunks and answer and self._llm):
            with tm.create_span("rag.self_reflection") as span:
                reflection = SelfReflection(llm_client=self._llm.generator_client)
                ref_result = await reflection.reflect(query, answer, retrieved_chunks)
                if ref_result.overall_score < 0.6 and ref_result.revised_answer:
                    answer = ref_result.revised_answer
                gaps = ref_result.gaps
                mc.record_retrieval_latency(
                    "reflection",
                    0.0,
                    ref_result.overall_score,
                    None,
                )
                span.set_attribute("score", ref_result.overall_score)
                span.set_attribute("requires_correction", ref_result.requires_correction)
            trace["reflection"] = {
                "score": ref_result.overall_score,
                "needs_correction": ref_result.requires_correction,
                "gaps": gaps,
            }
        elif retrieved_chunks and answer and self._llm:
            with tm.create_span("rag.self_reflection") as span:
                reflection = SelfReflection(llm_client=self._llm.generator_client)
                ref_result = await reflection.reflect(query, answer, retrieved_chunks)
                if ref_result.overall_score < 0.6 and ref_result.revised_answer:
                    answer = ref_result.revised_answer
                gaps = ref_result.gaps
                span.set_attribute("score", ref_result.overall_score)
                span.set_attribute("requires_correction", ref_result.requires_correction)
                span.set_attribute("skipped", False)
            trace["reflection"] = {
                "score": ref_result.overall_score,
                "needs_correction": ref_result.requires_correction,
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
