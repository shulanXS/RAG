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
from backend.generation.llm_client import LLMClient
from backend.generation.prompt_builder import PromptBuilder
from backend.generation.structured_output import StructuredOutputGenerator

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

        # ReAct 和 Plan-and-Execute Agent
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

        # 步骤 0: 语义缓存检查
        if semantic_cache_fn:
            cached = await semantic_cache_fn(query)
            if cached:
                latency = (time.perf_counter() - start) * 1000
                return OrchestratorResult(
                    answer=cached.get("answer", ""),
                    citations=cached.get("citations", []),
                    confidence=cached.get("confidence", "medium"),
                    latency_ms=latency,
                    trace=trace,
                    cache_hit=True,
                )

        # 步骤 1: 查询改写（多轮对话）
        from backend.retrieval.query_rewriter import QueryRewriter
        rewriter = QueryRewriter(llm_client=self._llm.generator_client if self._llm else None)
        rewritten = rewriter.rewrite(query, conversation_history)
        was_rewritten = rewritten.was_rewritten
        display_query = rewritten.rewritten

        trace["query_rewrite"] = {
            "was_rewritten": was_rewritten,
            "confidence": rewritten.confidence,
            "original": query,
            "rewritten": display_query,
        }

        # 步骤 2: 查询复杂度路由
        routing = self._router.route(display_query, conversation_history)
        complexity = routing.complexity

        trace["routing"] = {
            "complexity": complexity.value,
            "confidence": routing.confidence,
            "reasoning": routing.reasoning,
            "approach": routing.recommended_approach,
        }

        # 步骤 3: 根据复杂度执行不同路径
        retrieved_chunks: list[dict] = []
        answer: str = ""
        confidence: float = routing.confidence

        if complexity == QueryComplexity.SIMPLE:
            # 路径 3a: 简单查询 → 直接混合检索
            if self._hybrid_search:
                chunks, retrieval_context = await self._hybrid_search.search(display_query)
                retrieved_chunks = chunks
                trace["retrieval"] = {
                    "strategy": "hybrid",
                    "latency_ms": retrieval_context.total_latency_ms,
                    "num_chunks": len(chunks),
                }
                answer, citations = await self._generate_answer(
                    display_query, chunks
                )
            else:
                answer = "检索引擎未配置"
                citations = []

        elif complexity == QueryComplexity.MODERATE:
            # 路径 3b: 中等复杂度 → ReAct Agent
            react = self._get_react_agent()
            answer, conf, chunks = await react.run(query, display_query)
            retrieved_chunks = chunks
            confidence = conf
            trace["agent"] = {
                "type": "ReAct",
                "confidence": conf,
                "num_iterations": len(react._trace),
            }
            citations = self._extract_citations(chunks)

        elif complexity == QueryComplexity.COMPLEX:
            # 路径 3c: 复杂查询 → Plan-and-Execute
            plan_agent = self._get_plan_agent()
            answer, conf, chunks = await plan_agent.run(display_query)
            retrieved_chunks = chunks
            confidence = conf
            trace["agent"] = {
                "type": "Plan-and-Execute",
                "confidence": conf,
                "num_steps": len(plan_agent._trace),
            }
            citations = self._extract_citations(chunks)

        elif complexity == QueryComplexity.BEYOND_KB:
            # 路径 3d: 模型可直接回答 → 跳过检索
            trace["retrieval"] = {"strategy": "skip", "reason": "beyond_kb"}
            answer, citations = await self._direct_generate(display_query)
            confidence = 0.9

        else:
            # 默认路径: 混合检索
            if self._hybrid_search:
                chunks, retrieval_context = await self._hybrid_search.search(display_query)
                retrieved_chunks = chunks
                trace["retrieval"] = {
                    "strategy": "hybrid",
                    "latency_ms": retrieval_context.total_latency_ms,
                    "num_chunks": len(chunks),
                }
                answer, citations = await self._generate_answer(display_query, chunks)
            else:
                answer = "无法处理此查询"
                citations = []

        # 步骤 4: Memory Bank 更新
        if retrieved_chunks:
            self._memory_bank.add_evidence(retrieved_chunks)
            coverage = self._memory_bank.verify_coverage()
            trace["memory_bank"] = coverage

        # 步骤 5: 置信度映射
        conf_level = self._map_confidence(confidence)

        # 步骤 6: 写入语义缓存
        if semantic_cache_fn and answer:
            await semantic_cache_fn(query, {
                "answer": answer,
                "citations": citations,
                "confidence": conf_level,
            })

        latency = (time.perf_counter() - start) * 1000
        trace["total_latency_ms"] = latency

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
        )

    async def _generate_answer(
        self,
        query: str,
        chunks: list[dict],
    ) -> tuple[str, list[dict]]:
        """基于检索结果生成答案"""
        if self._llm is None:
            return "LLM 不可用", []

        context = self._prompt_builder.build_context(chunks)
        prompt = self._prompt_builder.build_prompt(query, context)
        response = await self._llm.generate_async(prompt)
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
