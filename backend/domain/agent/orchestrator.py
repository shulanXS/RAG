"""
orchestrator.py — 中央编排器
================================================================================
技术决策记录:
- 中央编排器统一调度 Router + ReAct 两层模式。
  Plan-and-Execute 在 P0 阶段被移除 — 它在 production RAG 中几乎没被使用过：
  99% 真实查询 Router 会分到 SIMPLE，剩下 1% 走 ReAct 足够。
- 从 Query Router 开始，根据复杂度决定走哪条路径。
- 所有路径最终汇合到生成层（LLM Generation）。
- Self-reflection 作为内部步骤内联，减少模块间依赖。
- Memory Bank 简化为轻量级内联实现，移除 714 行的外部模块。
- 移除项 (P0): HyDE — 仅 COMPLEX 路径用，主流程无收益却增加 200-500ms 延迟；
  已在 ARCHITECTURE.md 解释"为什么不做"。
- 移除项 (P0): StructuredOutputGenerator — LLMClient 已原生支持 JSON Schema 透传。
"""

from __future__ import annotations

import logging
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, AsyncGenerator

from backend.domain.agent.query_router import QueryComplexity, QueryRouter
from backend.domain.agent.react_agent import ReActAgent
from backend.domain.generation.llm_client import LLMClient
from backend.domain.generation.prompt_builder import PromptBuilder
from backend.observability.metrics import create_metrics_collector

logger = logging.getLogger(__name__)
from backend.observability.tracing import TracingManager

if TYPE_CHECKING:
    from backend.domain.tenant import TenantContext


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
    - cache_hit: 是否走语义缓存命中
    """
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: Literal["high", "medium", "low", "insufficient"] = "medium"
    complexity: QueryComplexity = QueryComplexity.MODERATE
    routing_confidence: float = 0.0
    latency_ms: float = 0.0
    trace: dict = field(default_factory=dict)
    cache_hit: bool = False


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
    │  4c. Complex   → ReAct (多步推理)   → 生成                    │
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
        chat_store=None,
        semantic_cache_fn=None,
    ):
        self._hybrid_search = hybrid_search_engine
        self._router = router or QueryRouter()
        self._llm = llm_client
        self._prompt_builder = PromptBuilder()
        self._metrics = create_metrics_collector()
        self._tracing = TracingManager()
        self._chat_store = chat_store
        # P1.2: 接受 cache callable（read 模式无参 / write 模式有 payload）
        self._semantic_cache_fn = semantic_cache_fn

        self._react_agent: ReActAgent | None = None

        # 读取 use_structured_output 配置（默认 True）
        try:
            from backend.config import get_config

            self._use_structured_output = get_config().llm.use_structured_output
        except Exception:
            self._use_structured_output = True

    def _get_react_agent(self) -> ReActAgent:
        if self._react_agent is None:
            self._react_agent = ReActAgent(
                llm_client=self._llm.router_client if self._llm else None,
                retrieval_fn=self._run_retrieval,
            )
        return self._react_agent

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
        session_id: str | None = None,
        semantic_cache_fn=None,
        tenant: "TenantContext | str | None" = None,
    ) -> OrchestratorResult:
        start = time.perf_counter()
        trace: dict = {}
        cache_hit = False
        mc = self._metrics
        tm = self._tracing

        # P1.2: 优先使用显式传入的 cache_fn，否则回退到构造期注入的实例
        effective_cache_fn = semantic_cache_fn or self._semantic_cache_fn

        # ---- P3.1(plan §4.1) 并行化: cache_lookup + query_rewrite ----
        # 旧实现: 串行 cache_lookup → rewrite → routing,cache 命中时 rewrite 浪费 ~150ms
        # 新实现: 两个 asyncio.create_task 并发,cache 命中时 cancel rewrite

        async def _cache_lookup_task():
            if not effective_cache_fn:
                return None
            try:
                return await effective_cache_fn(query)
            except Exception as e:
                logger.debug(f"semantic_cache_fn 读失败（降级）: {e}")
                return None

        async def _rewrite_task():
            from backend.domain.retrieval.query_rewriter import QueryRewriter
            rw = QueryRewriter(llm_client=self._llm.generator_client if self._llm else None)
            return await rw.rewrite_async(query, conversation_history)

        # 启动两个并发任务
        cache_task = asyncio.create_task(_cache_lookup_task())
        rewrite_task = asyncio.create_task(_rewrite_task())

        # ---- cache_check 决策 ----
        with tm.create_span("rag.cache_lookup") as span:
            span.set_attribute("query_length", len(query))
            # 等待 cache 完成(不阻塞 rewrite, 它独立运行)
            cached = await cache_task
            if cached:
                cache_hit = True
                mc.record_cache_hit(hit=True)
                # cache 命中 → cancel rewrite 任务, 节省 LLM 调用
                if not rewrite_task.done():
                    rewrite_task.cancel()
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

        # ---- load session history if not provided ----
        if session_id and conversation_history is None and self._chat_store:
            # P0-4: chat_store 改 async
            conversation_history = await self._chat_store.get_history(session_id, limit=20)

        # ---- 等待 rewrite 完成(若 cache 命中已 cancel,这里会抛 CancelledError 需处理) ----
        try:
            rewritten = await rewrite_task
        except asyncio.CancelledError:
            # 极端 case: cache_task 和 rewrite_task 几乎同时完成
            # 重新跑一次(rewrite 不可跳,因为后续 routing 需要)
            rewritten = await _rewrite_task()
        display_query = rewritten.rewritten

        with tm.create_span("rag.query_rewrite") as span:
            span.set_attribute("original_query_length", len(query))
            span.set_attribute("rewritten_query_length", len(display_query))

        trace["query_rewrite"] = {
            "was_rewritten": rewritten.was_rewritten,
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
            # P2-B7: signals 透出到 OTel span, 供 Jaeger 检索 / dashboard
            if routing.signals is not None:
                s = routing.signals
                span.set_attribute("signals.has_pronoun", s.has_pronoun)
                span.set_attribute("signals.is_multi_hop", s.is_multi_hop)
                span.set_attribute("signals.entity_count", s.entity_count)
                span.set_attribute("signals.query_length", s.query_length)
                span.set_attribute("signals.has_quote", s.has_quote)

        # P2-B7: signals.to_dict() 走真实 QuerySignals,
        # 单元测试里 Mock 也会带 to_dict, 但调用安全 (Mock 自动返回 Mock)
        signals_dict = None
        if routing.signals is not None and hasattr(routing.signals, "to_dict"):
            try:
                signals_dict = routing.signals.to_dict()
            except Exception:
                signals_dict = None
        trace["routing"] = {
            "complexity": complexity.value,
            "confidence": routing.confidence,
            "reasoning": routing.reasoning,
            "approach": routing.recommended_approach,
            "signals": signals_dict,
        }

        # ---- execute by complexity ----
        # P0: COMPLEX 路径改用 ReAct（多步推理）而非 Plan-and-Execute。
        # 实测 Plan-and-Execute 容易在长链路上偏离 query，且需要额外
        # LLM call 规划 plan (200-500ms 延迟收益不抵)。
        retrieved_chunks: list[dict] = []
        answer: str = ""
        confidence: float = routing.confidence
        citations: list[dict] = []

        if complexity == QueryComplexity.SIMPLE:
            if self._hybrid_search:
                # P2-B7: 把 routing.signals 透传给 hybrid_search,
                # search() 内部把 signals 写到 RetrievalContext (供 OTel / debug)
                # 同时 complexity 用于 DynamicRRFFusion 选 k
                chunks, retrieval_context = await self._hybrid_search.search(
                    display_query,
                    tenant=tenant,
                    complexity=complexity.value,
                    signals=routing.signals,
                )
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
                    "fusion_k_used": retrieval_context.fusion_k_used,
                }
                answer, citations = await self._generate_answer(display_query, chunks)
            else:
                answer = "检索引擎未配置"
                citations = []

        elif complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX):
            with tm.create_span("rag.agentic") as span:
                span.set_attribute("agent_type", "ReAct")
                span.set_attribute("complexity", complexity.value)
                react = self._get_react_agent()
                answer, conf, chunks = await react.run(display_query)
                retrieved_chunks = chunks
                confidence = conf
                span.set_attribute("num_iterations", len(react._trace))
                span.set_attribute("confidence", conf)
            trace["agent"] = {
                "type": "ReAct",
                "complexity": complexity.value,
                "confidence": conf,
                "num_iterations": len(react._trace),
            }
            citations = self._extract_citations(chunks)

        else:
            trace["retrieval"] = {"strategy": "skip", "reason": "beyond_kb"}
            answer, citations = await self._direct_generate(display_query)
            confidence = 0.9

        # ---- confidence mapping ----
        conf_level = self._map_confidence(confidence)

        # ---- save session history ----
        if session_id and self._chat_store and answer:
            try:
                # P0-4: chat_store 改 async
                await self._chat_store.add_message(session_id, "user", query)
                await self._chat_store.add_message(session_id, "assistant", answer)
            except Exception as e:
                logger.warning(f"Failed to save session history: {e}")

        # ---- write cache ----
        if effective_cache_fn and answer:
            try:
                await effective_cache_fn(query, {
                    "answer": answer,
                    "citations": citations,
                    "confidence": conf_level,
                })
            except Exception as e:
                logger.debug(f"semantic_cache_fn 写失败: {e}")

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
            cache_hit=cache_hit,
        )

    # 精简的 JSON Schema — 与 LLMClient.generate_async(structured_schema=...) 配合
    OUTPUT_SCHEMA: dict = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "直接回答用户问题"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                },
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "insufficient"],
            },
        },
        "required": ["answer", "citations", "confidence"],
    }

    async def _generate_answer(
        self,
        query: str,
        chunks: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        基于检索结果生成答案。

        P1-Phase1.2: citation verifier 已删除 — 答案-引用对齐验证无人消费（前端不展示），
        移除后每条 MODERATE/COMPLEX query 节省 200-500ms LLM 调用 + 一次额外 token 成本。
        """
        if self._llm is None:
            return "LLM 不可用", []

        llm_start = time.perf_counter()
        context = self._prompt_builder.build_context(chunks)
        prompt = self._prompt_builder.build_prompt(query, context)

        # 仅在 use_structured_output=True 时透传 schema（配置可关闭以兼容旧 LLM）
        structured_schema = self.OUTPUT_SCHEMA if getattr(self, "_use_structured_output", True) else None
        response = await self._llm.generate_async(
            prompt,
            structured_schema=structured_schema,
        )
        llm_latency = time.perf_counter() - llm_start

        self._metrics.record_llm_latency(
            self._llm.generator_model if self._llm else "unknown",
            llm_latency,
        )

        citations = self._extract_citations(chunks)
        return response, citations

    async def _direct_generate(self, query: str) -> tuple[str, list[dict]]:
        """直接生成（无需检索）

        仍走 LLMClient.generate_async，但显式不传 structured_schema
        （Beyond-KB 路径通常无引用，强制 schema 反而限制 LLM 表达）。
        """
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

    async def run_stream(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        session_id: str | None = None,
        tenant: "TenantContext | str | None" = None,
    ) -> AsyncGenerator[dict, None]:
        """
        流式执行编排器，yield 每一步的事件。

        Yields:
            {"stage": str, ...stage_specific_fields}
        """
        # ---- load session history ----
        if session_id and conversation_history is None and self._chat_store:
            # P0-4: chat_store 改 async
            conversation_history = await self._chat_store.get_history(session_id, limit=20)

        # ---- cache_check (P1.2: 真实 cache 命中) ----
        cache_hit = False
        if self._semantic_cache_fn:
            try:
                cached = await self._semantic_cache_fn(query)
            except Exception as e:
                logger.debug(f"semantic_cache_fn 读失败: {e}")
                cached = None
            if cached:
                cache_hit = True
                self._metrics.record_cache_hit(hit=True)
                yield {
                    "stage": "cache_check",
                    "cache_hit": True,
                    "similarity": cached.get("similarity", 1.0),
                    "done": True,
                }
                # 命中时直接 yield final answer 后结束
                yield {
                    "stage": "done",
                    "answer": cached.get("answer", ""),
                    "citations": cached.get("citations", []),
                    "confidence": cached.get("confidence", "medium"),
                    "session_id": session_id,
                    "cache_hit": True,
                    "done": True,
                }
                return
        if not cache_hit:
            self._metrics.record_cache_hit(hit=False)
            yield {"stage": "cache_check", "cache_hit": False, "done": True}

        # ---- query_rewrite ----
        from backend.domain.retrieval.query_rewriter import QueryRewriter

        yield {"stage": "rewrite", "done": False}
        rewriter = QueryRewriter(llm_client=self._llm.generator_client if self._llm else None)
        rewritten = await rewriter.rewrite_async(query, conversation_history)
        display_query = rewritten.rewritten
        yield {
            "stage": "rewrite",
            "was_rewritten": rewritten.was_rewritten,
            "query": display_query,
            "done": True,
        }

        # ---- routing ----
        yield {"stage": "routing", "done": False}
        routing = self._router.route(display_query, conversation_history)
        yield {
            "stage": "routing",
            "complexity": routing.complexity.value,
            "confidence": routing.confidence,
            "done": True,
        }

        # ---- retrieval ----
        retrieved_chunks: list[dict] = []
        complexity = routing.complexity

        yield {"stage": "retrieval", "done": False}

        if complexity == QueryComplexity.SIMPLE:
            if self._hybrid_search:
                chunks, retrieval_context = await self._hybrid_search.search(display_query, tenant=tenant)
                retrieved_chunks = chunks
                yield {
                    "stage": "retrieval",
                    "num_chunks": len(chunks),
                    "latency_ms": retrieval_context.total_latency_ms,
                    "done": True,
                }
            else:
                yield {"stage": "retrieval", "num_chunks": 0, "done": True}

        elif complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX):
            # P3.1(plan §3.5) 假流式清理: 删 ReAct 步骤流 + token 流叠加
            # 旧实现: 先 yield ReAct step events, 再 yield token 流(前端只渲染 token 流,
            #         ReAct 事件被忽略,后端在自嗨)
            # 新实现: ReAct 在后台 await 一次, 仅 yield 最终 token 流(前端用 native tools 流可视化)
            react = self._get_react_agent()
            full_answer, confidence, context_chunks = await react.run(display_query)
            retrieved_chunks = context_chunks
            if context_chunks and self._llm:
                ctx = self._prompt_builder.build_context(context_chunks)
                prompt = self._prompt_builder.build_prompt(display_query, ctx)
            else:
                prompt = full_answer or f"请回答以下问题：{display_query}"
        else:  # BEYOND_KB
            yield {
                "stage": "retrieval",
                "num_chunks": 0,
                "note": "beyond_kb: direct LLM",
                "done": True,
            }
            context_chunks = []
            full_answer = ""
            prompt = f"请回答以下问题：{display_query}"

        # ---- stream generation ----
        if self._llm is None:
            yield {"stage": "error", "message": "LLM not available", "done": True}
            return

        # token 级真流式：SIMPLE / MODERATE / COMPLEX / BEYOND_KB 全部走 generate_stream_async
        # MODERATE+COMPLEX 路径会先 yield ReAct 步骤事件，再 yield token 流。
        # 之前的"假流式"已修复（P0 阶段：移除 Plan-and-Execute 不流式分支）。
        if complexity in (
            QueryComplexity.SIMPLE,
            QueryComplexity.MODERATE,
            QueryComplexity.COMPLEX,
            QueryComplexity.BEYOND_KB,
        ):
            yield {"stage": "generating", "token": "", "done": False}
            async for token in self._llm.generate_stream_async(prompt):
                full_answer += token
                yield {"stage": "generating", "token": token, "done": False}

        # ---- extract citations ----
        citations = self._extract_citations(retrieved_chunks)

        # ---- save session history + write semantic cache ----
        if session_id and self._chat_store and full_answer:
            try:
                # P0-4: chat_store 改 async
                await self._chat_store.add_message(session_id, "user", query)
                await self._chat_store.add_message(session_id, "assistant", full_answer)
            except Exception as e:
                logger.warning(f"Failed to save session history: {e}")

        if self._semantic_cache_fn and full_answer:
            try:
                await self._semantic_cache_fn(query, {
                    "answer": full_answer,
                    "citations": citations,
                    "confidence": self._map_confidence(routing.confidence),
                })
            except Exception as e:
                logger.debug(f"semantic_cache_fn 写失败: {e}")

        yield {
            "stage": "done",
            "answer": full_answer,
            "citations": citations,
            "confidence": self._map_confidence(routing.confidence),
            "session_id": session_id,
            "cache_hit": False,
            "done": True,
        }
