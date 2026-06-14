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

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, AsyncGenerator

from backend.agentic.query_router import QueryComplexity, QueryRouter
from backend.agentic.react_agent import ReActAgent
from backend.generation.llm_client import LLMClient
from backend.generation.prompt_builder import PromptBuilder
from backend.generation.citation_verifier import CitationVerifier
from backend.observability.metrics import MetricsCollector

logger = logging.getLogger(__name__)
from backend.observability.tracing import TracingManager

if TYPE_CHECKING:
    from backend.security.tenant import TenantContext


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
    citation_verification: dict | None = None


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
        session_id: str = "default",
        chat_store=None,
        semantic_cache_fn=None,
    ):
        self._hybrid_search = hybrid_search_engine
        self._router = router or QueryRouter()
        self._llm = llm_client
        self._prompt_builder = PromptBuilder()
        self._memory_bank = SimpleMemoryBank(session_id=session_id)
        self._metrics = MetricsCollector()
        self._tracing = TracingManager()
        self._chat_store = chat_store
        # P1.2: 接受 cache callable（read 模式无参 / write 模式有 payload）
        self._semantic_cache_fn = semantic_cache_fn
        self._citation_verifier = CitationVerifier(
            llm_client=llm_client.generator_client if llm_client else None
        )

        self._react_agent: ReActAgent | None = None

        # 读取 use_structured_output 配置（默认 True）
        try:
            from backend.config import get_config

            self._use_structured_output = get_config().llm.use_structured_output
        except Exception:
            self._use_structured_output = True

    def _get_react_agent(self) -> ReActAgent:
        if self._react_agent is None:
            from backend.agentic.tool_registry import get_tool_registry
            self._react_agent = ReActAgent(
                llm_client=self._llm.router_client if self._llm else None,
                retrieval_fn=self._run_retrieval,
                tool_registry=get_tool_registry(),
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
        was_rewritten = False
        mc = self._metrics
        tm = self._tracing

        # P1.2: 优先使用显式传入的 cache_fn，否则回退到构造期注入的实例
        effective_cache_fn = semantic_cache_fn or self._semantic_cache_fn

        # ---- cache_check ----
        with tm.create_span("rag.cache_lookup") as span:
            span.set_attribute("query_length", len(query))
            if effective_cache_fn:
                try:
                    cached = await effective_cache_fn(query)
                except Exception as e:
                    logger.debug(f"semantic_cache_fn 读失败（降级）: {e}")
                    cached = None
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

        # ---- load session history if not provided ----
        if session_id and conversation_history is None and self._chat_store:
            conversation_history = self._chat_store.get_history(session_id, limit=20)

        # ---- query_rewrite ----
        from backend.retrieval.query_rewriter import QueryRewriter
        rewriter = QueryRewriter(llm_client=self._llm.generator_client if self._llm else None)
        rewritten = await rewriter.rewrite_async(query, conversation_history)
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
                chunks, retrieval_context = await self._hybrid_search.search(display_query, tenant=tenant)
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
                answer, citations, citation_verification = await self._generate_answer(display_query, chunks)
            else:
                answer = "检索引擎未配置"
                citations = []

        elif complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX):
            with tm.create_span("rag.agentic") as span:
                span.set_attribute("agent_type", "ReAct")
                span.set_attribute("complexity", complexity.value)
                react = self._get_react_agent()
                answer, conf, chunks = await react.run(query, display_query)
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

        # ---- save session history ----
        if session_id and self._chat_store and answer:
            try:
                self._chat_store.add_message(session_id, "user", query)
                self._chat_store.add_message(session_id, "assistant", answer)
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

        # P2.2: 写入 trace ring buffer
        try:
            import uuid
            from backend.observability.trace_store import record_trace

            spans_for_viewer = self._collect_spans(trace)
            record_trace(
                trace_id=str(uuid.uuid4()),
                started_at_ms=int(start * 1000),
                ended_at_ms=int((time.perf_counter()) * 1000),
                complexity=complexity.value,
                routing_confidence=routing.confidence,
                cache_hit=cache_hit,
                answer_length=len(answer),
                spans=spans_for_viewer,
            )
        except Exception as e:
            logger.debug(f"trace_record 失败: {e}")

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
            citation_verification=locals().get("citation_verification"),
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
            "gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "citations", "confidence"],
    }

    async def _generate_answer(
        self,
        query: str,
        chunks: list[dict],
    ) -> tuple[str, list[dict], dict | None]:
        """基于检索结果生成答案"""
        if self._llm is None:
            return "LLM 不可用", [], None

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

        # 答实对齐验证
        citation_verification = None
        if chunks and self._llm:
            try:
                verify_result = await self._citation_verifier.verify(query, response, chunks)
                citations = self._citation_verifier.to_citations(verify_result)
                citation_verification = {
                    "groundedness_score": verify_result.overall_groundedness_score,
                    "num_supported": verify_result.num_supported,
                    "num_total": verify_result.num_total,
                    "unsupported_claims": verify_result.unsupported_claims,
                }
            except Exception as e:
                logger.warning(f"Citation verification failed in _generate_answer: {e}")
                citations = self._extract_citations(chunks)
        else:
            citations = self._extract_citations(chunks)

        return response, citations, citation_verification

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

    @staticmethod
    def _collect_spans(trace: dict) -> list[dict]:
        """
        把 orchestrator 的 trace dict 转换成 trace_viewer 期望的 spans 列表。
        每个 span: {name, duration_ms, attrs}
        """
        spans = []
        if "query_rewrite" in trace:
            qr = trace["query_rewrite"]
            spans.append({
                "name": "rag.query_rewrite",
                "duration_ms": 0.0,  # 不计；偏 detail 时用 OTEL
                "attrs": {
                    "was_rewritten": qr.get("was_rewritten", False),
                    "confidence": qr.get("confidence", 0.0),
                },
            })
        if "routing" in trace:
            r = trace["routing"]
            spans.append({
                "name": "rag.routing",
                "duration_ms": 0.0,
                "attrs": {
                    "complexity": r.get("complexity"),
                    "confidence": r.get("confidence"),
                    "approach": r.get("approach"),
                },
            })
        if "retrieval" in trace:
            ret = trace["retrieval"]
            spans.append({
                "name": "rag.retrieval",
                "duration_ms": ret.get("latency_ms", 0.0),
                "attrs": {
                    "strategy": ret.get("strategy"),
                    "num_chunks": ret.get("num_chunks", 0),
                },
            })
        if "agent" in trace:
            a = trace["agent"]
            spans.append({
                "name": "rag.agentic",
                "duration_ms": 0.0,
                "attrs": {
                    "type": a.get("type"),
                    "num_iterations": a.get("num_iterations", a.get("num_steps", 0)),
                    "confidence": a.get("confidence", 0.0),
                },
            })
        if "reflection" in trace:
            rf = trace["reflection"]
            spans.append({
                "name": "rag.self_reflection",
                "duration_ms": 0.0,
                "attrs": {
                    "score": rf.get("score", 0.0),
                    "needs_correction": rf.get("needs_correction", False),
                },
            })
        if trace.get("cache_hit"):
            spans.append({
                "name": "rag.cache_lookup",
                "duration_ms": 0.0,
                "attrs": {"cache_hit": True},
            })
        return spans

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
            conversation_history = self._chat_store.get_history(session_id, limit=20)

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
        from backend.retrieval.query_rewriter import QueryRewriter

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
            # MODERATE 和 COMPLEX 都走 ReAct 路径（plan_execute 已在 P0 移除）。
            # run_stream 真实 yield LangGraph 每个节点的 step event；token-level
            # 流式由后续 _stream_final_generation() 提供。
            react = self._get_react_agent()
            full_answer = ""
            confidence = routing.confidence
            async for event in react.run_stream(query, display_query):
                yield event
                if event.get("step_type") == "final" and event.get("is_last"):
                    full_answer = event.get("content", full_answer)
                    confidence = event.get("confidence", confidence)
            context_chunks = list(getattr(react, "_current_chunks", []))
            retrieved_chunks = context_chunks
            yield {"stage": "retrieval", "num_chunks": len(context_chunks), "done": True}
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
                self._chat_store.add_message(session_id, "user", query)
                self._chat_store.add_message(session_id, "assistant", full_answer)
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
