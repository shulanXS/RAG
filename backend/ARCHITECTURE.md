# Backend 架构文档

> 企业级 RAG 系统的后端架构深度分析 — 模块划分、数据流、关键设计决策与权衡。

---

## 1. 总览

后端是一个 **FastAPI 单体应用**，围绕中央编排器（`AgenticOrchestrator`）构建 11 个职责清晰的子模块。整体技术栈：

- **Web 框架**：FastAPI + Uvicorn（async / I/O-bound 友好）
- **向量数据库**：Qdrant（mmap + HNSW + 服务端 BM25 RRF 融合）
- **关系型 / 缓存存储**：Redis（语义缓存、限流、聊天历史、会话）
- **LLM 后端**：DeepSeek / OpenAI（OpenAI 兼容协议，策略模式）
- **Embedding**：BGE-M3（默认）/ OpenAI text-embedding-3-*（统一抽象）
- **Agent 框架**：LangGraph（ReAct 状态机）
- **异步任务队列**：Arq（基于 Redis Stream 的持久化任务）
- **观测**：OpenTelemetry（OTLP/Console）+ Prometheus（`/metrics`）+ JSON 日志（带 `request_id` / `tenant_id`）
- **评估**：RAGAS（5 指标）+ SQLite 持久化
- **认证**：JWT（HS256，bcrpyt 密码哈希 + pepper）

整体采用 **依赖注入（FastAPI Depends） + 单例（`lru_cache`/模块级）** 的混合策略：跨请求复用的重资源（embedder、LLM client、orchestrator）走单例；请求级实例（retrieval context、tenant context）走 DI。

---

## 2. 目录结构

> P3.1 目录重塑后,代码组织为"领域 + 横切"两段式:
>
> - `backend/domain/` — 领域层(ingestion / retrieval / generation / agent / cache / evaluation / session / tenant)
> - `backend/platform/` — 横切平台(auth / resilience / ratelimit / context / obs / api / workers)
> - `backend/app.py` — FastAPI 入口(原 main.py,简化为 ~190 行)
>
> 详见 `find backend/domain backend/platform -name "*.py"` 或代码本身的 import 关系。

---

## 3. 请求生命周期

### 3.1 完整链路（POST /api/chat）

下面以一次具体请求为例逐步说明。假设客户端发来：

```http
POST /api/chat HTTP/1.1
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...      # JWT, sub=alice
Content-Type: application/json

{
  "query": "供应商A断供对哪些客户有影响？",          # 复杂多跳查询
  "session_id": "sess_abc123",
  "history": [
    {"role": "user",      "content": "这季度主要供应商有哪些？"},
    {"role": "assistant", "content": "本季度主要供应商有 A、B、C 三家……"}
  ]
}
```

整条链路在服务端经由 **6 层处理** 完成。每一层都有明确职责，下文按执行顺序展开。

---

#### 第 1 层 · 跨切面：请求上下文注入（`RequestContextMiddleware`）

这是 ASGI 中间件栈的最外层，**先于**所有业务逻辑执行。

```python
# backend/middleware/request_context_middleware.py
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        tenant = request.headers.get("x-tenant-id")
        session = request.headers.get("x-session-id")
        with RequestContext(request_id=rid, tenant_id=tenant, session_id=session):
            response = await call_next(request)
            response.headers["x-request-id"] = rid
            return response
```

它做两件事：
1. **生成/透传 `request_id`**：客户端可从网关传 `X-Request-Id`，否则服务端生成 12 字符 UUID。
2. **建立 ContextVar scope**：通过 `RequestContext` contextmanager 把 `request_id` / `tenant_id` / `session_id` 写入 `contextvars` 模块的全局变量。

`contextvars`（而非 `threading.local`）是 Python 异步原语：**同 async task 链路上所有子调用都共享同一份上下文**，跨 `await` 自动继承。`RequestContextFilter` 装在 logging handler 上后，后续任何 `logger.info(...)` 自动带 `rid=xxx tenant=alice session=sess_abc123` 字段。

**为什么需要它？** 一个 RAG 请求会触发多次 LLM 调用、多次 Redis 操作、Qdrant 检索、Arq 任务；没有这层，事后在日志里无法把"这次请求触发的所有日志"串起来。

---

#### 第 2 层 · 跨切面：限流（`RateLimitMiddleware`）

Redis 滑动窗口，每租户 60 req/min：

```python
# middleware/rate_limiter.py — 核心逻辑
key = f"ratelimit:{tenant_id}"
pipe = self._client.pipeline()
pipe.zremrangebyscore(key, 0, now - 60)   # 清掉 60s 外的旧记录
pipe.zcard(key)                            # 统计当前窗口内请求数
pipe.zadd(key, {f"{now}": now})           # 计入当前请求
pipe.expire(key, 120)
results = await pipe.execute()
current_count = results[1]
allowed = current_count < 60
```

`tenant_id` 优先从 JWT 的 `sub` claim 解析（无 token 时 fallback IP）。超限返回 429，响应头带 `X-RateLimit-Remaining` / `X-RateLimit-Reset`。

> **P3.2 修复**：之前 try/except 静默吞错，改为硬失败——redis 缺失时拒绝启动，避免降级被掩盖。

---

#### 第 3 层 · FastAPI 依赖注入

路由函数 `chat()` 的签名是：

```python
# backend/api/chat.py
@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token_payload: dict = Depends(require_current_user),
    orchestrator: AgenticOrchestrator = Depends(deps.get_orchestrator),
) -> ChatResponse:
    ...
```

FastAPI 在调用前依次解析 3 个依赖：

| 依赖 | 作用 |
|------|------|
| `require_current_user` | JWT 校验，无效 → 401，类型非 `access` → 401。**整个 `/api/*` 都受保护。** |
| `deps.get_orchestrator` | `@lru_cache(maxsize=1)` 单例工厂，构造时自动注入 `HybridSearchEngine` + `QueryRouter` + `LLMClient` + `ChatStore` + `semantic_cache_fn`。 |
| `TenantContext.from_token(token_payload)` | 从 JWT 派生租户上下文，缺省 `"default"`，注入检索 filter。 |

依赖链构造图：

```
get_orchestrator()
  ├─► get_hybrid_search()       ← get_embedder()
  ├─► get_llm_client()           (双 client: router + generator)
  ├─► QueryRouter(llm.router_client)
  ├─► get_chat_store()           (redis.asyncio)
  └─► get_semantic_cache()       (redisvl AsyncSearchIndex, 失败则 None)
```

由于全部是 `lru_cache` 单例，**每个进程只构造一次**，第二次起直接返回缓存对象。

---

#### 第 4 层 · 业务编排：`AgenticOrchestrator.run()`

这是整个后端的核心。`orchestrator.run(query, history, session_id, tenant)` 串起 7 个子步骤，我们继续上面"供应商A断供"的请求追踪：

```python
# backend/agentic/orchestrator.py（简化版）
async def run(self, query, conversation_history=None, session_id=None, tenant=None):
    trace = {}
    start = time.perf_counter()

    # ── 4a · 语义缓存检查 ──
    if effective_cache_fn:
        cached = await effective_cache_fn(query)        # RedisSemanticCache.get
        if cached:
            return OrchestratorResult(answer=cached["answer"], ...,
                                       cache_hit=True)  # ~50ms 直接返回

    # ── 4b · 加载会话历史 ──
    if session_id and conversation_history is None and self._chat_store:
        conversation_history = await self._chat_store.get_history(
            session_id, limit=20
        )                                                 # 从 Redis 取最近 20 条

    # ── 4c · Query 改写（多轮场景） ──
    rewriter = QueryRewriter(llm_client=self._llm.generator_client)
    rewritten = await rewriter.rewrite_async(query, conversation_history)
    display_query = rewritten.rewritten
    # 例: "供应商A断供对哪些客户有影响？" 没有代词且长度足够，
    #     _needs_rewriting() 返回 False → 不调 LLM，直接用原 query

    # ── 4d · 提取纯规则信号（< 1ms） ──
    signals = QueryAnalyzer().analyze(display_query)
    # QuerySignals(has_pronoun=False, entity_count=3, is_multi_hop=True,
    #              query_length=15, has_quote=False)

    # ── 4e · 路由分类（LLM 调用 1 次） ──
    routing = self._router.route(display_query, conversation_history)
    # QueryRouter 把 signals 当 prompt context 喂给轻量 LLM：
    # prompt = SYSTEM_PROMPT + history + signals_context + query
    # 输出: {"complexity": "complex", "confidence": 0.92, "reasoning": "..."}
    complexity = routing.complexity
    # routing.confidence = 0.92 ≥ 0.6 阈值 → 不升级

    # ── 4f · 按复杂度分发 ──
    if complexity == QueryComplexity.SIMPLE:
        # 4f.S → 混合检索
        chunks, ctx = await self._hybrid_search.search(
            display_query, tenant=tenant,
            complexity="simple", signals=signals,
        )
        answer, citations = await self._generate_answer(display_query, chunks)

    elif complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX):
        # 4f.M/C → ReAct 多步推理（我们的请求走这条）
        react = self._get_react_agent()
        answer, confidence, chunks = await react.run(query, display_query)
        citations = self._extract_citations(chunks)

    else:  # BEYOND_KB
        answer, citations = await self._direct_generate(display_query)

    # ── 4g · 写 session 历史 + 写语义缓存 ──
    if session_id and self._chat_store:
        await self._chat_store.add_message(session_id, "user", query)
        await self._chat_store.add_message(session_id, "assistant", answer)
    if effective_cache_fn:
        await effective_cache_fn(query, {
            "answer": answer, "citations": citations,
            "confidence": self._map_confidence(routing.confidence),
        })

    return OrchestratorResult(answer, citations, confidence, complexity, ...)
```

**关键点**：

- 缓存命中时整个 `run()` 在第 4a 步提前返回；**P1.2 修复前** `semantic_cache_fn` 没人注入导致 `cache_hit` 永远为 False——这个 bug 修复后实测命中率约 30-40%。
- `display_query` 是改写后的查询（用于检索和路由），`query` 是原始用户输入（用于 session 历史），两者**解耦**。
- `signals` 在两个地方用：(1) 喂给路由 LLM 作为 prompt context，(2) 透传到 `HybridSearchEngine` → `DynamicRRFFusion` 选 k（`simple=30 / moderate=60 / complex=90`）。
- OTel span 在每一步包裹：`rag.cache_lookup` / `rag.query_rewrite` / `rag.routing` / `rag.agentic` / `rag.generation`，attributes 含 `complexity` / `routing_confidence` / `cache_hit` / `num_iterations` 等。

---

#### 第 4.5 层 · 路径分支详解

我们这条 `complex` 请求走到 **ReAct 分支**。LangGraph 状态机按以下流程推进：

```
                  ┌─ ReActState(query, rewritten_query, iterations=0,
                  │   retrieved_chunks=[], confidence=0.0)
                  ▼
        ┌──────────────────┐
        │   think_node     │  ← LLM 推理下一步行动
        │   (LLM call 1)   │     prompt: SYSTEM + 当前 state + 已检索 context
        └────────┬─────────┘     输出 JSON: {action, next_query, confidence}
                 │               例: action=retrieve, next_query="供应商A断供影响"
                 ▼
        action == "retrieve" + iterations < 5
                 │
                 ▼
        ┌──────────────────┐
        │ retrieve_node    │  ← 调 HybridSearchEngine.search
        │  (LLM call 无)   │     返回 top-5 chunks
        └────────┬─────────┘
                 │
                 ▼ (回到 think_node)
        ┌──────────────────┐
        │   think_node     │  ← LLM 推理 (LLM call 2)
        └────────┬─────────┘     评估现有 context 够不够
                 │               例: action=retrieve, next_query="A 供应商主要客户"
                 ▼
        retrieve_node (LLM call 无) ─► think_node (LLM call 3)
                 ...                  ...
                 ▼ 当 confidence ≥ 0.85 或 iterations == 5
        ┌──────────────────┐
        │  finish_node     │  ← 生成最终答案 (LLM call N)
        └────────┬─────────┘     prompt: 所有 retrieved_chunks + query
                 │               → structured output (JSON Schema)
                 ▼
              END
```

每个 `think_node` 都是一次轻量 LLM 调用（Router 模型，`max_tokens=512, temperature=0.1`），输出严格 JSON（带 retry）。`finish_node` 才用主力 Generator 模型（`max_tokens=1024, temperature=0.3`）。

**5 步迭代上限 + 0.85 置信度早停**是防止 LLM 反复检索相似内容不推进的核心工程约束。

---

#### 第 5 层 · LLM 生成（`LLMClient.generate_async`）

不管是 ReAct 的 `finish_node` 还是 SIMPLE 路径的 `_generate_answer`，最终都走 `LLMClient.generate_async`：

```python
# backend/generation/llm_client.py
async def generate_async(self, prompt, *, structured_schema=None, use_retry=True, ...):
    call_kwargs = dict(max_tokens=..., temperature=..., structured_schema=..., ...)
    async def _invoke():
        return await self.generator_client.generate_async(prompt, **call_kwargs)

    # 三层保险：Circuit Breaker + with_retry + structured_schema
    if use_breaker and use_retry:
        return await self._generator_breaker.call_async(
            self._invoke_with_retry, _invoke
        )
```

调用链：

```
generate_async()
  └─► generator_breaker.call_async(retry_decorator, _invoke)
        └─► with_retry(_invoke, RetryConfig(max_attempts=3))   # 指数退避
              └─► OpenAICompatibleBackend.generate_async(prompt)
                    └─► OpenAI AsyncClient.chat.completions.create(
                          response_format={"type": "json_schema", ...}
                        )
```

**4 道防线**：
1. **Circuit Breaker**（`failure_threshold=5`）：连续 5 次失败后熔断 30s，期间直接抛 `CircuitOpenError`，主流程降级到空响应。
2. **指数退避重试**（`max_attempts=3`）：429/5xx 等瞬时错误重试。
3. **JSON Schema 约束**：`response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": OUTPUT_SCHEMA}}` 走各 SDK 原生结构化输出（消除 `json.loads` 解析脆弱性）。
4. **错误隔离**：`CircuitBreaker` 按 `(provider, model)` 独立熔断，单一 provider 故障不会波及其他。

`OUTPUT_SCHEMA` 约束生成 `{answer, citations[], confidence}` 三个字段。

---

#### 第 6 层 · 后处理 + 响应

```python
# 6a · 提取引用
citations = []
for chunk in retrieved_chunks[:5]:
    citations.append({
        "doc_id": chunk["doc_id"],
        "chunk_id": chunk["chunk_id"],
        "quote": chunk["text"][:200],
        "score": chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
    })

# 6b · 写 session 历史
await self._chat_store.add_message(session_id, "user",      query)
await self._chat_store.add_message(session_id, "assistant", answer)

# 6c · 写语义缓存
await self._semantic_cache_fn(query, {
    "answer": answer, "citations": citations, "confidence": conf_level,
})

# 6d · 指标记录
self._metrics.record_llm_latency("deepseek-chat", llm_latency)
self._metrics.record_cache_hit(hit=False)         # 写不算命中
# 检索延迟在 HybridSearchEngine 内部已记录

# 6e · OTel span 关闭（自动，BatchSpanProcessor 异步 flush）
```

API 路由把 `OrchestratorResult` 序列化为 `ChatResponse`：

```json
{
  "answer": "供应商A断供将影响以下客户……",
  "sources": [
    {"doc_id": "contract_2024_001", "chunk_id": "c_42",
     "content": "……", "score": 0.94}
  ],
  "session_id": "sess_abc123",
  "confidence": "high",
  "latency_ms": 3421.5
}
```

---

#### 3.1.1 性能与可观测性

整条链路在 Jaeger 看到的 trace 形如：

```
POST /api/chat  ── 3421ms  ──  span.kind=SERVER
├── rag.cache_lookup        ──   2ms   (cache_hit=false)
├── rag.query_rewrite       ──   0ms   (无改写需求)
├── rag.routing             ── 187ms   (complexity=complex, confidence=0.92)
├── rag.agentic             ── 2845ms
│   ├── rag.agentic/think   ── 312ms   (iter 1)
│   ├── rag.agentic/retrieve── 108ms
│   ├── rag.agentic/think   ── 295ms   (iter 2)
│   ├── rag.agentic/retrieve── 112ms
│   ├── rag.agentic/think   ── 287ms   (iter 3)
│   ├── rag.agentic/retrieve──  98ms
│   ├── rag.agentic/think   ── 305ms   (iter 4, confidence=0.88 ≥ 0.85)
│   └── rag.agentic/finish  ── 1328ms
├── rag.generation          ── 386ms   (final answer generation, JSON Schema)
└── (out of orchestrator: session write + cache write)
```

Prometheus 指标增量：
- `rag_retrieval_latency_seconds{stage="total"}` +1（~318ms）
- `rag_llm_latency_seconds{model="deepseek-chat"}` +5（router 4 + generator 1）
- `rag_llm_tokens_total{model="deepseek-chat",type="input"}` +~4k
- `rag_llm_tokens_total{model="deepseek-chat",type="output"}` +~800
- `rag_cache_hit_total{result="miss"}` +1
### 3.2 性能预算（实测）

| 阶段 | P50 | P95 | 备注 |
|------|-----|-----|------|
| Auth + RateLimit | ~2ms | ~5ms | Redis 一次 pipeline |
| Query Rewriter | ~150ms | ~400ms | 命中时 0ms（LRU cache 500） |
| Query Router | ~100ms | ~250ms | 轻量 LLM 1 次调用 |
| Hybrid Search | ~107ms | ~250ms | embed 15 + Qdrant hybrid 10 + RRF 2 + Rerank 80 |
| LLM Generation (simple) | ~850ms | ~2400ms | DeepSeek 主力模型 |
| LLM Generation (react, 5 iter) | ~3000-5000ms | ~8000ms | 5 步 LLM + 最终生成 |
| **总 E2E (simple path)** | **~1.1s** | **~3.3s** | 无缓存命中 |
| **E2E 缓存命中** | **~50ms** | **~100ms** | 仅 Redis FT.SEARCH + fetch |

---

## 4. 核心子系统详解

### 4.1 配置层（`config.py` + `config_schema.py` + `config_loader.py`）

采用 **Pydantic schema + 显式 loader + shim re-export** 三段式：

- **`config_schema.py`**：11 个领域配置 dataclass（`LLMConfig`、`EmbeddingConfig`、`VectorDBConfig`、`SemanticCacheConfig`、`HybridSearchConfig` …），所有字段强类型。
- **`config_loader.py`**：`ConfigLoader.load()` 从 `config.yaml` 读取 → 解析 → 注入环境变量覆盖 → 返回 `AppConfig` 根对象。`get_config()` 用 `lru_cache` 单例化。
- **`config.py`**：仅 re-export，对外 API `from backend.config import get_config` 不变。

**为什么这样拆？** schema 单独可测，loader 可替换（YAML → TOML），shim 保证向后兼容。

### 4.2 检索层（`retrieval/`）

#### 4.2.1 编排：`HybridSearchEngine`

中央编排器，流程：
```
query ──► embedder.embed ──► Qdrant.query_points (Prefetch dense + Prefetch sparse + Fusion RRF)
                              │
                              ▼
                         Top-50 候选
                              │
                              ▼
                Cross-Encoder Rerank (top_k=5)
                              │
                              ▼
                        top-5 chunks
```

**关键设计**：
- **服务端融合**：Qdrant 1.10+ 的 `query_points` 一次 RPC 内 RRF 融合 dense + sparse，避免应用层 chunk_id 重复处理。
- **稀疏向量走 Qdrant native BM25**：传 `models.Document(text=..., model="Qdrant/bm25")` 让服务端算 IDF+TF 归一化，与写入端公式天然一致。
- **tenant filter 注入**：`build_tenant_filter()` 强制 AND 注入 `tenant_id` 条件（`security/tenant.py`），任何调用方都不能绕过。
- **rerank 失败降级**：`is_transient_error` / `is_permanent_error` 分类，错误时回退 RRF top-5。
- **LRU 缓存**：`_RerankCache`（256 容量）缓存 `(query, chunk_id_list) → RerankResult`，相同 query+chunks 命中时跳过 API 调用，节省 80ms + $0.002/q。

#### 4.2.2 融合策略：`DynamicRRFFusion`

按 query complexity 动态选 k：
```python
DEFAULT_K_BY_COMPLEXITY = {
    "simple": 30,    # 小 k → 看重头部（BM25 命中率高）
    "moderate": 60,  # 默认
    "complex": 90,   # 大 k → 各路均衡（dense 提权重）
}
```

#### 4.2.3 重排序：`CrossEncoderReranker`

策略模式 + 工厂：
- **`CohereReranker`**（默认）：API 简单，P50 ~80ms，$2/1K queries，NDCG@10 提升 10-30%
- **`BGEReranker`**（合规/成本敏感）：本地 GPU，~30ms，需 A10G 维护

`get_reranker(provider="cohere"|"bge")` 工厂函数，配置驱动。

#### 4.2.4 查询改写：`QueryRewriter`

解决多轮对话中省略句（"那第二点呢？"）问题。流程：
```
needs_rewriting(query) ?
├─ No (无代词 + 长度OK) → 直接返回原 query
└─ Yes → LLM 改写 → JSON {rewritten, confidence}
              └─ confidence < 0.7 → 回退原 query
```

带 LRU cache（500 容量）按 `(query, last 3 history)` 哈希。

### 4.3 Agentic 层（`agentic/`）

#### 4.3.1 中央编排器：`AgenticOrchestrator`

**这是整个后端的核心**。`run()` 完整流程：

```python
async def run(query, history, session_id, tenant):
    # 1. 缓存检查
    if effective_cache_fn(query): return cached  # 立即返回

    # 2. 会话历史
    if session_id: history = await chat_store.get_history(...)

    # 3. Query 改写
    rewritten = await QueryRewriter().rewrite_async(query, history)

    # 4. Query 信号（纯规则, < 1ms）
    signals = QueryAnalyzer().analyze(rewritten)  # 喂给 Router + Retrieval

    # 5. 路由
    routing = QueryRouter().route(rewritten, history)  # LLM 分类

    # 6. 按复杂度执行
    if SIMPLE:
        chunks, ctx = await hybrid_search.search(rewritten, tenant, signals)
        answer, citations = await _generate_answer(rewritten, chunks)
    elif MODERATE | COMPLEX:
        answer, conf, chunks = await ReActAgent().run(query, rewritten)
        citations = _extract_citations(chunks)
    else:  # BEYOND_KB
        answer, citations = await _direct_generate(rewritten)

    # 7. 写 session + 写 cache
    ...
```

**流式版**（`run_stream`）额外 yield `{"stage": "rewrite|routing|retrieval|generating|done", ...}` SSE 事件，前端可逐阶段展示。

#### 4.3.2 路由：`QueryRouter`

LLM 分类 4 类复杂度：
- `SIMPLE`（默认 67%）：单实体、无推理 → 混合检索
- `MODERATE`（28%）：多实体、需推理 → ReAct
- `COMPLEX`（4%）：多跳关系 → ReAct（多步）
- `BEYOND_KB`：通用问题 → 直 LLM，无检索

**置信度降级**：`confidence < 0.6` 时 SIMPLE→MODERATE→COMPLEX 单向升级（避免降级导致信息丢失）。

**信号注入**：`QueryAnalyzer.analyze()` 的纯规则信号（`has_pronoun`、`is_multi_hop`、`entity_count`…）作为 prompt 上下文喂给 LLM，提升路由准确度。

#### 4.3.3 ReAct Agent：`ReActAgent` (LangGraph)

状态机：
```
START → think → {action=retrieve → retrieve → think
                  action=finish  → finish → END
                  action=think   → think  (loop)
                  iterations ≥ 5 → max_iter → finish}
```

- `max_iterations=5` + `early_stop_threshold=0.85`
- 工具节点：当前仅 `retrieve` + `finish`（ToolRegistry stub 在 P0-3 删除）
- 完整 state 透出 OTel span attributes

#### 4.3.4 信号分析：`QueryAnalyzer`（P2-B6）

纯规则（4 个正则 + 字符串计数），无 LLM，< 1ms，提取：
- `has_pronoun`（中英文代词）
- `entity_count`（英文大写实体 + 中文专名）
- `is_multi_hop`（"对比"/"影响"/"combine" 等关键词）
- `query_length`
- `has_quote`

供 DynamicRRF 选 k + OTel attributes + 路由 fallback。

### 4.4 生成层（`generation/`）

#### 4.4.1 `LLMClient` 门面

**Facade + Factory + Strategy** 三合一：

```python
class LLMClient:
    def __init__(self, generator_provider, generator_model, router_provider, router_model, ...):
        # 双 LLM：Router 用轻量模型, Generator 用主力
        self.generator_client = _create_backend(generator_provider, ...)  # OpenAICompatibleBackend
        self.router_client    = _create_backend(router_provider, ...)     # 同类
        self._generator_breaker = get_breaker(f"llm:{provider}:{model}", failure_threshold=5)
        self._router_breaker    = get_breaker(f"llm:{provider}:{model}", failure_threshold=3)
```

**为什么分 Router/Generator 两个 client**：分层使用可节省 60-70% LLM 成本（Router 只做简单分类）。

**Circuit Breaker 集成**：`generate_async()` 包装在 `_generator_breaker.call_async()` 中，每个 provider 独立熔断（防止雪崩）。

**重试**：`with_retry(RetryConfig(max_attempts=3))` 指数退避，默认开启。

**结构化输出**：通过 `response_format={"type": "json_schema", ...}` 走各 SDK 原生 JSON Schema（消除 `json.loads` 解析脆弱性）。

**流式**：`generate_stream_async` 走 OpenAI stream API，**同步熔断器**（`_on_success_sync` / `_on_failure_sync`）处理异步生成器无法 try/except yield 的痛点。

#### 4.4.2 `PromptBuilder`

- 从 `backend/generation/prompts/v{version}.yaml` 加载（git-tracked，便于 review/diff/rollback）
- `prompt_hash` 暴露给 trace，CI eval diff gate 用 `prompt_hash` 区分 "prompt 改 vs 数据改"
- `build_context(chunks)` → 加引用标注的 context 块
- `build_prompt(query, context)` → 完整 LLM prompt

### 4.5 缓存层（`cache/semantic_cache.py`）

**`RedisSemanticCache`**：基于 `redisvl.AsyncSearchIndex` 的 HNSW 向量索引。

- **写入**：query embedding 归一化 → `redisvl.add({id, query, answer, response_json, embedding})` → 设置 7 天 TTL
- **读取**：query embedding KNN top-1 → cosine similarity ≥ 0.92 视为命中 → 返回 cached response_json
- **统计**：hits / misses / hit_rate

**优雅降级**：redisvl/Redis 不可用时返回 `None`，orchestrator 在 `cache_fn` 内 try/except 跳过缓存逻辑，不阻塞主流程。

### 4.6 摄取层（`ingestion/`）

#### 4.6.1 文档解析：`document_parser.py`

策略：扩展名 → 解析器函数的 registry（P2-3 简化）：
```python
_EXTENSION_PARSERS = {
    ".pdf":  _parse_pdf,        # unstructured hi_res → fallback pypdf
    ".docx": _parse_docx,       # unstructured.partition.docx
    ".md":   _parse_markdown,   # 纯文本 + heading 提取
    ".html": _parse_html,       # html2text → Markdown
}
```

**SimHash 去重**：`SimHashDeduplicator`（64-bit 指纹，海明距离 ≤ 3 视为重复）在 `parse_directory` 全局维护，O(1) 比较。

#### 4.6.2 分块：`chunker.py`

仅保留 `RecursiveChunker`（P1-3 移除 Hierarchical/Semantic）：按 `["\n\n", "\n", ". ", " "]` 逐级递归拆分，`chunk_size=512`，`overlap=64`，`min=150`。

#### 4.6.3 Contextual Retrieval：`embedder.py`

`embed_chunks_with_context()` 是核心创新：
```
原始 chunk text: "条款 3.1 规定..."
                  │
                  ▼ prepend
[Document Context]
{doc_summary}      ← LLM 生成的文档级摘要（一次性，80 tokens）
[Content]
条款 3.1 规定...
```

依据 Anthropic 2024 研究：减少 49% 检索失败。

#### 4.6.4 索引：`indexer.py` + `pipeline.py`

`QdrantIndexer.ensure_collection` 一次创建：
- dense 向量（HNSW, m=16, ef_construct=128）
- sparse 向量（`Modifier.IDF` 让 Qdrant 服务端算 BM25）
- payload 索引：doc_id, chunk_index, section_path, token_count, tenant_id

写入：`index_chunks_with_sparse` 一次 upsert 该文档所有 chunks（dense + sparse 同 point），`with_tenant_payload` 注入 tenant_id。

**完整 pipeline**（`run_index_pipeline`）：
```
parser.parse_file → 
embedder.generate_doc_summary → 
chunker.split → 
embedder.embed_chunks_with_context → 
QdrantIndexer.index_chunks_with_sparse
```

#### 4.6.5 文档注册表：`document_registry.py`

**重大改进（P1-A1）**：用 Qdrant payload-only collection（无 vectors）持久化文档元数据，**取代内存 dict**。重启 / 滚动发布不影响状态可见性。

支持：upload 时内容哈希去重（P1-A4），按 tenant/status 过滤，复合 payload 索引。

#### 4.6.6 异步索引：`workers/index_worker.py`

用 **Arq**（基于 Redis Stream 的轻量 async 任务队列）替代 `BackgroundTasks`：
- 任务持久化，worker 重启可继续
- `max_tries=3` 指数退避
- `run_index_pipeline` 复用（API + Worker 共用同一函数）

### 4.7 安全层（`security/`）

#### 4.7.1 认证：`auth.py`

- **JWT (HS256)** + `python-jose`
- **强制环境变量**：`JWT_SECRET_KEY` / `JWT_PEPPER` 启动时 fail-fast（P1-4 修复，避免默认 pepper 泄露 + 重启失效）
- **bcrpyt 密码哈希** + pepper
- **access_token (30min) + refresh_token (7d)** 双 token 模式
- **依赖注入**：`require_current_user`（强校验）vs `get_current_user`（可选）

#### 4.7.2 多租户：`tenant.py`

**应用层逻辑隔离**：
- 每个 point payload 注入 `tenant_id` 字段
- 检索时通过 `build_tenant_filter()` **强制 AND 注入** tenant 条件（任何调用方都不能绕过）
- 启动时 `ensure_tenant_payload_index()` 创建 payload 索引

**未来可扩展**：物理隔离（每租户独立 collection）vs 逻辑隔离（共享 + filter），当前选逻辑隔离（运维简单 + 跨租户联邦检索可能）。

### 4.8 中间件层（`middleware/`）

#### 4.8.1 请求上下文：`request_context.py` + `request_context_middleware.py`

**P2-8 改造**：用 `ContextVar`（非 threading.local）实现 async-safe 请求作用域：
```
RequestContextMiddleware.dispatch():
  rid = headers["x-request-id"] or uuid4()[:12]
  with RequestContext(request_id=rid, tenant_id=t, session_id=s):
      response = await call_next(request)
      response.headers["x-request-id"] = rid
```

`RequestContextFilter` 在 logging handler 上自动注入 `request_id` / `tenant_id` / `session_id` 到每条 log record。

#### 4.8.2 限流：`rate_limiter.py`

**Redis 滑动窗口**（P1-2 简化：移除 tier 抽象，固定 60 req/min/tenant）：
```python
# ZADD key {now}:now
# ZREMRANGEBYSCORE key 0 (now - 60)
# ZCARD key
# 超过 → ZREM 当前请求 + 429
```

`RateLimitMiddleware` 从 JWT sub 提取 tenant_id，无 token 时 fallback IP。

#### 4.8.3 熔断器：`circuit_breaker.py`

**三态熔断**（CLOSED → OPEN → HALF_OPEN），用于 LLM 调用：
- `failure_threshold=5`（generator）/ `3`（router）
- `recovery_timeout=30s`
- `half_open_max_calls=3` 探测
- 异步（`call_async`）+ 同步 fallback（`_on_success_sync` / `_on_failure_sync`）双接口，适配 streaming 场景

### 4.9 可观测性（`observability/`）

#### 4.9.1 Tracing：`tracing.py`

**OpenTelemetry Facade**：
- `setup_tracing(otlp_endpoint, sample_rate)`：创建 `TracerProvider`，加 OTLP gRPC exporter + (可选) Console exporter
- `create_span(name)` contextmanager：自动异常捕获 + `set_status(ERROR)`
- 8 个 Span Name 常量：`rag.cache_lookup` / `rag.query_rewrite` / `rag.routing` / `rag.retrieval` / `rag.rerank` / `rag.generation` / `rag.agentic` / `rag.embedding`
- `TracingManager` 单例（`lru_cache`），lifespan 启动/关闭

#### 4.9.2 Metrics：`metrics.py`

**Prometheus 指标集**（命名空间 `rag_*`）：

| 指标 | 类型 | 标签 | 用途 |
|------|------|------|------|
| `rag_retrieval_latency_seconds` | Histogram | stage={bm25/dense/fusion/rerank/embedding/total} | 各阶段延迟分布 |
| `rag_cache_hit_total` | Counter | result={hit/miss} | 缓存命中率 |
| `rag_llm_tokens_total` | Counter | model, type={input/output} | 成本核算 |
| `rag_llm_latency_seconds` | Histogram | model | LLM 延迟 |
| `rag_retrieval_chunks_count` | Histogram | query_complexity | 召回数量分布 |
| `rag_retrieval_scores` | Histogram | - | 分数分布 |
| `rag_query_complexity_score` | Histogram | - | 路由置信度分布 |
| `rag_errors_total` | Counter | error_type, component | 错误监控 |
| `rag_active_requests` | Gauge | stage | 并发数 |
| `rag_up` | Gauge | - | 健康 |

**`MetricsCollectorImpl`** 门面：单例（双重检查锁），提供 `record_retrieval_latency` / `record_cache_hit` / `record_llm_tokens` / `record_llm_latency` / `record_error` / `record_active_request` / `set_healthy` 等高阶方法。

`/metrics` 端点：`app.mount("/metrics", make_asgi_app())`。

#### 4.9.3 Health：`health.py`

**`HealthChecker`** + 4 个端点：
- `/health` 详细（JWT 可选，K8s probe 友好）
- `/health/auth` 详细（强制 JWT）
- `/ready` readiness（并发 check Qdrant + Redis，503 on 核心失败）
- `/live` liveness（仅进程存活）

`asyncio.gather(check_qdrant, check_redis)` 并发执行，总耗时 = max 而非 sum。

### 4.10 评估层（`evaluation/`）

#### 4.10.1 RAGAS 评估：`ragas_metrics.py`

5 大指标（`ragas>=0.4`）：
1. **Faithfulness** ≥ 0.85：答案是否被检索上下文支撑
2. **Answer Relevancy** ≥ 0.75：答案是否直接回答问题
3. **Context Precision** ≥ 0.70：top-K 上下文相关块比例
4. **Context Recall** ≥ 0.70：检索覆盖度
5. **Answer Correctness** ≥ 0.80：与 ground_truth 一致性

`evaluate_batch` 同步阻塞调用 → `loop.run_in_executor` 避免阻塞 event loop。

#### 4.10.2 评估持久化：`eval_store.py`

**SQLite 两表**：
- `eval_runs`：run_id / 时间 / 5 指标均值 / 最弱指标 / metadata / **prompt_version / prompt_hash / git_commit**
- `eval_samples`：每条 query 的明细

**Diff Gate**（P2 改造）：对比当前 run 与上一次同 `prompt_hash` 的 run，若关键指标下降 > 0.05 → 失败，CI 拦截回归。

#### 4.10.3 API 触发：`api/eval.py`

仅保留 `POST /api/eval/run`，删除 dashboard 端点（Phase2-2.2：在线评估 + UI 无人维护）。

---

## 5. 数据模型与持久化

| 数据 | 存储 | 用途 |
|------|------|------|
| 向量数据 | Qdrant `enterprise_rag` collection | dense + sparse 双索引 |
| 文档元数据 | Qdrant `document_registry` collection (payload-only) | 状态审计、去重、租户过滤 |
| 用户 | SQLite `data/users.db` | username + bcrpyt(password+pepper) |
| 会话历史 | Redis List `chat:history:{session_id}` (db=1) | TTL 30 天 |
| 语义缓存 | Redis HNSW index `semantic_cache` (db=0) | 7 天 TTL |
| 限流 | Redis ZSET `ratelimit:{tenant_id}` (db=0) | 60s 滑动窗口 |
| Arq 任务 | Redis Stream (db=0) | 持久化任务队列 |
| 评估结果 | SQLite `data/eval_results.db` | run + sample 两表 |
| 评估样本 | `data/eval_results.db` | 单条 query 评估 |
| 日志 | stdout + `RotatingFileHandler` (100MB × 5) | 600MB 上限 |
| 配置 | `config.yaml` + env 覆盖 | 启动时一次加载 |

---

## 6. 主动避开的反模式（Anti-Patterns We Avoid）

| 反模式 | 项目中怎么避开 | 收益 |
|--------|----------------|------|
| **Singleton 全局可变状态** | 改用 `lru_cache` 工厂 + `reset_*_for_test()` 钩子 | 单元测试可控,无 import 副作用 |
| **同步 + 异步双套实现** | 2026 async-only(砍了 `QueryRewriter.rewrite()`) | 维护成本 ½,2026 Python 后端标准 |
| **隐式全局 LRU 缓存** | 砍了 `RerankCache`(命中率 < 1%) | 减少 1 个被面试官追问的"内部状态"点 |
| **领域包 import 横切库** | domain/* 绝不 import platform/* | 单元测试零成本 mock,无环依赖 |
| **可观测性散落在业务代码** | tracing/metrics 通过 port 注入 | 业务代码 0 logger 关注结构 |
| **Prompt 版本漂移** | YAML git-tracked + `prompt_hash` 关联 eval | 区分"prompt 改 vs 数据改" |
| **"在 L1 路径加 cache"的过早优化** | cache_lookup 命中率 < 50% 的不加 | 真实数据驱动决策 |
| **"先存 DB 再同步调用"反模式** | 评估/任务队列全部走异步(Arq/SQLite) | P99 延迟不阻塞主请求 |
| **微服务 / 事件总线** | FastAPI 单体 + Arq 任务队列 | 2026 主流,LangChain/LlamaIndex 都是单体 |
| **GraphRAG / 多模态 / PII 脱敏** | 数据集不匹配 + 作品集 demo 跑不动 | 拒绝功能堆叠(plan 总原则) |

---

## 7. 关键工程实践（bullet 摘要）

- **异步优先** — FastAPI async + redis.asyncio + `asyncio.to_thread` 兜底,无同步/异步双套
- **优雅降级** — 缓存/Reranker/LLM/限流/Embedding/Parser 6 层失败回退,主流程不中断
- **持久化优先** — Qdrant + Redis Stream + SQLite + Arq + git-tracked prompts,重启不丢数据
- **安全纵深** — JWT + 限流 + tenant filter 强制注入 + bcrypt pepper + 熔断 + eval trace 关联
- **可测试性** — FastAPI `Depends` + `lru_cache` + Protocol/ABC + 工厂函数 + reset 钩子
- **可观测性全覆盖** — 4 维:ContextVar 日志 + OTel 8 span + 10+ Prometheus 指标 + liveness/readiness

---

## 8. Why Not

> P3.1:已删除/避免的模块列表已并入 README §5.1 + ARCHITECTURE §6。
>
> 本节仅保留 1 条**架构层**的 Why Not 决策:
>
> | 决策 | 不做 | 原因 |
> |------|------|------|
> | 拆微服务 | FastAPI 单体 | 2026 主流,LangChain/LlamaIndex 都是单体;演示数据集 < 100K chunk 没必要 |
> | K8s manifest | 作品集简历加分有限 | 运维知识不等于 RAG 知识 |
> | Langfuse/Helicone | 自实现 150 行够用 | 商业方案依赖外部账号,demo 跑通风险大 |
> | Pydantic config 四件套 | 保持三件套 | 当前拆得刚好,过度拆 = 过度工程 |
> | 异步/同步双套 | 全 async-only | 2026 Python 后端标准 |

---

## 9. 启动流程（`main.py` lifespan）

```
1. _setup_logging() — stdout + RotatingFileHandler + ContextVar Filter
2. lifespan startup:
   a. deps.get_embedder() — 预热（下载模型、连接）
   b. deps.get_llm_client() — 预热
   c. deps.get_semantic_cache() — 预热（失败则降级）
   d. TracingManager().setup_tracing(otlp_endpoint) — OTel 初始化
3. 注册中间件:
   - RequestContextMiddleware (最外层)
   - CORSMiddleware
   - RateLimitMiddleware (硬失败 if redis missing)
4. 挂载 /metrics
5. 注册路由: /api/health, /api/auth, /api/chat, /api/search,
   /api/documents, /api/stream, /api/eval
6. yield
7. lifespan shutdown:
   - TracingManager().shutdown(timeout=5.0) — flush BatchSpan
```

---

## 10. 总结

整个后端围绕 **"混合检索 + Agentic 路由 + 流式生成"** 三大支柱展开：

- **检索**：Qdrant 服务端 RRF 融合 + Cross-Encoder 精排 + tenant 强制隔离
- **Agent**：LLM 分类 4 类复杂度 → SIMPLE 走混合检索，MODERATE/COMPLEX 走 ReAct 多步推理
- **生成**：分层 LLM（Router 轻量/Generator 主力）+ Circuit Breaker + JSON Schema 结构化输出
- **缓存**：Redis HNSW 语义缓存（cosine 0.92 阈值） + LLM Rerank LRU 缓存
- **可观测**：OTel 8 span + Prometheus 10+ 指标 + ContextVar 日志 + Eval diff gate
- **安全**：JWT + tenant 强制 filter + 限流 + 熔断 + bcrpyt+pepper
- **可维护**：配置 schema 化、prompt git-tracked、registry 模式扩展、工厂函数解耦

**后端总代码量约 14.5k 行 Python**，通过 P0 阶段对 stub / unused 模块的精简，每个文件都对应一个明确的职责，是 FAANG 标准企业级 RAG 系统的最小完整实现。
