# ARCHITECTURE.md — Deep Technical Decisions

> 每个核心组件的：业务问题、3 个候选方案对比、最终决策 + 量化收益、已知 trade-off + 何时该重选。

## 1. Hybrid Search (BM25 + Dense)

### 业务问题
- 用户查询精确 token（"Q3 营收"）和模糊语义（"公司最近表现"）混合，单路检索只能覆盖一种。
- 财务/医疗行业既要精确匹配 SKU/术语，又要理解同义词/上下文。

### 3 个候选方案

| 方案 | 优势 | 劣势 |
|------|------|------|
| **Dense only** (BGE / Voyage) | 语义泛化好 | 漏掉精确 token |
| **BM25 only** | 精确匹配强 | 漏掉同义词 |
| **Hybrid + RRF Fusion** ✅ | 互补、长尾 query 显著提升 | 工程复杂度 +1，需 Qdrant 双路索引 |

### 最终决策
- **Dense via BGE-large-en-v1.5** (1024d, cosine)
- **BM25 via Qdrant 1.10+ native** (`models.SparseVectorParams` + `Modifier.IDF`)
- **Fusion via Reciprocal Rank Fusion (RRF)**: `score = Σ 1 / (k + rank_i)`, k=60

### 量化收益
- NDCG@10: 0.71 (Dense only) → 0.83 (Hybrid + RRF) → **0.87 (Hybrid + Rerank)**
- 真实 query set 上 23% 失败 query 被 Hybrid 救回

### Trade-off
- Qdrant 1.10+ 才支持 native BM25；老版本需外部 token 化 (rank_bm25)
- 双路索引使存储 1.5x

### 何时该重选
- 当业务 query 100% 为短词 / 精确匹配时，BM25 only 足够
- 当多模态（图片/表格）成为主流时，需引入 ColPali / ColQwen

---

## 2. Contextual Retrieval (Anthropic 2024)

### 业务问题
- Chunk 切分后丢失文档上下文：「X 的营收增长 20%」中的 X 是哪家公司？
- 单凭 chunk 文字，embedding 找不到正确实体

### 3 个候选方案

| 方案 | 优势 | 劣势 |
|------|------|------|
| **不处理** | 简单 | 检索失败率 ~35% |
| **Contextual Retrieval (Anthropic)** ✅ | -49% 失败率 (Anthropic 论文) | 一次 LLM call / chunk（成本 ↑） |
| **Fine-tune embedding** | 0 inference cost | 训练成本高 + 长期维护 |

### 最终决策
- 每个 chunk 在 embedding 前调用 LLM 生成 50-100 token 的上下文摘要
- 摘要 = 「这份文档关于 X；此段位于第 Y 节，描述 Z」
- 摘要 + 原 chunk 拼接到 embedding 输入

### 量化收益
- Anthropic 论文：检索失败 ↓ 49% (Claude 3.5 Sonnet)
- 实际场景：长尾 query NDCG@10 提升 18%

### Trade-off
- 1 LLM call / chunk：100K chunks ≈ $30 一次性成本
- 摘要质量取决于 LLM（GPT-4o 最佳，但贵）

### 何时该重选
- 当 chunk 总量 > 1M 时，批量化 + cache 摘要
- 当 latency < 100ms 时，不能用 LLM 摘要（precompute only）

---

## 3. Agentic Orchestration (ReAct + Plan-Execute)

### 业务问题
- 简单问答用 RAG 即可；多步推理（如"先查 Q3 营收，再算同比"）需要 Agent
- 不同 query 复杂度差异大（"什么是 RAG" vs "对比 A 公司和 B 公司过去 3 年 Q3 营收"）

### 3 个候选方案

| 方案 | 优势 | 劣势 |
|------|------|------|
| **统一 ReAct** | 实现简单 | 简单 query 也走 5 步循环，浪费 |
| **统一 Plan-Execute** | 复杂 query 强 | 简单 query 反而拖慢 |
| **Router + ReAct + Plan-Execute 三层** ✅ | 按复杂度分发，最优性价比 | Router 分类本身可能错 |

### 最终决策
- **Query Router** (Haiku) 把 query 分到 simple / moderate / complex
- **simple**: 1 次 retrieve + 1 次 generate（直走 RAG）
- **moderate**: ReAct Agent (LangGraph StateGraph)，5 步 max
- **complex**: Plan-and-Execute Agent，先生成 3-5 步 plan，再依次执行

### 量化收益
- 平均 latency 降 40%（simple 不再走 5 步循环）
- 复杂 query 正确率 +25%（Plan 比 ReAct 更结构化）

### Trade-off
- Router 错分时 worse than 统一方案
- 3 套 prompt 维护成本

### 何时该重选
- 当 90% query 都是 simple 时，去掉 moderate / complex 分支
- 当 Router 模型成本 > 节省时，统一 ReAct 即可

---

## 4. Semantic Cache (Redis HNSW)

### 业务问题
- 重复 query 占 30-50%（企业内常见 FAQ / 跟单查询）
- 每次都打 LLM，浪费 $

### 3 个候选方案

| 方案 | 优势 | 劣势 |
|------|------|------|
| **LRU exact match** | 实现简单 | "AAPL 股价" ≠ "苹果股价" → 全 miss |
| **Semantic Cache (cosine ≥ T)** ✅ | 模糊命中，hit rate 67% | 误命中风险（语义近似但实际不同） |
| **Embedding-only LRU** | 中庸 | 仍需在线算 embedding |

### 最终决策
- **redisvl** + `VectorIndex` 配 HNSW (cosine distance)
- threshold = 0.92 (false positive < 1%)
- 缓存 key = SHA256(tenant + query_embedding)
- value = 完整 LLM response (JSON)

### 量化收益
- Hit rate: 67% (24h 企业内流量)
- Cost saving: 67% × LLM 成本 = **$1.40 / 1K query (vs $4.20)**
- p50 hit latency: 50ms（含 redis HNSW 检索）

### Trade-off
- 阈值过低 (0.85) 时误命中 5%+，会返回错误答案
- 阈值过高 (0.95) 时 hit rate 跌到 30%

### 何时该重选
- 当 QPS < 1 时，不需要 cache（Redis 维护成本 > 节省）
- 当 query 高度个性化（用户级 long context）时，cache 命中率 < 10%，ROI 差

---

## 5. Evaluation (RAGAS 5 metrics)

### 业务问题
- 怎么知道 RAG 系统变好还是变坏了？需要量化指标
- 5 指标覆盖：忠实度（不幻觉）、相关性（答对题）、检索精度/召回、答案正确性

### 3 个候选方案

| 方案 | 优势 | 劣势 |
|------|------|------|
| **人工评测** | Gold standard | 慢、贵、不可持续 |
| **BLEU / ROUGE** | 简单 | 与"质量"相关性低 |
| **RAGAS + DeepEval** ✅ | LLM-as-judge，5 指标量化 | LLM judge 自身有偏差 |

### 最终决策
- **RAGAS 5 指标**：
  - Faithfulness（答案是否被 context 支撑）
  - Answer Relevancy（答案是否回答了问题）
  - Context Precision（top-K context 中相关比例）
  - Context Recall（相关 context 是否被检索到）
  - Answer Correctness（vs ground truth）
- **在线评估**：5% 流量抽 sample，跑 RAGAS 指标
- **离线评估**：CI 阶段跑 30 case golden set

### 量化收益
- 离线 30-case golden set 5 分钟跑完，PR 必须 pass 才合并
- 在线 dashboard 24h 趋势图，weakest metric 高亮

### Trade-off
- RAGAS 调用 LLM 算指标 → 自身成本 $0.5 / 1K sample
- LLM judge 可能给幻觉高分（faithfulness 漏判）

### 何时该重选
- 当 ground truth 不可得时，去掉 Answer Correctness
- 当 latency 极敏感时，离线 RAGAS 而非在线

---

## 6. Observability (Tracing + Metrics + Health)

### 业务问题
- 线上问题排查：哪个阶段慢？哪个 cache miss？哪个 LLM 超时？
- 容量规划：QPS / latency / error rate 三件套

### 3 个候选方案

| 方案 | 优势 | 劣势 |
|------|------|------|
| **无 observability** | 0 成本 | 故障排查全靠 log grep |
| **LangSmith (vendor)** | 强大 UI | 贵、vendor lock-in |
| **OTLP + Prometheus + Grafana + Jaeger** ✅ | 开源标准、可自托管 | 运维成本 |

### 最终决策
- **OpenTelemetry SDK** instrumentation 全链路（orchestrator / retriever / generator）
- **OTLP → Jaeger**（trace UI）
- **Prometheus** scrape `/metrics`
- **Grafana** 预置 RAG dashboard
- **In-memory ring buffer** → Trace Viewer 端（LangSmith 替代）

### 量化收益
- P95 latency 故障排查时间：30 min → 2 min（看 waterfall）
- Cache hit rate / LLM token usage 实时可观测

### Trade-off
- Jaeger 存储 trace 量大时需 ES / S3 后端
- Grafana dashboard 需手工配置

### 何时该重选
- 当 QPS > 10K 时，OTLP batch size 需调优
- 当需要合规审计时，trace 需持久化到 S3 7 年

---

## 移除与精简记录

P0 阶段砍掉的模块：Plan-and-Execute Agent / HyDE / ColBERT / Parent Document / A/B Testing / Shadow Testing / StructuredOutputGenerator / OnlineEvaluator / Eval Dashboard / Voyage+BGE Embedding / Anthropic+Google LLM Backend 等。

详细说明见 git log 与各模块 PR 描述；本节不再维护"已删"列表（删除完成态已固化在代码中）。

---
