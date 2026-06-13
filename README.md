# Enterprise-Grade RAG System

> 2026 年 FAANG 标准企业级 RAG 完整架构，涵盖 Hybrid Search、Contextual Retrieval、Agentic Orchestration、RAGAS Evaluation 全链路。

---

## 项目亮点（简历关键词）

| 技术亮点 | 说明 |
|---------|------|
| **Hybrid Search** | BM25 + Dense 向量双路并行，RRF 融合，解决 naive RAG 40% 检索失败问题 |
| **Contextual Retrieval** | Anthropic 2024 方法论：embedding 前为每个 chunk 添加文档级上下文摘要，减少 49% 检索失败 |
| **Cross-Encoder Reranking** | 两阶段检索：top-50 粗召回 → Cross-Encoder 精排到 top-5，NDCG@10 提升 10-30% |
| **Agentic Orchestration** | LangGraph 实现 Router + ReAct + Plan-and-Execute 三层递进架构 |
| **Memory Bank** | claim-evidence 链路追踪，监管行业可解释性必备 |
| **Semantic Cache** | Redis FT.SEARCH 语义缓存，cosine ≥ 0.92 命中，节省 40-80% LLM 调用成本 |
| **RAGAS Evaluation** | 五指标量化评估体系（Faithfulness / Answer Relevancy / Context Precision / Recall / Correctness），CI/CD 集成 |
| **Pydantic 配置管理** | YAML + Pydantic 类型验证，启动时捕获配置错误 |

---

## 架构图

### 全局架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              用户请求 (Query)                                    │
│                    [身份认证 → ACL 解析 → Query Rewrite]                         │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │        Query Router (Haiku)          │
                    │   复杂度分类: simple / moderate /    │
                    │   complex / beyond_kb                  │
                    └──────────────────┬──────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌──────────────────┐     ┌─────────────────────────────────┐     ┌──────────────────┐
│  Semantic Cache  │     │     Hybrid Search Engine         │     │  Direct LLM      │
│  (Redis)        │     │                                 │     │  (Beyond KB)     │
│  cosine≥0.92    │     │  ┌─────────┐    ┌──────────┐  │     └──────────────────┘
│  hit: ~50ms     │     │  │  BM25   │ RRF│  Dense   │  │
└──────────────────┘     │  │ (top-50)│ ←→│(top-50) │  │
                         │  └────┬────┘    └────┬─────┘  │
                         └───────┼───────────────┼─────────┘
                                 │               │
                                 └───────┬───────┘
                                         ▼
                               ┌──────────────────┐
                               │  Cross-Encoder   │
                               │  Reranker        │
                               │  (Cohere Rerank) │
                               │  top-50 → top-5  │
                               └──────────┬─────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
           ┌────────────────┐   ┌────────────────┐   ┌──────────────────┐
           │  Simple Path   │   │  ReAct Agent   │   │ Plan-Execute    │
           │  (直接生成)     │   │  (LangGraph)   │   │  (LangGraph)    │
           └────────┬───────┘   └────────┬───────┘   └────────┬────────┘
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          ▼
                              ┌──────────────────────┐
                              │    Memory Bank       │
                              │  (Claim-Evidence)   │
                              │  证据链路追踪         │
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  Structured Output   │
                              │  JSON Schema 约束    │
                              │  (Claude/GPT-4o)    │
                              └──────────────────────┘
```

### 索引管道 (Ingestion Pipeline)

```
原始文档 (PDF/DOCX/MD/HTML)
           │
           ▼
┌────────────────────────┐
│   Document Parser      │
│  (Unstructured)        │
│  多格式 → Markdown     │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   Chunking Strategy    │
│  Recursive/Hierarchical│
│  /Semantic             │
│  chunk_size=512,      │
│  overlap=64, min=150  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Contextual Embedding  │  ← Anthropic 2024 方法论
│ Haiku 为每个 chunk    │
│ 生成文档级上下文摘要    │
│ prepend 50-100 tokens │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Embedding Model      │
│  (voyage-3-large /   │
│   text-embedding-3-large)│
│  dimension: 1024      │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Qdrant Indexer       │
│  HNSW + Sparse Vector │
│  Batch upsert         │
└────────────────────────┘
```

### ReAct Agent 状态机 (LangGraph)

```
START
  │
  ▼
┌──────────────┐
│    Think     │ ← LLM 推理下一步行动
│   (node)     │
└──────┬───────┘
       │
       ├───────────────────────┐
       ▼                       ▼
 action=retrieve          action=finish
       │                       │
       ▼                       ▼
┌──────────────┐      ┌──────────────┐
│   Retrieve   │      │    Finish   │
│   (node)     │      │  (生成答案) │
└──────┬───────┘      └──────────────┘
       │
       ▼
  ┌────────────┐
  │ iterations │───(No)──→ Think
  │   < max?  │───(Yes)──→ Max Iter → Finish
  └────────────┘
```

---

## 技术决策与 BQ 面试素材

### 决策 1：为什么需要 Hybrid Search？

**问题背景**：纯向量检索在精确标识符（SKU、合同号、政策编号）上召回率极差，因为这些词在训练语料中很少出现；纯 BM25 无法处理语义 paraphrase。

**数据支撑**：行业数据显示 naive RAG 在约 40% 的查询上检索阶段就失败了。

**决策过程**：

| 方案 | 精确匹配 | 语义理解 | 融合难度 | 2026 状态 |
|------|---------|---------|---------|---------|
| 纯向量 | 差 | 强 | 无 | 不推荐生产 |
| 纯 BM25 | 强 | 差 | 无 | 不推荐生产 |
| **Hybrid + RRF** | **强** | **强** | **中** | **FAANG 标准** |

**最终决策**：BM25 + Dense 双路并行，RRF (Reciprocal Rank Fusion) 融合。RRF 的核心优势是不需要 score 归一化，k=60 参数对不同量纲天然鲁棒。

**风险**：引入 BM25 意味着额外维护一路索引栈。选择 Qdrant native sparse vector 避免引入 Elasticsearch 的运维复杂度。

---

### 决策 2：为什么需要 Contextual Retrieval？

**问题背景**：embedding 模型看到的是孤立的 chunk，没有文档级上下文。「第三章讨论的X观点」类型的检索，chunk 本身不包含章节信息。

**研究依据**：Anthropic 2024 年 contextual retrieval 论文证明：embed 前 prepend 50-100 token 的文档上下文摘要，减少 49% 检索失败；叠加 Reranker 后提升至 67%。

**决策过程**：

| 方案 | 精度 | 成本 | 实现 |
|------|------|------|------|
| 固定前缀 (doc_title) | 低 | 极低 | 1 行 |
| **LLM 生成摘要块** | **高** | **中** | **采用** |
| Document Summary Chaining | 最高 | 高 | 过度设计 |

**最终决策**：使用 Haiku 4.5 生成 50-100 token 摘要作为每个 chunk 的 prefix。成本可忽略（$0.8/1M tokens），ROI 最高。

---

### 决策 3：为什么需要两阶段检索（粗召回 + 精排）？

**问题背景**：Bi-encoder 在 query 和 doc 独立编码时缺乏深度交互，导致「看起来相似但实际不相关」的 chunk 被误召回。

**Cross-Encoder 原理**：将 query+doc 联合编码，通过 Transformer attention 捕获细粒度相关性。

**选型对比**：

| 方案 | NDCG@10 提升 | 延迟 | 成本 |
|------|------------|------|------|
| Bi-encoder only | 基准 | ~10ms | 低 |
| **Cross-encoder Reranker** | **+10-30%** | **~80-100ms** | **中** |
| LLM-as-Reranker | 最高 | >1s | 极高 |

**最终决策**：两阶段：top-50 RRF 融合结果 → Cohere Rerank 3.5 精排到 top-5。Cohere Rerank 3.5 是 2026 年企业默认值（$2/1K 查询，~80ms）。合规/成本敏感场景迁移到 BGE Reranker v2-m3 自托管。

**权衡**：每条查询增加约 80ms 延迟，但 LLM 上下文从 500 chunks 压缩到 5 chunks，LLM 生成时间减少 60%，净效果是端到端延迟降低。

---

### 决策 4：为什么需要 Agentic 编排？

**问题背景**：线性「检索 → 生成」管道无法处理复杂多跳问题，也浪费资源在简单查询上。

**决策过程**：从简单到复杂，渐进式引入 Agentic 能力：

| 模式 | 适用场景 | 复杂度 | 决策 |
|------|---------|--------|------|
| Router | 简单分类+路由 | ⭐ | **必须** |
| ReAct | 推理+工具调用 | ⭐⭐ | **采用** |
| Plan-and-Execute | 复杂分析 | ⭐⭐⭐ | **采用** |

**最终决策**：LangGraph 实现 Router + ReAct + Plan-and-Execute 三层递进。用 Haiku 4.5 做复杂度分类，Sonnet 做生成。分层使用模型，每年可节省 60-70% LLM 成本。

**风险缓解**：max_iterations=5 防止无限循环；early_stop_threshold=0.85 置信度达标时提前退出。

---

### 决策 5：向量数据库选型

**候选对比**：

| 数据库 | 规模 | 混合搜索 | 自托管成本 | 决策 |
|--------|------|---------|----------|------|
| Pinecone | 无上限 | 需外部 BM25 | 托管 | 3-4x 溢价 |
| **Qdrant** | **10M-100M** | **✅ 原生** | **~$50/节点** | **✅ 选用** |
| pgvector | <10M | ✅ ParadeDB | PG 成本 | 零迁移首选 |
| Milvus | >100M | ✅ | 高运维 | GPU 场景 |

**最终决策**：Qdrant。Docker 一行启动，HNSW+mmap 零拷贝，sparse vector 原生支持 Hybrid Search。对于 1M 以下向量，pgvector 是更轻量替代。

---

## 项目结构

```
rag_system/
├── config.yaml                  # 全局配置（9 个配置块）
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py               # Pydantic 配置加载器
│   │
│   ├── ingestion/              # 索引管道 (5 files)
│   │   ├── document_parser.py  # 多格式解析 (PDF/DOCX/MD/HTML)
│   │   ├── chunker.py         # 智能分块 (Recursive/Hierarchical/Semantic)
│   │   ├── embedder.py        # 多后端 Embedding (Voyage/OpenAI/BGE)
│   │   └── indexer.py         # Qdrant 索引写入
│   │
│   ├── retrieval/              # 混合检索 (7 files)
│   │   ├── bm25_retriever.py  # BM25 关键词检索
│   │   ├── vector_retriever.py # Qdrant HNSW 向量检索
│   │   ├── fusion.py          # RRF + Weighted 融合
│   │   ├── reranker.py        # Cross-Encoder 精排 (Cohere/BGE)
│   │   ├── query_rewriter.py  # 多轮对话改写
│   │   ├── hyde.py            # HyDE 假设性答案
│   │   └── hybrid_search.py   # 混合检索编排引擎
│   │
│   ├── agentic/               # Agentic 编排 (5 files)
│   │   ├── query_router.py    # 查询复杂度路由器
│   │   ├── memory_bank.py     # Claim-Evidence 追踪
│   │   ├── react_agent.py     # ReAct Agent (LangGraph)
│   │   ├── plan_execute.py    # Plan-and-Execute (LangGraph)
│   │   └── orchestrator.py    # 中央编排器
│   │
│   ├── generation/            # 生成层 (3 files)
│   │   ├── llm_client.py      # 统一 LLM 接口
│   │   ├── prompt_builder.py  # Prompt 模板 + 上下文组装
│   │   └── structured_output.py # JSON Schema 输出
│   │
│   ├── cache/                # 语义缓存 (1 file)
│   │   └── semantic_cache.py  # Redis 语义缓存
│   │
│   └── evaluation/            # 评估模块 (3 files)
│       ├── ragas_metrics.py   # RAGAS 五指标
│       ├── deepeval_tests.py  # DeepEval pytest 套件
│       └── test_dataset.py   # 30 条标注测试集
│
├── scripts/                   # 入口脚本
│   ├── ingest.py             # 一键索引
│   ├── demo.py               # 端到端演示
│   └── eval.py               # 评估报告
│
└── tests/                    # 单元测试
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_agentic.py
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 3. 配置 API Keys
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export COHERE_API_KEY=your_key

# 4. 索引文档
python scripts/ingest.py --source data/sample_docs --strategy hierarchical

# 5. 端到端查询
python scripts/demo.py --query "这篇文档的核心结论是什么？"

# 6. 评估报告
python scripts/eval.py --report

# 交互式对话
python scripts/demo.py --interactive --agent react
```

---

## 评估指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| Faithfulness | ≥ 0.85 | 答案是否被检索上下文支撑 |
| Answer Relevancy | ≥ 0.75 | 答案是否直接回答问题 |
| Context Precision | ≥ 0.70 | top-K 上下文中相关块的比例 |
| Context Recall | ≥ 0.70 | 检索上下文覆盖必要信息的程度 |
| Answer Correctness | ≥ 0.80 | 与 ground truth 的一致性 |

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 文档解析 | Unstructured, pypdf, python-docx |
| Embedding | Voyage AI, OpenAI, BGE-M3 |
| 向量存储 | Qdrant (HNSW + Sparse) |
| 检索融合 | rank_bm25, RRF |
| Reranker | Cohere Rerank 3.5, BGE Reranker v2-m3 |
| Agent 编排 | LangGraph, LangChain |
| LLM 生成 | Claude 3.7 Sonnet, GPT-4o |
| 语义缓存 | Redis / Valkey (FT.SEARCH) |
| 评估 | RAGAS, DeepEval |
| 配置管理 | Pydantic, PyYAML |

---

## 面试亮点提示

在面试中强调以下内容：

1. **Hybrid Search 的 RRF 融合**：解释为什么不需要 score 归一化，k=60 的由来
2. **两阶段检索的延迟分析**：Reranker 增加 80ms，但 LLM 生成时间减少 60%，净效果是端到端延迟降低
3. **Contextual Retrieval 的实现细节**：为什么 prepend 上下文摘要比 full-text 好
4. **Agentic 系统的失败模式**：如何用 max_iterations 和置信度阈值防止无限循环
5. **评估驱动开发**：RAGAS 五指标体系，以及 CI/CD 集成的方式
6. **配置管理的设计**：为什么用 Pydantic 而非 dict，确保启动时验证
