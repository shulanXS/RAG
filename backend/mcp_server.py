"""
mcp_server.py — MCP (Model Context Protocol) Server
================================================================================
技术决策记录:
- MCP 是 Anthropic 在 2024-11 推出的 LLM-工具互联协议，2025-12 捐给 Linux Foundation。
- 2026 简历没 MCP 是明显短板 (Anthropic/OpenAI/Google 都已原生支持)。
- 这里暴露 3 个 tools: rag_retrieve / rag_search / rag_cite
  让 Claude Desktop / Cursor / Cline 等 MCP-aware 客户端能直接调 RAG。

设计:
- Transport: stdio (官方推荐) — 与 Claude Desktop 集成最简单
- Tool 命名: 统一 rag_* 前缀，避免与客户端已有工具冲突
- 不走 AgenticOrchestrator (重)，直接调 HybridSearchEngine (快) — MCP 客户端通常
  自己有 LLM，不需要 RAG 内部再调一次 LLM
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# MCP server 启动入口
# ----------------------------------------------------------------------
def main():
    """MCP server 启动入口 — stdio transport"""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        logger.error(
            "mcp 包未安装。请运行: pip install mcp>=1.0.0"
        )
        sys.exit(1)

    # 让 backend.* 顶层模块可被 import
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from backend.api.deps import get_hybrid_search, get_embedder
    from backend.config import get_config

    server = Server("rag-mcp-server")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="rag_retrieve",
                description=(
                    "从 RAG 知识库检索与 query 最相关的文档片段。"
                    "返回 top-k chunks 及其元数据 (doc_id, chunk_index, text, score)。"
                    "不会生成答案 — 调用方 (LLM 客户端) 自己综合多个 retrieve 结果。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "查询文本（用户问题或子问题）",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回的 chunk 数量，默认 5",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "tenant_id": {
                            "type": "string",
                            "description": "多租户隔离用，缺省走 default tenant",
                            "default": "default",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="rag_search",
                description=(
                    "用 query 走完整 RAG 流程 (检索 + Rerank + 生成) — 返回带 citations 的最终答案。"
                    "比 rag_retrieve 慢 (~1-2s) 但适合客户端没有 LLM 的场景。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用户问题",
                        },
                        "tenant_id": {
                            "type": "string",
                            "description": "多租户隔离用，缺省走 default tenant",
                            "default": "default",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="rag_cite",
                description=(
                    "验证 LLM 给出的答案是否能被 KB 检索结果支撑。"
                    "返回 grounded claims 和 unsupported claims (hallucination 风险)。"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "LLM 给出的答案",
                        },
                        "query": {
                            "type": "string",
                            "description": "对应的问题（用于检索）",
                        },
                        "tenant_id": {
                            "type": "string",
                            "description": "多租户隔离用，缺省走 default tenant",
                            "default": "default",
                        },
                    },
                    "required": ["answer", "query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            if name == "rag_retrieve":
                return await _handle_retrieve(arguments)
            elif name == "rag_search":
                return await _handle_search(arguments)
            elif name == "rag_cite":
                return await _handle_cite(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            logger.exception(f"MCP tool {name} failed")
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_retrieve(args: dict) -> list[TextContent]:
        query = args["query"]
        top_k = args.get("top_k", 5)
        tenant_id = args.get("tenant_id", "default")
        cfg = get_config()
        embedder = get_embedder()
        hybrid = get_hybrid_search()
        chunks, ctx = await hybrid.search(query, tenant=tenant_id)
        # Top-k 截断 + 序列化
        top_chunks = chunks[:top_k]
        result = {
            "query": query,
            "tenant_id": tenant_id,
            "num_chunks": len(top_chunks),
            "retrieval_latency_ms": ctx.total_latency_ms,
            "chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_id": c.get("doc_id"),
                    "text": c.get("text", "")[:500],
                    "score": c.get("rerank_score", c.get("rrf_score", 0.0)),
                    "doc_title": c.get("doc_title", ""),
                }
                for c in top_chunks
            ],
        }
        import json
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    async def _handle_search(args: dict) -> list[TextContent]:
        from backend.api.deps import get_orchestrator
        query = args["query"]
        tenant_id = args.get("tenant_id", "default")
        orchestrator = get_orchestrator()
        result = await orchestrator.run(query=query, tenant=tenant_id)
        out = {
            "query": query,
            "answer": result.answer,
            "confidence": result.confidence,
            "complexity": result.complexity.value,
            "cache_hit": result.cache_hit,
            "latency_ms": result.latency_ms,
            "citations": result.citations[:5],  # top-5
        }
        import json
        return [TextContent(type="text", text=json.dumps(out, ensure_ascii=False, indent=2))]

    async def _handle_cite(args: dict) -> list[TextContent]:
        """Verify if answer is grounded in retrieved contexts."""
        from backend.api.deps import get_llm_client
        from backend.generation.citation_verifier import CitationVerifier
        from backend.api.deps import get_hybrid_search

        answer = args["answer"]
        query = args["query"]
        tenant_id = args.get("tenant_id", "default")

        hybrid = get_hybrid_search()
        chunks, _ = await hybrid.search(query, tenant=tenant_id)
        contexts = [c.get("text", "") for c in chunks[:5]]

        llm_client = get_llm_client()
        verifier = CitationVerifier(llm_client=llm_client.generator_client if llm_client else None)
        verification = await verifier.verify(answer=answer, contexts=contexts)

        import json
        return [TextContent(type="text", text=json.dumps(verification, ensure_ascii=False, indent=2))]

    # stdio transport 启动
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
