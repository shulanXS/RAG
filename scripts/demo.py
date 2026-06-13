#!/usr/bin/env python3
"""
demo.py — 端到端 RAG 演示脚本
================================================================================
用法:
    # 默认 ReAct Agent 模式
    python scripts/demo.py --query "这篇文档的核心结论是什么？"

    # Plan-and-Execute 模式
    python scripts/demo.py --agent plan --query "对比各章节的核心论点"

    # 简单混合检索模式
    python scripts/demo.py --agent simple --query "X产品的价格？"

    # 交互式对话
    python scripts/demo.py --interactive

技术决策说明:
- 支持三种 Agent 模式: simple (直接检索) / react / plan
- 交互式对话支持多轮上下文
- 全链路 tracing 输出（各阶段延迟）
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.syntax import Syntax
from rich.logging import RichHandler

from backend.config import get_config, ConfigLoader
from backend.ingestion.embedder import Embedder
from backend.retrieval.hybrid_search import HybridSearchEngine
from backend.agentic import (
    QueryComplexity,
    QueryRouter,
    ReActAgent,
    PlanExecuteAgent,
    AgenticOrchestrator,
)
from backend.generation import LLMClient
from backend.cache import RedisSemanticCache

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=False)],
)
logger = logging.getLogger(__name__)


async def run_query(
    query: str,
    agent_mode: str = "react",
    interactive: bool = False,
) -> None:
    """执行单条查询"""
    config = get_config()

    # 初始化组件
    embedder = Embedder(backend=config.embedding.backend)
    hybrid_search = HybridSearchEngine.from_config(config, embedder)
    llm_client = LLMClient(
        generator_provider=config.llm.generator.provider,
        generator_model=config.llm.generator.model,
        router_provider=config.llm.router.provider,
        router_model=config.llm.router.model,
    )
    router = QueryRouter(
        llm_client=llm_client.router_client,
        complexity_threshold=config.agentic.complexity_threshold,
    )

    # 语义缓存
    semantic_cache = None
    if config.cache.enabled:
        try:
            semantic_cache = RedisSemanticCache(
                host=config.cache.host,
                port=config.cache.port,
                similarity_threshold=config.cache.similarity_threshold,
                ttl_days=config.cache.ttl_days,
            )
        except Exception as e:
            logger.warning(f"语义缓存初始化失败: {e}")

    # 创建编排器
    orchestrator = AgenticOrchestrator(
        hybrid_search_engine=hybrid_search,
        router=router,
        llm_client=llm_client,
    )

    # 定义缓存查询函数
    async def cache_fn(query: str, response: dict = None):
        if semantic_cache is None:
            return None
        if response is None:
            return await semantic_cache.get(query, embedder.embed(query))
        else:
            await semantic_cache.set(query, embedder.embed(query), response)

    # 执行查询
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")

    result = await orchestrator.run(
        query=query,
        conversation_history=[] if interactive else None,
        semantic_cache_fn=cache_fn,
    )

    # =========================================================================
    # 输出结果
    # =========================================================================

    # 答案
    console.print("\n")
    answer_panel = Panel(
        Markdown(result.answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    )
    console.print(answer_panel)

    # 元信息
    meta_table = Table(title="执行元信息", show_header=False, box=None)
    meta_table.add_column("key", style="cyan")
    meta_table.add_column("value", style="white")

    meta_table.add_row("复杂度", result.complexity.value)
    meta_table.add_row("路由置信度", f"{result.routing_confidence:.2f}")
    meta_table.add_row("置信度", result.confidence)
    meta_table.add_row("总延迟", f"{result.latency_ms:.0f}ms")
    meta_table.add_row("缓存命中", str(result.cache_hit))
    meta_table.add_row("查询改写", f"{'是' if result.was_rewritten else '否'}")

    if result.trace.get("routing"):
        meta_table.add_row(
            "路由策略", result.trace["routing"].get("approach", "N/A")
        )
    if result.trace.get("retrieval"):
        ret_trace = result.trace["retrieval"]
        meta_table.add_row(
            "检索块数", str(ret_trace.get("num_chunks", 0))
        )
        meta_table.add_row(
            "检索延迟", f"{ret_trace.get('latency_ms', 0):.0f}ms"
        )

    console.print(meta_table)

    # 引用
    if result.citations:
        cit_table = Table(title="引用来源", show_header=True)
        cit_table.add_column("#", style="dim", width=3)
        cit_table.add_column("来源", style="cyan")
        cit_table.add_column("引用片段", style="white")

        for i, cit in enumerate(result.citations[:5], 1):
            quote = cit.get("quote", "")[:100]
            cit_table.add_row(str(i), cit.get("doc_id", "N/A"), quote + "...")

        console.print(cit_table)

    # 阶段延迟 breakdown
    if result.trace.get("retrieval"):
        latency_table = Table(title="检索延迟 Breakdown", show_header=False, box=None)
        ret_trace = result.trace["retrieval"]
        latency_table.add_row("retrieval", f"{ret_trace.get('latency_ms', 0):.0f}ms")
        console.print(latency_table)

    # Memory Bank 覆盖度
    if result.trace.get("memory_bank"):
        mb = result.trace["memory_bank"]
        console.print(f"\n[dim]Memory Bank — 覆盖率: {mb.get('coverage', 0):.1%} "
                     f"({mb.get('verified_claims', 0)}/{mb.get('total_claims', 0)} claims)[/dim]")


async def interactive_mode(agent_mode: str = "react") -> None:
    """交互式对话"""
    console.print(Panel(
        "[bold]Enterprise RAG — 交互式对话[/bold]\n"
        "输入问题后按回车查询，输入 [bold]quit[/bold] 或 [bold]exit[/bold] 退出\n",
        border_style="cyan",
    ))

    history = []
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            break

        if query.strip().lower() in ("quit", "exit", "q"):
            break

        if not query.strip():
            continue

        await run_query(query, agent_mode=agent_mode, interactive=True)
        history.append({"role": "user", "content": query})


async def run_streaming_query(query: str, agent_mode: str = "react") -> None:
    """Execute a streaming query"""
    from backend.generation.streaming import LLMStreamer
    from backend.config import get_config
    from backend.ingestion.embedder import Embedder
    from backend.retrieval.hybrid_search import HybridSearchEngine
    from backend.agentic import QueryRouter, AgenticOrchestrator
    from backend.generation import LLMClient

    config = get_config()

    embedder = Embedder(backend=config.embedding.backend)
    hybrid_search = HybridSearchEngine.from_config(config, embedder)
    llm_client = LLMClient(
        generator_provider=config.llm.generator.provider,
        generator_model=config.llm.generator.model,
        router_provider=config.llm.router.provider,
        router_model=config.llm.router.model,
    )
    router = QueryRouter(llm_client=llm_client.router_client)

    orchestrator = AgenticOrchestrator(
        hybrid_search_engine=hybrid_search,
        router=router,
        llm_client=llm_client,
    )

    # Run pipeline
    result = await orchestrator.run(query=query)

    # Stream the answer
    streamer = LLMStreamer()
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")
    console.print("[bold green]Answer (streaming):[/bold green] ", end="")

    full_answer = ""
    async for token in streamer.stream(
        prompt=result.trace.get("retrieval_context", f"问题: {query}\n\n请简要回答。"),
        provider=config.llm.generator.provider,
        model=config.llm.generator.model,
    ):
        print(token, end="", flush=True)
        full_answer += token

    console.print()


def main():
    parser = argparse.ArgumentParser(description="RAG 端到端演示")
    parser.add_argument("--query", type=str, help="查询内容")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["simple", "react", "plan"],
        default="react",
        help="Agent 模式 (默认: react)",
    )
    parser.add_argument("--interactive", action="store_true", help="交互式对话")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件")
    parser.add_argument("--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载配置
    if args.config != "config.yaml":
        ConfigLoader.load(args.config)

    try:
        if args.stream:
            asyncio.run(run_streaming_query(args.query or input("Query: ")))
            return
        elif args.interactive:
            asyncio.run(interactive_mode(agent_mode=args.agent))
        elif args.query:
            asyncio.run(run_query(args.query, agent_mode=args.agent))
        else:
            parser.print_help()
    except KeyboardInterrupt:
        console.print("\n[yellow]已退出[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
