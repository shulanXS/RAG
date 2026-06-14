#!/usr/bin/env python3
"""
ingest.py — 一键文档索引脚本
================================================================================
用法:
    python scripts/ingest.py --source data/sample_docs --strategy hierarchical
    python scripts/ingest.py --source /path/to/docs --clear  # 清空后重建

技术决策说明:
- 完整流程: 解析 → 分块 → Contextual Retrieval → Embedding → 索引入库
- 支持增量索引: 已存在的 doc_id 会被自动覆盖
- 进度可视化: 使用 rich 库显示处理进度
- Contextual Retrieval: 每个 chunk embed 前用 Haiku 生成文档级上下文摘要，
  prepend 到 chunk 前（Anthropic 2024: -49% 检索失败）
"""

import argparse
import asyncio
import itertools
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

from backend.config import get_config, ConfigLoader
from backend.ingestion import DocumentParserFactory, get_chunker, Embedder, QdrantIndexer
from backend.ingestion.chunker import count_tokens
from backend.generation.llm_client import LLMClient

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


async def index_documents(
    source_path: str,
    strategy: str = "recursive",
    clear_existing: bool = False,
) -> dict:
    """
    执行完整的文档索引流程

    Args:
        source_path: 文档目录路径
        strategy: 分块策略 (recursive | hierarchical | semantic)
        clear_existing: 是否清空现有索引后重建

    Returns:
        索引统计信息
    """
    # 加载配置
    config = get_config()

    # =========================================================================
    # 步骤 1: 文档解析
    # =========================================================================
    console.print("\n[bold cyan]步骤 1/5: 解析文档[/bold cyan]")
    source = Path(source_path)
    if not source.exists():
        console.print(f"[red]错误: 目录不存在: {source_path}[/red]")
        return {}

    parser = DocumentParserFactory()
    documents = parser.parse_directory(source)

    if not documents:
        console.print("[yellow]警告: 未找到可解析的文档[/yellow]")
        return {}

    console.print(f"  解析完成: {len(documents)} 个文档")

    # 统计
    total_chars = sum(len(d.content) for d in documents)
    total_tokens = sum(count_tokens(d.content) for d in documents)

    # =========================================================================
    # 步骤 2: 分块 + Contextual Retrieval（生成文档摘要）
    # =========================================================================
    console.print("\n[bold cyan]步骤 2/5: 分块处理 + Contextual Retrieval[/bold cyan]")

    # 初始化轻量 LLM（用于生成文档摘要）
    contextual_llm = LLMClient(
        generator_provider=config.llm.generator.provider,
        generator_model=config.llm.generator.model,
        router_provider=config.llm.router.provider,
        router_model=config.llm.router.model,
    )
    console.print(f"  Contextual LLM: {config.llm.generator.model}")

    embedder = Embedder(
        backend=config.embedding.backend,
        contextual_llm_client=contextual_llm.generator_client,
        contextual_prefix_tokens=config.embedding.contextual_prefix_tokens,
    )
    console.print(f"  Embedder: {config.embedding.backend} ({config.embedding.voyage_model})")

    chunker = get_chunker(
        strategy=strategy,
        config={
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
            "min_chunk_size": config.chunking.min_chunk_size,
            "heading_levels": config.chunking.heading_levels,
        },
    )

    all_chunks = []
    chunk_stats = {"total": 0, "kept": 0, "dropped": 0, "avg_size": 0}
    doc_summaries: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"  处理文档中 ({strategy})...", total=len(documents))

        for doc in documents:
            # 为文档生成上下文摘要（Anthropic 2024 contextual retrieval）
            doc_summary = await embedder.generate_doc_summary(doc.content, doc.doc_id)
            doc_summaries[doc.doc_id] = doc_summary

            chunks = chunker.split(
                text_units=doc.text_units,
                doc_id=doc.doc_id,
                metadata={
                    "headings": doc.metadata.get("headings", []),
                    "doc_summary": doc_summary,
                },
            )
            # 过滤碎片块
            valid_chunks = [c for c in chunks if c.token_count >= config.chunking.min_chunk_size]
            all_chunks.extend(valid_chunks)

            chunk_stats["total"] += len(chunks)
            chunk_stats["kept"] += len(valid_chunks)
            chunk_stats["dropped"] += len(chunks) - len(valid_chunks)

            progress.advance(task)

    summaries_count = sum(1 for s in doc_summaries.values() if s)
    console.print(f"  分块完成: {chunk_stats['kept']} 个有效块, "
                 f"{chunk_stats['dropped']} 个被丢弃 (小/空)")
    console.print(f"  文档摘要: {summaries_count}/{len(documents)} 个已生成")

    # =========================================================================
    # 步骤 3: 初始化 Qdrant Indexer
    # =========================================================================
    console.print("\n[bold cyan]步骤 3/5: 初始化 Qdrant Indexer[/bold cyan]")

    indexer = QdrantIndexer(
        url=config.vector_db.url,
        collection_name=config.vector_db.collection_name,
        vector_size=config.vector_db.vector_size,
        distance=config.vector_db.distance,
        batch_size=config.vector_db.batch_size,
    )
    indexer.ensure_collection(with_sparse_index=True)
    console.print(f"  Qdrant: {config.vector_db.url}/{config.vector_db.collection_name}")

    # =========================================================================
    # 步骤 4: Embedding + 索引入库
    # =========================================================================
    console.print("\n[bold cyan]步骤 4/5: Embedding + 索引入库[/bold cyan]")

    indexed_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total_chunks = len(all_chunks)
        task = progress.add_task("  Embedding + 入库...", total=total_chunks)

        batch_size = config.embedding.batch_size
        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i:i + batch_size]

            # 按 doc_id 分组，每组使用同一个 doc_summary 做 contextual embedding
            for doc_id, doc_chunks in itertools.groupby(batch, key=lambda c: c.doc_id):
                doc_chunks = list(doc_chunks)
                doc_summary = doc_summaries.get(doc_id, "")

                # Contextual Retrieval: 使用 prepend 了文档上下文的 embedding
                embeddings = embedder.embed_chunks_with_context(doc_chunks, doc_summary)

                # P1.1: 使用 sparse+Modifier.IDF 写双路索引（Qdrant 服务端 BM25）
                indexer.index_chunks_with_sparse(
                    chunks=doc_chunks,
                    dense_embeddings=embeddings,
                    doc_id=doc_id,
                )
                indexed_count += len(doc_chunks)

            progress.advance(task, len(batch))

    console.print(f"  索引完成: {indexed_count} 个 chunks 已写入 Qdrant")

    # =========================================================================
    # 步骤 5: 汇总报告
    # =========================================================================
    console.print("\n[bold cyan]步骤 5/5: 索引汇总[/bold cyan]")

    info = indexer.get_collection_info()

    table = Table(title="索引统计", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    table.add_row("解析文档数", str(len(documents)))
    table.add_row("总字符数", f"{total_chars:,}")
    table.add_row("总 token 数 (估算)", f"{total_tokens:,}")
    table.add_row("分块策略", strategy)
    table.add_row("有效 chunks", str(chunk_stats["kept"]))
    table.add_row("丢弃 chunks", str(chunk_stats["dropped"]))
    table.add_row("Qdrant points", f"{info.get('points_count', 0):,}")

    console.print(table)

    return {
        "documents": len(documents),
        "total_tokens": total_tokens,
        "total_chunks": chunk_stats["kept"],
        "dropped_chunks": chunk_stats["dropped"],
        "strategy": strategy,
        "qdrant_points": info.get("points_count", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 文档索引工具")
    parser.add_argument(
        "--source",
        type=str,
        default="data/sample_docs",
        help="文档目录路径",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["fixed", "recursive", "hierarchical", "semantic"],
        default="recursive",
        help="分块策略 (默认: recursive)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="清空现有 collection 后重建",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载配置
    if args.config != "config.yaml":
        ConfigLoader.load(args.config)

    console.print("[bold green]Enterprise RAG — 文档索引工具[/bold green]")
    console.print(f"  文档目录: {args.source}")
    console.print(f"  分块策略: {args.strategy}")
    console.print(f"  配置文件: {args.config}")

    try:
        result = asyncio.run(
            index_documents(
                source_path=args.source,
                strategy=args.strategy,
                clear_existing=args.clear,
            )
        )

        if result:
            console.print("\n[bold green]索引完成![/bold green]")
        else:
            console.print("\n[yellow]未处理任何文档[/yellow]")
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]索引已取消[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]索引失败: {e}[/red]")
        logger.exception("索引过程异常")
        sys.exit(1)


if __name__ == "__main__":
    main()
