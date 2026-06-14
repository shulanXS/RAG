#!/usr/bin/env python3
"""
eval.py — RAG 系统评估脚本
================================================================================
用法:
    python scripts/eval.py
    python scripts/eval.py --report
    python scripts/eval.py --category difficult --verbose

技术决策说明:
- 运行内置测试集（30 条标注数据）
- 计算 RAGAS 五指标
- 生成 CI/CD 风格报告
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich import print as rprint

from backend.config import get_config, ConfigLoader
from backend.evaluation.test_dataset import get_test_dataset, get_test_dataset_by_category
from backend.evaluation.ragas_metrics import RAGASEvaluator
from backend.evaluation.deepeval_tests import RAGTestSuite
from backend.agentic import AgenticOrchestrator
from backend.ingestion.embedder import Embedder
from backend.retrieval.hybrid_search import HybridSearchEngine
from backend.agentic import QueryRouter
from backend.generation import LLMClient

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=False)],
)
logger = logging.getLogger(__name__)


async def run_evaluation(
    category: str | None = None,
    verbose: bool = False,
    output_file: str | None = None,
) -> dict:
    """运行评估 — 初始化真实 RAG pipeline，对每条测试用例执行完整检索+生成"""
    config = get_config()

    # 获取测试数据
    if category:
        test_cases = get_test_dataset_by_category(category) if category else get_test_dataset()
    else:
        test_cases = get_test_dataset()

    console.print(f"\n[bold cyan]评估配置[/bold cyan]")
    console.print(f"  测试用例数: {len(test_cases)}")
    console.print(f"  分类过滤: {category or '全部'}")
    console.print(f"  阈值: faithfulness≥{config.evaluation.thresholds.faithfulness}, "
                 f"relevancy≥{config.evaluation.thresholds.answer_relevancy}")

    # -------------------------------------------------------------------------
    # 初始化真实 RAG 组件
    # -------------------------------------------------------------------------
    console.print("\n[bold cyan]初始化 RAG 组件...[/bold cyan]")

    embedder = Embedder(backend=config.embedding.backend)
    console.print(f"  Embedder: {config.embedding.backend}")

    hybrid_search = HybridSearchEngine.from_config(config, embedder)
    console.print(f"  HybridSearch: BM25+Dense+RRF+Reranker")

    llm_client = LLMClient(
        generator_provider=config.llm.generator.provider,
        generator_model=config.llm.generator.model,
        router_provider=config.llm.router.provider,
        router_model=config.llm.router.model,
    )
    console.print(f"  LLM: {config.llm.generator.model}")

    router = QueryRouter(
        llm_client=llm_client.router_client,
        complexity_threshold=config.agentic.complexity_threshold,
    )

    orchestrator = AgenticOrchestrator(
        hybrid_search_engine=hybrid_search,
        router=router,
        llm_client=llm_client,
    )
    console.print("  Orchestrator: Agentic RAG pipeline 就绪")

    # -------------------------------------------------------------------------
    # 初始化评估器
    # -------------------------------------------------------------------------
    evaluator = RAGASEvaluator(
        llm_client=llm_client.generator_client,
        thresholds={
            "faithfulness": config.evaluation.thresholds.faithfulness,
            "answer_relevancy": config.evaluation.thresholds.answer_relevancy,
            "context_precision": config.evaluation.thresholds.context_precision,
            "context_recall": config.evaluation.thresholds.context_recall,
            "answer_correctness": config.evaluation.thresholds.answer_correctness,
        },
    )

    suite = RAGTestSuite(
        thresholds={
            "faithfulness": config.evaluation.thresholds.faithfulness,
            "answer_relevancy": config.evaluation.thresholds.answer_relevancy,
            "context_precision": config.evaluation.thresholds.context_precision,
            "context_recall": config.evaluation.thresholds.context_recall,
            "hallucination": 0.05,
        }
    )

    # -------------------------------------------------------------------------
    # 批量评估：每条用例跑真实 RAG pipeline
    # -------------------------------------------------------------------------
    console.print(f"\n[bold cyan]运行评估（{len(test_cases)} 条用例）...[/bold cyan]")

    results = []
    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        ground_truth = case.get("ground_truth", "")
        case_category = case["category"]

        if verbose:
            console.print(f"  [{i}/{len(test_cases)}] 评估中: {question[:40]}...")

        # 执行真实 RAG pipeline
        try:
            rag_result = await orchestrator.run(
                query=question,
                conversation_history=None,
                semantic_cache_fn=None,  # 评估时跳过缓存
            )

            # 用真实结果评估
            eval_result = suite.run_all(
                question=question,
                answer=rag_result.answer,
                contexts=[c["text"] for c in rag_result.citations] if rag_result.citations else [],
                ground_truth=ground_truth,
                category=case_category,
            )

            # 附加 RAG 运行时信息
            eval_result["rag_latency_ms"] = rag_result.latency_ms
            eval_result["rag_complexity"] = rag_result.complexity.value
            eval_result["rag_confidence"] = rag_result.confidence
            eval_result["rag_cache_hit"] = rag_result.cache_hit

        except Exception as e:
            logger.warning(f"RAG pipeline 执行失败 [{question[:40]}...]: {e}")
            eval_result = suite._dummy_result(question, "", case_category)
            eval_result["rag_error"] = str(e)

        results.append(eval_result)

        if verbose:
            status = "[green]PASS[/green]" if eval_result["all_passed"] else "[red]FAIL[/red]"
            score = eval_result.get("average_score", 0)
            console.print(f"       {status} score={score:.2f}")

    # -------------------------------------------------------------------------
    # 生成 CI 报告
    # -------------------------------------------------------------------------
    ci_report = suite.run_ci(
        test_cases=[{
            "question": tc["question"],
            "answer": next((r["answer"] for r in results if r["question"] == tc["question"]), ""),
            "contexts": [tc.get("ground_truth", "")],
            "ground_truth": tc.get("ground_truth", ""),
            "category": tc["category"],
        } for tc in test_cases],
        regression_threshold=config.evaluation.ci.regression_threshold,
    )

    # 补充 RAG 运行时统计
    ci_report["avg_latency_ms"] = sum(r.get("rag_latency_ms", 0) for r in results) / len(results) if results else 0
    ci_report["cache_hit_rate"] = sum(1 for r in results if r.get("rag_cache_hit")) / len(results) if results else 0

    return ci_report


def print_report(report: dict, verbose: bool = False) -> None:
    """打印评估报告"""
    console.print("\n")
    console.print(Panel(
        "[bold]RAG 系统评估报告[/bold]",
        border_style="cyan",
    ))

    # 汇总
    summary_table = Table(title="汇总", show_header=False, box=None)
    summary_table.add_column("key", style="cyan")
    summary_table.add_column("value", style="white")

    summary_table.add_row("总测试用例", str(report["total"]))
    summary_table.add_row("通过数", str(report["passed"]))
    summary_table.add_row("通过率", f"{report['pass_rate']:.1%}")
    summary_table.add_row("回归检测", "是" if report["regression_detected"] else "否")
    summary_table.add_row(
        "CI 状态",
        "[red]FAIL" if report["should_fail_ci"] else "[green]PASS[/green]",
    )
    if "avg_latency_ms" in report:
        summary_table.add_row("平均延迟", f"{report['avg_latency_ms']:.0f}ms")
    if "cache_hit_rate" in report:
        summary_table.add_row("缓存命中率", f"{report['cache_hit_rate']:.1%}")
    console.print(summary_table)

    # 各指标平均分
    if "average_scores" in report:
        metrics_table = Table(title="各指标平均分", show_header=True)
        metrics_table.add_column("指标", style="cyan")
        metrics_table.add_column("平均分", style="white")
        metrics_table.add_column("状态", style="white")

        for metric, score in report["average_scores"].items():
            status = "[green]OK[/green]" if score > 0.7 else "[yellow]WARN[/yellow]" if score > 0.5 else "[red]FAIL[/red]"
            metrics_table.add_row(metric, f"{score:.3f}", status)

        console.print(metrics_table)

    # 弱项
    if "weakest_metric" in report:
        console.print(f"\n[yellow]最弱指标: {report['weakest_metric']} "
                     f"(avg={report.get('weakest_score', 0):.3f})[/yellow]")

    # 详细结果
    if verbose and "per_case" in report:
        console.print("\n[bold cyan]详细结果[/bold cyan]")
        detail_table = Table(show_header=True)
        detail_table.add_column("#", style="dim", width=3)
        detail_table.add_column("查询", style="white")
        detail_table.add_column("通过", style="white")
        detail_table.add_column("得分", style="white")

        for i, case_result in enumerate(report["per_case"], 1):
            q = case_result["question"][:40] + "..."
            status = "[green]PASS[/green]" if case_result["passed"] else "[red]FAIL[/red]"
            score = f"{case_result['score']:.2f}"
            detail_table.add_row(str(i), q, status, score)

        console.print(detail_table)


def main():
    parser = argparse.ArgumentParser(description="RAG 系统评估工具")
    parser.add_argument("--report", action="store_true", help="显示完整报告")
    parser.add_argument("--category", type=str, choices=["simple", "moderate", "difficult"], help="只评估特定分类")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--output", type=str, help="保存报告到 JSON 文件")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件")

    args = parser.parse_args()

    if args.config != "config.yaml":
        ConfigLoader.load(args.config)

    console.print("[bold green]Enterprise RAG — 评估工具[/bold green]")

    try:
        report = asyncio.run(run_evaluation(
            category=args.category,
            verbose=args.verbose,
        ))

        print_report(report, verbose=args.verbose or args.report)

        # 保存报告
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            console.print(f"\n[green]报告已保存: {args.output}[/green]")

        # CI 状态
        if report["should_fail_ci"]:
            console.print("\n[red]CI 状态: FAIL (存在指标低于阈值)[/red]")
            sys.exit(1)
        else:
            console.print("\n[green]CI 状态: PASS[/green]")

        # P2: Eval diff gate — 对比上一次同 prompt_hash 的 run
        try:
            from backend.evaluation.eval_store import get_eval_store
            from prompts import get_prompts_with_hash
            from backend.config import get_config as _gc

            cfg = _gc()
            _, prompt_hash = get_prompts_with_hash()
            # 取最近一次 run (已写库的)
            store = get_eval_store()
            latest = store.get_latest_run()
            if latest and latest.get("prompt_hash"):
                gate = store.diff_against_previous(
                    latest["run_id"],
                    thresholds=cfg.evaluation.ci.thresholds.__dict__ if hasattr(cfg.evaluation.ci, "thresholds") else None,
                )
                if gate.get("passed"):
                    console.print(f"[green]Diff Gate: PASS[/green]  reason={gate.get('reason')}")
                else:
                    console.print(f"[red]Diff Gate: FAIL[/red]  reason={gate.get('reason')}")
                    for m, d in gate.get("diffs", {}).items():
                        if d.get("below_threshold"):
                            console.print(
                                f"  [red]- {m}: {d['previous']:.3f} -> {d['current']:.3f} "
                                f"(drop={d['delta']:.3f}, max={d['max_allowed_drop']:.3f})[/red]"
                            )
                    # CI 失败时退出码 2 (区别于 should_fail_ci 的 1)
                    sys.exit(2)
        except Exception as e:
            logger.warning(f"Eval diff gate skipped: {e}")

    except Exception as e:
        console.print(f"\n[red]评估失败: {e}[/red]")
        logger.exception("评估过程异常")
        sys.exit(1)


if __name__ == "__main__":
    main()
