"""
eval.py — Evaluation API (Phase2-2.2: Dashboard 后端 API 已删除)

Phase2-2.2: 删除 /summary /runs /runs/{id}/samples /run/{id} 4 个端点，
仅保留 POST /api/eval/run (CI 触发用，状态查 SQLite / 离线 grep 日志)。
Online Evaluator + Dashboard 整体下线，前端 /eval 页面也已删除。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends

from backend.evaluation.eval_store import get_eval_store
from backend.security.auth import require_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run")
async def trigger_run(
    background_tasks: BackgroundTasks,
    category: str = "simple",
    token_payload: dict = Depends(require_current_user),
) -> dict:
    """
    触发一次 eval run：真跑 RAGASEvaluator 对 golden dataset 评估。

    Phase2-2.2: dashboard 端点删除后，in-memory _RUN_STATUS 同步移除。
    Run 状态查询直接走 offline path：scripts/eval.py 输出 + eval_runs 表。
    """
    run_id = f"manual_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    async def _run_evaluation():
        try:
            from backend.api.deps import get_orchestrator
            from backend.config import get_config
            from backend.evaluation.ragas_metrics import RAGASEvaluator
            from backend.evaluation.test_dataset import get_test_dataset_by_category

            test_cases = get_test_dataset_by_category(category)
            cfg = get_config()
            evaluator = RAGASEvaluator(
                llm_client=None,
                thresholds={
                    "faithfulness": cfg.evaluation.thresholds.faithfulness,
                    "answer_relevancy": cfg.evaluation.thresholds.answer_relevancy,
                    "context_precision": cfg.evaluation.thresholds.context_precision,
                    "context_recall": cfg.evaluation.thresholds.context_recall,
                    "answer_correctness": cfg.evaluation.thresholds.answer_correctness,
                },
            )
            orchestrator = get_orchestrator()

            samples = []
            agg: dict[str, list[float]] = {
                "faithfulness": [], "answer_relevancy": [],
                "context_precision": [], "context_recall": [],
                "answer_correctness": [],
            }
            passed = 0

            for case in test_cases:
                try:
                    rag_result = await orchestrator.run(
                        query=case["question"],
                        conversation_history=None,
                    )
                    answer = rag_result.answer
                    contexts = [c.get("text", "") for c in rag_result.citations] if rag_result.citations else []
                    latency_ms = getattr(rag_result, "latency_ms", 0.0)
                except Exception as e:
                    logger.warning(f"RAG pipeline failed for '{case['question'][:40]}': {e}")
                    answer = ""
                    contexts = []
                    latency_ms = 0.0

                report = await evaluator.evaluate(
                    question=case["question"],
                    answer=answer,
                    retrieved_contexts=contexts,
                    ground_truth=case.get("ground_truth"),
                )

                metrics_by_name = {r.metric: r.score for r in report.results}
                for r in report.results:
                    if r.metric in agg:
                        agg[r.metric].append(r.score)

                overall_pass = report.overall_pass
                if overall_pass:
                    passed += 1

                samples.append({
                    "sample_id": f"s_{len(samples)}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "query": case["question"],
                    "answer": answer[:1000],
                    "faithfulness": metrics_by_name.get("faithfulness", 0.0),
                    "answer_relevancy": metrics_by_name.get("answer_relevancy", 0.0),
                    "context_precision": metrics_by_name.get("context_precision", 0.0),
                    "context_recall": metrics_by_name.get("context_recall", 0.0),
                    "answer_correctness": metrics_by_name.get("answer_correctness", 0.0),
                    "overall_pass": overall_pass,
                    "latency_ms": latency_ms,
                })

            def _avg(xs):
                return sum(xs) / len(xs) if xs else 0.0

            weakest_name = ""
            if any(agg.values()):
                weakest_name = min(
                    (k for k in agg if agg[k]),
                    key=lambda k: _avg(agg[k]),
                    default="",
                )

            store = get_eval_store()
            started = datetime.utcnow().isoformat()
            ended = datetime.utcnow().isoformat()
            store.save_run(
                run_id=run_id,
                started_at=started,
                ended_at=ended,
                total_cases=len(test_cases),
                passed_cases=passed,
                avg_faithfulness=_avg(agg["faithfulness"]),
                avg_answer_relevancy=_avg(agg["answer_relevancy"]),
                avg_context_precision=_avg(agg["context_precision"]),
                avg_context_recall=_avg(agg["context_recall"]),
                avg_answer_correctness=_avg(agg["answer_correctness"]),
                weakest_metric=weakest_name,
                metadata={"source": "manual_trigger", "category": category},
            )
            store.save_samples(run_id, samples)
            logger.info(
                f"Eval run {run_id} completed: {passed}/{len(test_cases)} passed, "
                f"weakest={weakest_name}"
            )
        except Exception as e:
            logger.exception(f"Eval run {run_id} failed: {e}")

    background_tasks.add_task(_run_evaluation)
    return {
        "run_id": run_id,
        "status": "queued",
        "category": category,
    }
