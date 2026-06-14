"""
eval.py — Evaluation dashboard API (P2.3)

技术决策:
- POST /api/eval/run 真实调用 RAGASEvaluator 对 golden dataset 跑评估，
  写入 SQLite；之前是写死假数据的 stub，已在 P0 阶段替换。
- 真实跑评估是 LLM 密集操作（每个 query 调一次 LLM-as-judge），
  所以走 BackgroundTasks，并提供 /api/eval/run/{run_id} 轮询状态。
- 仅对 simple 类别（10 条）跑在线触发，moderate/difficult 通过
  scripts/eval.py 离线跑（避免 API 超时）。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from backend.evaluation.eval_store import get_eval_store
from backend.security.auth import require_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"])


# --------------------------------------------------------------------------
# In-memory run status (jobs in flight)
# --------------------------------------------------------------------------
# 跑完即落 SQLite，这张表只存 in-flight 状态。重启后清空。
_RUN_STATUS: dict[str, dict] = {}
_RUN_STATUS_LOCK = asyncio.Lock()


@router.get("/summary")
async def get_summary(
    window: str = Query(default="24h", description="1h|24h|7d|30d"),
    token_payload: dict = Depends(require_current_user),
) -> dict[str, Any]:
    """获取最新一次 eval run 的汇总指标"""
    store = get_eval_store()
    latest = store.get_latest_run()
    if not latest:
        return {
            "available": False,
            "message": "no eval runs yet; click 'Run Now' to trigger",
        }
    return {
        "available": True,
        "run_id": latest["run_id"],
        "started_at": latest["started_at"],
        "ended_at": latest["ended_at"],
        "total_cases": latest["total_cases"],
        "passed_cases": latest["passed_cases"],
        "pass_rate": (latest["passed_cases"] / latest["total_cases"]) if latest["total_cases"] else 0,
        "avg_faithfulness": latest["avg_faithfulness"],
        "avg_answer_relevancy": latest["avg_answer_relevancy"],
        "avg_context_precision": latest["avg_context_precision"],
        "avg_context_recall": latest["avg_context_recall"],
        "avg_answer_correctness": latest["avg_answer_correctness"],
        "weakest_metric": latest["weakest_metric"],
    }


@router.get("/runs")
async def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    token_payload: dict = Depends(require_current_user),
) -> dict[str, Any]:
    """列出所有 eval run 摘要（用于趋势图）"""
    store = get_eval_store()
    runs = store.list_runs(limit=limit)
    return {"runs": runs, "total": len(runs)}


@router.get("/runs/{run_id}/samples")
async def get_run_samples(
    run_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    token_payload: dict = Depends(require_current_user),
) -> dict[str, Any]:
    """获取指定 run 的样本详情"""
    store = get_eval_store()
    samples = store.get_samples_for_run(run_id, limit=limit)
    return {"samples": samples, "total": len(samples)}


@router.post("/run")
async def trigger_run(
    background_tasks: BackgroundTasks,
    category: str = Query(default="simple", description="simple|moderate|difficult"),
    token_payload: dict = Depends(require_current_user),
) -> dict[str, Any]:
    """
    触发一次 eval run：真跑 RAGASEvaluator 对 golden dataset 评估。

    实际执行流（替换之前的 stub）:
    1. 从 test_dataset.py 加载指定 category 的测试用例
    2. 对每条用例：跑真实 RAG pipeline（orchestrator.run）拿 answer
    3. 用 RAGASEvaluator 评估 answer vs ground_truth
    4. 写 SQLite (eval_runs + eval_samples)
    """
    run_id = f"manual_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    async with _RUN_STATUS_LOCK:
        _RUN_STATUS[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "category": category,
            "started_at": datetime.utcnow().isoformat(),
            "total_cases": 0,
            "completed_cases": 0,
        }

    async def _run_evaluation():
        async with _RUN_STATUS_LOCK:
            _RUN_STATUS[run_id]["status"] = "running"

        try:
            # 1. 加载测试数据
            from backend.evaluation.test_dataset import get_test_dataset_by_category
            test_cases = get_test_dataset_by_category(category)

            async with _RUN_STATUS_LOCK:
                _RUN_STATUS[run_id]["total_cases"] = len(test_cases)

            # 2. 真跑 RAG pipeline
            from backend.api.deps import get_orchestrator
            orchestrator = get_orchestrator()

            # 3. 评估器
            from backend.config import get_config
            from backend.evaluation.ragas_metrics import RAGASEvaluator
            cfg = get_config()
            evaluator = RAGASEvaluator(
                llm_client=None,  # 用 ragas 自带 evaluator LLM
                thresholds={
                    "faithfulness": cfg.evaluation.thresholds.faithfulness,
                    "answer_relevancy": cfg.evaluation.thresholds.answer_relevancy,
                    "context_precision": cfg.evaluation.thresholds.context_precision,
                    "context_recall": cfg.evaluation.thresholds.context_recall,
                    "answer_correctness": cfg.evaluation.thresholds.answer_correctness,
                },
            )

            samples = []
            agg = {
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
                except Exception as e:
                    logger.warning(f"RAG pipeline failed for '{case['question'][:40]}': {e}")
                    answer = ""
                    contexts = []

                report = await evaluator.evaluate(
                    question=case["question"],
                    answer=answer,
                    retrieved_contexts=contexts,
                    ground_truth=case.get("ground_truth"),
                )

                # 累计
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
                    "faithfulness": _metric(agg, "faithfulness", -1, 0.0),
                    "answer_relevancy": _metric(agg, "answer_relevancy", -1, 0.0),
                    "context_precision": _metric(agg, "context_precision", -1, 0.0),
                    "context_recall": _metric(agg, "context_recall", -1, 0.0),
                    "answer_correctness": _metric(agg, "answer_correctness", -1, 0.0),
                    "overall_pass": overall_pass,
                    "latency_ms": getattr(rag_result, "latency_ms", 0.0) if 'rag_result' in dir() else 0.0,
                })

                async with _RUN_STATUS_LOCK:
                    _RUN_STATUS[run_id]["completed_cases"] = len(samples)

            # 4. 写 SQLite
            def _avg(xs):
                return sum(xs) / len(xs) if xs else 0.0

            weakest_name, weakest_score = "", 0.0
            if any(agg.values()):
                weakest_name = min(
                    (k for k in agg if agg[k]),
                    key=lambda k: _avg(agg[k]),
                    default="",
                )
                weakest_score = _avg(agg.get(weakest_name, []))

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

            async with _RUN_STATUS_LOCK:
                _RUN_STATUS[run_id]["status"] = "completed"
                _RUN_STATUS[run_id]["ended_at"] = ended
                _RUN_STATUS[run_id]["weakest_metric"] = weakest_name
            logger.info(
                f"Eval run {run_id} completed: {passed}/{len(test_cases)} passed, "
                f"weakest={weakest_name}"
            )
        except Exception as e:
            logger.exception(f"Eval run {run_id} failed: {e}")
            async with _RUN_STATUS_LOCK:
                _RUN_STATUS[run_id]["status"] = "failed"
                _RUN_STATUS[run_id]["error"] = str(e)[:500]

    background_tasks.add_task(_run_evaluation)
    return {
        "run_id": run_id,
        "status": "queued",
        "category": category,
        "poll_url": f"/api/eval/run/{run_id}",
    }


@router.get("/run/{run_id}")
async def get_run_status(
    run_id: str,
    token_payload: dict = Depends(require_current_user),
) -> dict[str, Any]:
    """查询 in-flight 评估 run 的状态"""
    async with _RUN_STATUS_LOCK:
        status = _RUN_STATUS.get(run_id)

    if status:
        return status

    # in-memory 不在则从 SQLite 读
    store = get_eval_store()
    record = store.get_run_by_id(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": run_id,
        "status": "completed",
        "started_at": record["started_at"],
        "ended_at": record["ended_at"],
        "weakest_metric": record["weakest_metric"],
    }


def _metric(agg: dict, key: str, idx: int, default: float) -> float:
    """helper: 取最近一次累计的指标值"""
    if not agg.get(key):
        return default
    return agg[key][idx] if -len(agg[key]) <= idx < len(agg[key]) else default
