"""EvalMonkey UI — FastAPI backend.

Run with:
    cd <path-to-evalmonkey>
    uvicorn ui.backend.main:app --reload --port 8080
"""
from __future__ import annotations

# Load .env automatically — EVAL_MODEL and all LLM API keys must be
# available before any evalmonkey modules are imported.
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; user must export vars manually


import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from evalmonkey.reporting.history import calculate_production_reliability, get_history
from evalmonkey.scenarios.standard_benchmarks import SUPPORTED_BENCHMARKS

from .db import get_backend
from .run_engine import execute_run, get_queue
from .schemas import (
    BenchmarkInfo,
    RunRecord,
    RunSummary,
    StartBenchmarkRequest,
    StartChaosRequest,
)

app = FastAPI(title="EvalMonkey UI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Config (exposes env-based defaults to the frontend) ──────────────────────

@app.get("/api/config")
def get_config():
    """Return environment-detected defaults so the UI can pre-select the right judge model."""
    return {
        "default_eval_model": os.getenv("EVAL_MODEL", "gpt-4o"),
        "has_bedrock_key":    bool(os.getenv("BEDROCK_API_KEY")),
        "has_openai_key":     bool(os.getenv("OPENAI_API_KEY")),
        "has_anthropic_key":  bool(os.getenv("ANTHROPIC_API_KEY")),
    }


# ── Benchmarks ────────────────────────────────────────────────────────────────

@app.get("/api/benchmarks", response_model=List[BenchmarkInfo])
def list_benchmarks(category: Optional[str] = None):
    """List supported benchmarks, optionally filtered by agent category."""
    items = [
        BenchmarkInfo(id=k, description=v["description"], category=v["agent_category"])
        for k, v in SUPPORTED_BENCHMARKS.items()
    ]
    if category:
        items = [b for b in items if b.category.lower() == category.lower()]
    return items


# ── Runs ──────────────────────────────────────────────────────────────────────

@app.get("/api/runs", response_model=List[RunSummary])
def list_runs(limit: int = 50):
    runs = get_backend().get_all_runs(limit=limit)
    return [RunSummary(**r.model_dump()) for r in runs]


@app.get("/api/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: str):
    run = get_backend().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunSummary(**run.model_dump())


@app.get("/api/runs/{run_id}/samples")
def get_run_samples(run_id: str):
    samples = get_backend().get_samples(run_id)
    return [s.model_dump() for s in samples]


# ── Start runs ────────────────────────────────────────────────────────────────

@app.post("/api/run/benchmark")
async def start_benchmark(req: StartBenchmarkRequest, background_tasks: BackgroundTasks):
    run = RunRecord(
        scenario=req.scenario,
        run_type="baseline",
        target_url=req.target_url if not req.use_sample_agent else "http://127.0.0.1:8001/solve",
        eval_model=req.eval_model,
        request_key=req.request_key,
        response_path=req.response_path,
        sample_count=req.limit,
    )
    get_backend().save_run(run)
    # Initialize queue before background task
    get_queue(run.id)
    background_tasks.add_task(_run_benchmark_task, run.id, req)
    return {"run_id": run.id, "status": "started"}


@app.post("/api/run/chaos")
async def start_chaos(req: StartChaosRequest, background_tasks: BackgroundTasks):
    run = RunRecord(
        scenario=req.scenario,
        run_type="chaos",
        target_url=req.target_url if not req.use_sample_agent else "http://127.0.0.1:8001/solve",
        eval_model=req.eval_model,
        request_key=req.request_key,
        response_path=req.response_path,
        chaos_profile=req.chaos_profile,
        sample_count=req.limit,
    )
    get_backend().save_run(run)
    get_queue(run.id)
    background_tasks.add_task(_run_chaos_task, run.id, req)
    return {"run_id": run.id, "status": "started"}


async def _run_benchmark_task(run_id: str, req: StartBenchmarkRequest):
    await execute_run(run_id, req, chaos_profile=None)


async def _run_chaos_task(run_id: str, req: StartChaosRequest):
    await execute_run(run_id, req, chaos_profile=req.chaos_profile)


# ── SSE Stream ────────────────────────────────────────────────────────────────

@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str):
    run = get_backend().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return StreamingResponse(
        _event_generator(run_id, run.status),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


async def _event_generator(run_id: str, initial_status: str) -> AsyncGenerator[str, None]:
    # Flush already-stored samples immediately (handles page reload / late connect)
    existing = get_backend().get_samples(run_id)
    scores_so_far = []
    for s in existing:
        scores_so_far.append(s.score or 0)
        current_avg = int(sum(scores_so_far) / len(scores_so_far))
        event = {
            "type": "sample",
            "index": s.sample_index,
            "eval_id": s.eval_id,
            "question": s.question,
            "agent_output": s.agent_output,
            "expected_rubric": s.expected_rubric,
            "score": s.score,
            "reasoning": s.reasoning,
            "current_score": current_avg,
        }
        yield f"data: {json.dumps(event)}\n\n"

    # If already finished, send complete event and stop
    if initial_status in ("completed", "failed"):
        run = get_backend().get_run(run_id)
        if run and run.status == "completed":
            yield f"data: {json.dumps({'type': 'complete', 'final_score': run.score, 'failure_count': run.details.get('failure_count', 0)})}\n\n"
        elif run and run.status == "failed":
            yield f"data: {json.dumps({'type': 'error', 'message': run.details.get('error', 'Run failed')})}\n\n"
        return

    # Otherwise drain the live queue
    queue = get_queue(run_id)
    already_seen = {s.sample_index for s in existing}

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=60.0)
            # Skip sample events we already sent from DB
            if event.get("type") == "sample" and event.get("index") in already_seen:
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("complete", "error"):
                break
        except asyncio.TimeoutError:
            # Send heartbeat to keep connection alive
            yield ": heartbeat\n\n"


# ── History & Reliability ──────────────────────────────────────────────────────

@app.get("/api/history")
def get_all_history():
    """Get score history for all scenarios from the CLI history file."""
    from evalmonkey.reporting.history import get_history as _get_history
    return _get_history()


@app.get("/api/history/{scenario}")
def get_scenario_history(scenario: str):
    return get_history(scenario)


@app.get("/api/reliability")
def get_reliability():
    """Get production reliability for all scenarios that have history."""
    history = get_history()
    scenarios = {h["scenario"] for h in history}
    result = {}
    for s in scenarios:
        result[s] = {
            "reliability": calculate_production_reliability(s),
            "baseline_count": sum(1 for h in history if h["scenario"] == s and h["run_type"] == "baseline"),
            "chaos_count": sum(1 for h in history if h["scenario"] == s and h["run_type"] == "chaos"),
        }
    return result


@app.get("/api/reliability/{scenario}")
def get_scenario_reliability(scenario: str):
    return {
        "scenario": scenario,
        "reliability": calculate_production_reliability(scenario),
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
