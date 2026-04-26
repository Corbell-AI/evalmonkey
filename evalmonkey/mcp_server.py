from mcp.server.fastmcp import FastMCP
import asyncio
from typing import Optional

from evalmonkey.simulator.load_gen import LoadGenerator
from evalmonkey.evals.runner import LLMJudgeProvider
from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
from evalmonkey.evals.local_assets import load_local_evals
from evalmonkey.reporting.history import record_run, get_history

# Initialize the FastMCP server
mcp = FastMCP("EvalMonkey")

@mcp.tool()
async def run_benchmark(scenario: str, target_url: str, request_key: str = "question", response_path: str = "data") -> str:
    """
    Run an EvalMonkey benchmark against an HTTP agent.
    Returns a textual summary of the score and reasoning.
    """
    evals_to_run = load_standard_benchmark(scenario, limit=1)
    if not evals_to_run:
        return f"Scenario {scenario} not found."
    
    generator = LoadGenerator(target_url, request_key=request_key, response_path=response_path)
    judge = LLMJudgeProvider()
    
    resp = await generator.run_scenario(scenario, evals_to_run[0].input_payload)
    if resp["status"] == "error":
        return f"Benchmark failed to execute. Error: {resp.get('error_message')}"
        
    agent_output_text = str(resp.get("data", "No output returned"))
    evaluation = judge.score_run(evals_to_run[0].expected_behavior_rubric, agent_output_text)
    score = evaluation.get("score", 0)
    reasoning = evaluation.get("reasoning", "No reasoning provided.")
    
    record_run(scenario, "baseline", score, details={"reasoning": reasoning, "sample_size": 1})
    return f"Benchmark Complete. Score: {score}/100. Reasoning: {reasoning}"


@mcp.tool()
async def run_chaos(scenario: str, target_url: str, chaos_profile: str, request_key: str = "question", response_path: str = "data") -> str:
    """
    Run a benchmark with an injected Chaos Profile (e.g. client_payload_bloat, client_empty_payload, server_latency_spike).
    """
    evals_to_run = load_standard_benchmark(scenario, limit=1)
    if not evals_to_run:
        return f"Scenario {scenario} not found."
        
    generator = LoadGenerator(target_url, request_key=request_key, response_path=response_path)
    judge = LLMJudgeProvider()
    
    resp = await generator.run_scenario(scenario, evals_to_run[0].input_payload, chaos_profile=chaos_profile)
    agent_output_text = str(resp.get("data", resp.get("error_message", "No output")))
    
    evaluation = judge.score_run(evals_to_run[0].expected_behavior_rubric, agent_output_text)
    score = evaluation.get("score", 0)
    reasoning = evaluation.get("reasoning", "No reasoning provided.")
    
    record_run(scenario, "chaos", score, details={"chaos_profile": chaos_profile, "sample_size": 1})
    return f"Chaos Simulation Complete. Scenario: {scenario}, Chaos Profile: {chaos_profile}. Score: {score}/100. Reasoning: {reasoning}"


@mcp.tool()
def get_benchmark_history(scenario: str) -> str:
    """Fetch the chronological evaluation history for a given scenario to trace improvement over time."""
    hist = get_history(scenario)
    if not hist:
        return f"No history found for scenario: {scenario}"
    
    lines = []
    for h in hist:
        lines.append(f"[{h['timestamp']}] Type: {h['run_type']} | Score: {h['score']}/100")
    
    return "\n".join(lines)


@mcp.tool()
async def generate_improvement_evals(
    scenario: str,
    target_url: str,
    output_dir: Optional[str] = None,
    limit: int = 3,
    request_key: str = "question",
    response_path: str = "data",
    langfuse_dataset: Optional[str] = None,
) -> str:
    """
    Run a benchmark, capture failing traces, and generate LLM-synthesised
    improvement eval scenarios saved to output/.

    Use this when the agent score is low and you want targeted test cases
    to hand to a coding agent (Claude Code / Cursor) to fix the agent.

    Args:
        scenario:         Standard benchmark ID (e.g. 'gsm8k', 'mmlu', 'arc').
        target_url:       Full URL of the agent's HTTP endpoint.
        output_dir:       Optional override for where to save assets.
        limit:            How many benchmark samples to evaluate.
        request_key:      JSON body key for the question (default 'question').
        response_path:    Dot-path to extract the agent's answer (default 'data').
        langfuse_dataset: If provided and LANGFUSE_* keys are set, push evals to Langfuse.
    """
    from evalmonkey.evals.asset_generator import EvalAssetGenerator, FailingTrace, build_output_dir

    evals_to_run = load_standard_benchmark(scenario, limit=limit)
    if not evals_to_run:
        return f"Scenario '{scenario}' not found. Run `evalmonkey list-benchmarks` for valid IDs."

    generator = LoadGenerator(target_url, request_key=request_key, response_path=response_path)
    judge = LLMJudgeProvider()
    asset_gen = EvalAssetGenerator()

    for eval_task in evals_to_run:
        resp = await generator.run_scenario(scenario, eval_task.input_payload)
        agent_output_text = str(resp.get("data", resp.get("error_message", "No output")))
        evaluation = judge.score_run(eval_task.expected_behavior_rubric, agent_output_text)
        score = evaluation.get("score", 0)
        reasoning = evaluation.get("reasoning", "")
        asset_gen.record_failure(FailingTrace(
            scenario=scenario,
            eval_id=eval_task.id,
            input_payload=eval_task.input_payload,
            agent_output=agent_output_text,
            expected_rubric=eval_task.expected_behavior_rubric,
            score=score,
            reasoning=reasoning,
        ))

    if not asset_gen.has_failures:
        return (
            f"✅ All {limit} samples scored above the failure threshold — no improvement evals needed! "
            f"Your agent is performing well on '{scenario}'."
        )

    out_dir = output_dir or build_output_dir(scenario)
    saved_path = asset_gen.save(out_dir)

    langfuse_msg = ""
    if langfuse_dataset:
        ok = asset_gen.export_to_langfuse(langfuse_dataset)
        langfuse_msg = (
            f"\n✅ Evals pushed to Langfuse dataset: '{langfuse_dataset}'"
            if ok
            else "\n⚠️ Langfuse export skipped — set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable."
        )

    return (
        f"⚠️ {asset_gen.failure_count} of {limit} sample(s) failed (score < threshold).\n"
        f"Eval assets saved to: {saved_path}\n\n"
        f"Files written:\n"
        f"  • traces.json           — raw failing traces\n"
        f"  • evals.json            — new targeted test cases\n"
        f"  • improvement_prompt.md — copy-pasteable brief for Claude Code / Cursor\n\n"
        f"Next step — paste this into Claude Code or Cursor:\n"
        f"  cat {saved_path}/improvement_prompt.md\n\n"
        f"Then re-run to verify the fix:\n"
        f"  evalmonkey run-benchmark --scenario {scenario} --target-url {target_url}"
        f"{langfuse_msg}"
    )


@mcp.tool()
def get_eval_assets(output_dir: str) -> str:
    """
    Read and return the saved eval assets (traces.json + evals.json +
    improvement_prompt.md) from a previous benchmark run's output directory.

    Use this to inspect what went wrong and what improvement evals were
    generated, directly inside Claude Code or Cursor.
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(output_dir)
    if not path.exists():
        return f"Output directory not found: {output_dir}"

    result_parts = []

    traces_file = path / "traces.json"
    if traces_file.exists():
        traces = _json.loads(traces_file.read_text())
        result_parts.append(f"=== FAILING TRACES ({len(traces)}) ===")
        for t in traces:
            result_parts.append(
                f"\n[{t.get('eval_id', '?')}] Score={t.get('score', '?')}/100 | "
                f"Chaos={t.get('chaos_profile') or 'none'}\n"
                f"  Q: {str(t.get('input_payload', {}).get('question', ''))[:200]}\n"
                f"  A: {t.get('agent_output', '')[:200]}\n"
                f"  Reason: {t.get('reasoning', '')[:200]}"
            )

    evals_file = path / "evals.json"
    if evals_file.exists():
        evals = _json.loads(evals_file.read_text())
        result_parts.append(f"\n\n=== GENERATED IMPROVEMENT EVALS ({len(evals)}) ===")
        for ev in evals:
            result_parts.append(
                f"\n[{ev.get('id', '?')}] {ev.get('description', '')}\n"
                f"  Q: {ev.get('input_payload', {}).get('question', '')}\n"
                f"  Expected: {ev.get('expected_behavior_rubric', '')}"
            )

    prompt_file = path / "improvement_prompt.md"
    if prompt_file.exists():
        result_parts.append(f"\n\n=== IMPROVEMENT PROMPT ===\n{prompt_file.read_text()}")

    return "\n".join(result_parts) if result_parts else f"No eval asset files found in {output_dir}"


@mcp.tool()
async def run_full_pipeline(
    scenario: str,
    target_url: str,
    chaos_profiles: Optional[str] = "client_prompt_injection,client_payload_bloat,client_empty_payload",
    limit: int = 3,
    request_key: str = "question",
    response_path: str = "data",
    langfuse_dataset: Optional[str] = None,
) -> str:
    """
    Run the COMPLETE EvalMonkey pipeline in a single call:
      1. Baseline benchmark
      2. All specified chaos tests
      3. Generate improvement eval assets from any failures
      4. Optionally push evals to Langfuse

    Perfect for automating the full agent quality loop from Claude Code or Cursor.

    Args:
        scenario:         Standard benchmark ID (e.g. 'gsm8k', 'arc', 'mmlu').
        target_url:       Full URL of the agent's HTTP endpoint.
        chaos_profiles:   Comma-separated chaos profile names to inject.
        limit:            Number of benchmark samples per run.
        request_key:      JSON body key for the question.
        response_path:    Dot-path to extract the agent's answer.
        langfuse_dataset: Push generated evals to this Langfuse dataset (optional).
    """
    from evalmonkey.evals.asset_generator import EvalAssetGenerator, FailingTrace, build_output_dir

    profiles = [p.strip() for p in (chaos_profiles or "").split(",") if p.strip()]
    evals_to_run = load_standard_benchmark(scenario, limit=limit)
    if not evals_to_run:
        return f"Scenario '{scenario}' not found."

    generator = LoadGenerator(target_url, request_key=request_key, response_path=response_path)
    judge = LLMJudgeProvider()
    asset_gen = EvalAssetGenerator()
    summary_lines = [f"🐵 EvalMonkey Full Pipeline — Scenario: {scenario}\n"]

    # ── 1. Baseline Run ──────────────────────────────────────────────────────
    baseline_scores = []
    for ev in evals_to_run:
        resp = await generator.run_scenario(scenario, ev.input_payload)
        out = str(resp.get("data", resp.get("error_message", "")))
        result = judge.score_run(ev.expected_behavior_rubric, out)
        score = result.get("score", 0)
        baseline_scores.append(score)
        asset_gen.record_failure(FailingTrace(
            scenario=scenario, eval_id=ev.id, input_payload=ev.input_payload,
            agent_output=out, expected_rubric=ev.expected_behavior_rubric,
            score=score, reasoning=result.get("reasoning", ""),
        ))

    baseline_avg = int(sum(baseline_scores) / len(baseline_scores)) if baseline_scores else 0
    record_run(scenario, "baseline", baseline_avg, details={"sample_size": len(baseline_scores)})
    summary_lines.append(f"📊 Baseline Score:  {baseline_avg}/100")

    # ── 2. Chaos Runs ────────────────────────────────────────────────────────
    for profile in profiles:
        chaos_scores = []
        for ev in evals_to_run:
            resp = await generator.run_scenario(scenario, ev.input_payload, chaos_profile=profile)
            out = str(resp.get("data", resp.get("error_message", "")))
            result = judge.score_run(ev.expected_behavior_rubric, out)
            score = result.get("score", 0)
            chaos_scores.append(score)
            asset_gen.record_failure(FailingTrace(
                scenario=scenario, eval_id=ev.id, input_payload=ev.input_payload,
                agent_output=out, expected_rubric=ev.expected_behavior_rubric,
                score=score, reasoning=result.get("reasoning", ""),
                chaos_profile=profile,
            ))
        chaos_avg = int(sum(chaos_scores) / len(chaos_scores)) if chaos_scores else 0
        record_run(scenario, "chaos", chaos_avg, details={"chaos_profile": profile, "sample_size": len(chaos_scores)})
        summary_lines.append(f"🔥 Chaos [{profile}]: {chaos_avg}/100")

    # ── 3. Eval Asset Generation ─────────────────────────────────────────────
    if asset_gen.has_failures:
        out_dir = build_output_dir(scenario)
        saved = asset_gen.save(out_dir)

        langfuse_msg = ""
        if langfuse_dataset:
            ok = asset_gen.export_to_langfuse(langfuse_dataset)
            langfuse_msg = (
                f"\n✅ Langfuse dataset '{langfuse_dataset}' updated."
                if ok else "\n⚠️ Langfuse export skipped — credentials not set."
            )

        summary_lines.append(
            f"\n⚠️  {asset_gen.failure_count} failing trace(s) captured.\n"
            f"📁 Eval assets saved to: {saved}\n"
            f"   • traces.json, evals.json, improvement_prompt.md\n"
            f"\n💡 To fix your agent, run:\n"
            f"   cat {saved}/improvement_prompt.md\n"
            f"   Then paste into Claude Code or Cursor."
            f"{langfuse_msg}"
        )
    else:
        summary_lines.append("\n✅ No failures detected — agent is healthy!")

    return "\n".join(summary_lines)
