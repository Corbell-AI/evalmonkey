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
    
    # Save history
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
