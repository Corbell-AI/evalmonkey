import typer
import asyncio
import subprocess
import time
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import box

from evalmonkey.simulator.load_gen import LoadGenerator
from evalmonkey.evals.local_assets import load_local_evals
from evalmonkey.evals.runner import LLMJudgeProvider
from evalmonkey.reporting.markdown import (
    print_banner, 
    print_benchmark_score, 
    print_chaos_result,
    print_history_trends
)
from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark, get_supported_benchmarks
from evalmonkey.reporting.history import record_run, get_history, calculate_production_reliability

app = typer.Typer(help="EvalMonkey: Open-source Agent Benchmarking and Chaos Framework")
console = Console()

@app.command()
def list_benchmarks():
    """Lists the 10 off-the-shelf benchmark datasets natively supported."""
    print_banner()
    console.print("\n[bold cyan]🐵 EvalMonkey Natively Supported Benchmarks 🐵[/bold cyan]")
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("Scenario ID", style="bold white")
    table.add_column("Description")
    
    benchmarks = get_supported_benchmarks()
    for b_id, desc in benchmarks.items():
        table.add_row(b_id, desc)
        
    console.print(table)
    console.print("\n[dim]Run them via: evalmonkey run-benchmark --scenario <id> --target-url <url>[/dim]\n")


def _spawn_sample_agent(sample_agent: str):
    console.print(f"[bold yellow]=> Spawning sample agent '{sample_agent}' in the background...[/bold yellow]")
    if sample_agent == "rag_app":
        target_url = "http://127.0.0.1:8001/solve"
        proc = subprocess.Popen(["python", "apps/rag_app/app.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        return proc, target_url
    elif sample_agent == "research_agent":
        target_url = "http://127.0.0.1:8002/solve"
        proc = subprocess.Popen(["python", "apps/research_agent/app.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        return proc, target_url
    return None, None

@app.command()
def run_benchmark(
    scenario: str = typer.Option(..., help="Scenario ID, standard benchmark (e.g. gsm8k), or custom_eval ID"),
    target_url: str = typer.Option(None, help="Address of the BYO agent API (e.g. http://localhost:8000). Required unless using --sample-agent."),
    sample_agent: str = typer.Option(None, help="Automatically spawn a sample agent in the background (rag_app or research_agent)"),
    eval_file: str = typer.Option("custom_evals.yaml", help="Path to evaluation assets"),
    limit: int = typer.Option(5, help="Number of samples from the dataset to load and evaluate.")
):
    """
    Run the agent against a standard dataset or custom evaluation and record the score.
    """
    print_banner()
    
    agent_process = None
    if sample_agent:
        agent_process, target_url = _spawn_sample_agent(sample_agent)
        if not agent_process:
            console.print(f"[bold red]Unknown sample agent: {sample_agent}[/bold red]")
            return

    if not target_url:
        console.print("[bold red]❌ --target-url is required if not using --sample-agent[/bold red]")
        return
    
    standard_evals = load_standard_benchmark(scenario, limit=limit)
    
    if standard_evals:
        console.print(f"[bold cyan]=> Loaded {len(standard_evals)} samples from standard benchmark subset: {scenario}[/bold cyan]")
        evals_to_run = standard_evals
    else:
        console.print(f"[bold cyan]=> Loading local BYO eval assets from {eval_file}[/bold cyan]")
        evals = load_local_evals(eval_file)
        target_eval = next((e for e in evals if e.id == scenario), None)
        if not target_eval:
            console.print(f"[bold red]Scenario {scenario} not found in standard sets or {eval_file}[/bold red]")
            if agent_process: agent_process.terminate()
            return
        evals_to_run = [target_eval]

    try:
        generator = LoadGenerator(target_url)
        judge = LLMJudgeProvider()
        scores = []
        overall_reasoning = ""
        
        console.print(f"[bold cyan]=> Executing {len(evals_to_run)} payload(s) sequentially against {target_url}...[/bold cyan]")
        
        for idx, eval_task in enumerate(evals_to_run):
            resp = asyncio.run(generator.run_scenario(scenario, eval_task.input_payload))
            if resp["status"] == "error":
                console.print(f"[bold red]❌ Request failed on sample {idx}: {resp.get('error_message')}[/bold red]")
                continue
                
            agent_output_text = str(resp.get("data", "No output returned"))
            evaluation = judge.score_run(eval_task.expected_behavior_rubric, agent_output_text)
            scores.append(evaluation.get("score", 0))
            if idx == 0:
                overall_reasoning = evaluation.get("reasoning", "No reasoning provided.")
        
        if not scores:
            console.print("[bold red]❌ All requests failed to execute![/bold red]")
            return
            
        final_score = int(sum(scores) / len(scores))
        
        # Check baseline from history
        hist = get_history(scenario)
        baseline = hist[-1]["score"] if hist else None
        
        record_run(scenario, "baseline", final_score, details={"reasoning": overall_reasoning, "sample_size": len(scores)})
        print_benchmark_score(scenario, final_score, overall_reasoning, baseline)
    finally:
        if agent_process:
            agent_process.terminate()


@app.command()
def run_chaos(
    scenario: str = typer.Option(..., help="Scenario ID"),
    target_url: str = typer.Option(None, help="Address of the BYO agent API"),
    sample_agent: str = typer.Option(None, help="Automatically spawn a sample agent in the background (rag_app or research_agent)"),
    chaos_profile: str = typer.Option(..., help="The X-Chaos-Profile value to inject (e.g. latency_spike, schema_error)"),
    eval_file: str = typer.Option("custom_evals.yaml", help="Path to evaluation assets"),
    limit: int = typer.Option(5, help="Number of benchmark samples to evaluate in Chaos Mode")
):
    """
    Injects a chaos header against an endpoint, recording the degradation of the capability.
    """
    print_banner()
    agent_process = None
    if sample_agent:
        agent_process, target_url = _spawn_sample_agent(sample_agent)
        if not agent_process:
            console.print(f"[bold red]Unknown sample agent: {sample_agent}[/bold red]")
            return
            
    if not target_url:
        console.print("[bold red]❌ --target-url is required if not using --sample-agent[/bold red]")
        return

    console.print(f"[bold red]=> 🔥 INJECTING CHAOS PROFILE: {chaos_profile} 🔥[/bold red]")
    if chaos_profile.startswith("client_"):
        console.print("[bold yellow]⚠️ WARNING: Client-side prompt mutations append adversarial instructions which slightly increases your prompt token payload size![/bold yellow]")
    
    standard_evals = load_standard_benchmark(scenario, limit=limit)
    if standard_evals:
        evals_to_run = standard_evals
    else:
        target_eval = next((e for e in load_local_evals(eval_file) if e.id == scenario), None)
        if not target_eval:
            console.print(f"[bold red]Scenario {scenario} not found.[/bold red]")
            if agent_process: agent_process.terminate()
            return
        evals_to_run = [target_eval]

    try:
        generator = LoadGenerator(target_url)
        judge = LLMJudgeProvider()
        scores = []
        
        for eval_task in evals_to_run:
            resp = asyncio.run(generator.run_scenario(scenario, eval_task.input_payload, chaos_profile=chaos_profile))
            agent_output_text = str(resp.get("data", resp.get("error_message", "No output")))
            evaluation = judge.score_run(eval_task.expected_behavior_rubric, agent_output_text)
            scores.append(evaluation.get("score", 0))
            
        if not scores:
            return
            
        final_score = int(sum(scores) / len(scores))
        
        hist = get_history(scenario)
        baselines = [h for h in hist if h["run_type"] == "baseline"]
        original_baseline = baselines[-1]["score"] if baselines else 0
        
        record_run(scenario, "chaos", final_score, details={"chaos_profile": chaos_profile, "sample_size": len(scores)})
        print_chaos_result(scenario, chaos_profile, final_score, original_baseline)
    finally:
        if agent_process:
            agent_process.terminate()


@app.command()
def history(scenario: str = typer.Option(None, help="Specific scenario ID to view history for. Views all if blank.")):
    """
    Prints the beautiful historical trend of agent scores and the Production Reliability metric.
    """
    print_banner()
    
    hist = get_history(scenario)
    if not hist:
        console.print("[bold yellow]No local history found. Run a benchmark first![/bold yellow]")
        return
    
    scenarios = {h["scenario"] for h in hist} if not scenario else {scenario}
    for s in scenarios:
        s_hist = [h for h in hist if h["scenario"] == s]
        reliability = calculate_production_reliability(scenario=s)
        print_history_trends(s, s_hist, reliability)

if __name__ == "__main__":
    app()
