from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich import box

console = Console()

def print_banner():
    banner = Text(
        r"""
  ______            _ __  __             _                
 |  ____|          | |  \/  |           | |               
 | |____   ____ _  | | \  / | ___  _ __ | | _____ _   _   
 |  __\ \ / / _` | | | |\/| |/ _ \| '_ \| |/ / _ \ | | |  
 | |___\ V / (_| | | | |  | | (_) | | | |   <  __/ |_| |  
 |______\_/ \__,_| |_|_|  |_|\___/|_| |_|_|\_\___|\__, |  
                                                   __/ |  
                                                  |___/   
        """,
        style="bold magenta"
    )
    console.print(Align.center(banner))
    console.print(Align.center(Text("Agent Benchmarking & Chaos Framework", style="dim italic")))
    console.print("\n")


def print_benchmark_score(scenario_name: str, score: int, reasoning: str, baseline_score: int | None = None):
    if baseline_score is not None:
        diff = score - baseline_score
        diff_text = f"[bold green]+{diff}[/bold green]" if diff > 0 else (f"[bold red]{diff}[/bold red]" if diff < 0 else "[bold yellow]0[/bold yellow]")
        title_color = "green" if diff > 0 else "red" if diff < 0 else "yellow"
        score_display = f"[bold {title_color}]{score}/100[/bold {title_color}] (Diff: {diff_text} )"
    else:
        score_display = f"[bold cyan]{score}/100[/bold cyan]"

    table = Table(box=box.MINIMAL_DOUBLE_HEAD, show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Scenario", f"[bold white]{scenario_name}[/bold white]")
    table.add_row("Score", score_display)
    if baseline_score is not None:
        table.add_row("Previous", f"{baseline_score}/100")
    table.add_row("Reasoning", f"[dim]{reasoning}[/dim]")

    panel = Panel(table, title="[bold]Benchmark Results[/bold]", border_style="cyan", padding=(1, 2))
    console.print("\n")
    console.print(Align.center(panel))


def print_chaos_result(scenario_name: str, profile: str, score: int, baseline_score: int):
    diff = score - baseline_score
    if abs(diff) <= 10:
        resilience = "[bold green]HIGHLY RESILIENT[/bold green]"
        color = "green"
    else:
        resilience = "[bold red]DEGRADED CAPABILITY[/bold red]"
        color = "red"

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_row("Scenario:", scenario_name)
    table.add_row("Chaos Profile:", f"[bold magenta]{profile}[/bold magenta]")
    table.add_row("Baseline Capability Score:", f"{baseline_score}")
    table.add_row("Post-Chaos Resilience Score:", f"[{color}]{score}[/{color}]")
    table.add_row("Status:", resilience)

    panel = Panel(table, title="[bold red]🔥 Chaos Engineering Report 🔥[/bold red]", border_style="red", box=box.HEAVY)
    console.print("\n")
    console.print(Align.center(panel))


def print_history_trends(scenario_name: str, history: list, production_reliability: float):
    if not history:
        console.print(f"[bold yellow]No history found for scenario {scenario_name}.[/bold yellow]")
        return
        
    console.print(f"\n[bold magenta]📈 Historical Trend for: {scenario_name} 📈[/bold magenta]")
    
    # Prune to last 10
    history = history[-10:]
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    table.add_column("Date", style="dim")
    table.add_column("Run Type", style="cyan")
    table.add_column("Score", justify="right")
    
    for h in history:
        date_str = h["timestamp"].split("T")[0] + " " + h["timestamp"].split("T")[1][:5]
        score_color = "green" if h["score"] > 80 else "yellow" if h["score"] > 50 else "red"
        
        table.add_row(
            date_str,
            h["run_type"].upper(),
            f"[bold {score_color}]{h['score']}[/bold {score_color}]"
        )
        
    console.print(table)
    
    # Production Reliability Printout
    rel_color = "green" if production_reliability > 80 else "yellow" if production_reliability > 60 else "red"
    console.print(f"\n🚀 [bold white]Production Reliability Metric:[/bold white] [bold {rel_color}]{production_reliability:.1f} / 100.0[/bold {rel_color}]")
    console.print("[dim](Calculated as 60% of most recent baseline capability + 40% most recent chaos resilience)[/dim]\n")
