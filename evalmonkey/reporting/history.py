import os
import json
from datetime import datetime

HISTORY_FILE = os.path.expanduser("~/.evalmonkey/history.json")

def _ensure_history_file():
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

def record_run(scenario: str, run_type: str, score: int, details: dict = None):
    """
    Saves a benchmarking run to local history. run_type should be 'baseline' or 'chaos'.
    """
    _ensure_history_file()
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
        
    history.append({
        "timestamp": datetime.now().isoformat(),
        "scenario": scenario,
        "run_type": run_type,
        "score": score,
        "details": details or {}
    })
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_history(scenario: str = None) -> list:
    """Returns local history, optionally filtered by scenario."""
    _ensure_history_file()
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
        
    if scenario:
        history = [h for h in history if h.get("scenario") == scenario]
    return history

def calculate_production_reliability(scenario: str = None) -> float:
    """
    Calculates the 'Production Reliability' metric.
    We define it as: 
    60% weighting on the latest 'baseline' capability score 
    40% weighting on the latest 'chaos' resilience score.
    Returns a score 0-100.
    """
    history = get_history(scenario)
    if not history:
        return 0.0
        
    # Get the most recent baseline
    baselines = [h for h in history if h["run_type"] == "baseline"]
    latest_baseline = baselines[-1]["score"] if baselines else 0.0
    
    # Get the most recent chaos test
    chaos_tests = [h for h in history if h["run_type"] == "chaos"]
    latest_chaos = chaos_tests[-1]["score"] if chaos_tests else 0.0
    
    return (latest_baseline * 0.6) + (latest_chaos * 0.4)
