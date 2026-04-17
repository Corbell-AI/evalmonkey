import os
import json
import pytest
from evalmonkey.reporting.history import record_run, get_history, calculate_production_reliability
from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark

@pytest.fixture(autouse=True)
def mock_history_file(tmp_path, monkeypatch):
    """Mocks the history file path to a temp directory for tests."""
    mock_file = tmp_path / "test_history.json"
    monkeypatch.setattr("evalmonkey.reporting.history.HISTORY_FILE", str(mock_file))
    yield

def test_history_recording():
    # Record a baseline
    record_run("gsm8k", "baseline", 85, {"reasoning": "Standard."})
    hist = get_history("gsm8k")
    assert len(hist) == 1
    assert hist[0]["score"] == 85
    assert hist[0]["run_type"] == "baseline"

    # Record chaos
    record_run("gsm8k", "chaos", 40, {"chaos_profile": "Latency"})
    hist = get_history("gsm8k")
    assert len(hist) == 2
    
def test_production_reliability_calculation():
    # Baseline only
    record_run("gsm8k", "baseline", 100)
    # PR should be 100 * 0.6 + 0 = 60.0 since no chaos yet
    assert calculate_production_reliability("gsm8k") == 60.0
    
    # Add chaos
    record_run("gsm8k", "chaos", 50)
    # PR should be 100 * 0.6 + 50 * 0.4 = 60 + 20 = 80.0
    assert calculate_production_reliability("gsm8k") == 80.0

def test_load_standard_benchmark_stub():
    # If using XLAM
    scenarios = load_standard_benchmark("unknown")
    assert len(scenarios) == 0

    # Tests handling GSM8k fallback if datasets package is available
    gsm_scenarios = load_standard_benchmark("gsm8k", limit=1)
    # Only verify types and existence without mocking out HF completely, depends on network unless mocked, 
    # but since local test env may lack downloading access easily, we verify it doesn't crash.
    assert isinstance(gsm_scenarios, list)
