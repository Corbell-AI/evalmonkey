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
    # Unknown benchmark returns empty list
    scenarios = load_standard_benchmark("unknown")
    assert len(scenarios) == 0


def test_load_standard_benchmark_gsm8k_mocked():
    """Verify gsm8k loader builds correct EvalScenario objects without hitting the network."""
    from unittest.mock import patch
    mock_item = {
        "question": "If John has 3 apples and buys 2 more, how many does he have?",
        "answer": "#### 5",
    }
    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter([mock_item])
        scenarios = load_standard_benchmark("gsm8k", limit=1)

    assert isinstance(scenarios, list)
    assert len(scenarios) == 1
    assert "apples" in scenarios[0].input_payload["question"]
    assert "5" in scenarios[0].expected_behavior_rubric
