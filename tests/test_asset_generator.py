"""
Tests for evalmonkey.evals.asset_generator
"""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evalmonkey.evals.asset_generator import (
    EvalAssetGenerator,
    FailingTrace,
    build_output_dir,
    DEFAULT_FAILURE_THRESHOLD,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_trace(score: int, chaos: str = None) -> FailingTrace:
    return FailingTrace(
        scenario="gsm8k",
        eval_id=f"gsm8k_{score}",
        input_payload={"question": "What is 2 + 2?"},
        agent_output="The answer is 5." if score < 70 else "The answer is 4.",
        expected_rubric="Agent must return 4.",
        score=score,
        reasoning="Subtracted instead of added." if score < 70 else "Correct.",
        chaos_profile=chaos,
    )


# ─── record_failure ──────────────────────────────────────────────────────────

def test_record_failure_accumulates_below_threshold():
    gen = EvalAssetGenerator(failure_threshold=70)
    gen.record_failure(_make_trace(40))
    gen.record_failure(_make_trace(60))
    assert gen.failure_count == 2
    assert gen.has_failures


def test_record_failure_ignores_passing_traces():
    gen = EvalAssetGenerator(failure_threshold=70)
    gen.record_failure(_make_trace(80))
    gen.record_failure(_make_trace(100))
    assert gen.failure_count == 0
    assert not gen.has_failures


def test_record_failure_boundary_below_threshold():
    gen = EvalAssetGenerator(failure_threshold=70)
    gen.record_failure(_make_trace(69))
    assert gen.failure_count == 1


def test_record_failure_boundary_at_threshold():
    gen = EvalAssetGenerator(failure_threshold=70)
    gen.record_failure(_make_trace(70))
    # threshold is strict <, so 70 is NOT a failure
    assert gen.failure_count == 0


def test_record_failure_with_chaos_profile():
    gen = EvalAssetGenerator(failure_threshold=70)
    gen.record_failure(_make_trace(30, chaos="client_prompt_injection"))
    assert gen.failure_count == 1
    assert gen._failures[0].chaos_profile == "client_prompt_injection"


# ─── generate_improvement_evals ──────────────────────────────────────────────

@patch("evalmonkey.evals.asset_generator.call_llm")
def test_generate_improvement_evals_returns_list(mock_llm):
    evals = [
        {"id": "test_1", "description": "desc", "input_payload": {"question": "q"}, "expected_behavior_rubric": "r"},
        {"id": "test_2", "description": "desc2", "input_payload": {"question": "q2"}, "expected_behavior_rubric": "r2"},
    ]
    mock_llm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(evals)))]
    )
    gen = EvalAssetGenerator()
    gen.record_failure(_make_trace(30))
    result = gen.generate_improvement_evals(n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["id"] == "test_1"


@patch("evalmonkey.evals.asset_generator.call_llm")
def test_generate_improvement_evals_unwraps_wrapper_key(mock_llm):
    """LLM sometimes returns {"evals": [...]} — we must unwrap it."""
    evals = [{"id": "x", "description": "d", "input_payload": {"question": "q"}, "expected_behavior_rubric": "r"}]
    mock_llm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({"evals": evals})))]
    )
    gen = EvalAssetGenerator()
    gen.record_failure(_make_trace(20))
    result = gen.generate_improvement_evals()
    assert isinstance(result, list)
    assert len(result) == 1


@patch("evalmonkey.evals.asset_generator.call_llm")
def test_generate_improvement_evals_fallback_on_error(mock_llm):
    """If LLM throws, we get a stub list — not an exception."""
    mock_llm.side_effect = Exception("LLM unavailable")
    gen = EvalAssetGenerator()
    for _ in range(3):
        gen.record_failure(_make_trace(10))
    result = gen.generate_improvement_evals(n=3)
    assert isinstance(result, list)
    assert len(result) == 3  # one stub per failure, capped at n


def test_generate_improvement_evals_returns_empty_when_no_failures():
    gen = EvalAssetGenerator()
    result = gen.generate_improvement_evals()
    assert result == []


# ─── save ────────────────────────────────────────────────────────────────────

@patch("evalmonkey.evals.asset_generator.call_llm")
def test_save_creates_output_files(mock_llm, tmp_path):
    evals = [{"id": "e1", "description": "d", "input_payload": {"question": "q"}, "expected_behavior_rubric": "r"}]
    mock_llm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(evals)))]
    )
    gen = EvalAssetGenerator()
    gen.record_failure(_make_trace(25))

    saved = gen.save(str(tmp_path / "out"))
    saved_path = Path(saved)

    assert (saved_path / "traces.json").exists()
    assert (saved_path / "evals.json").exists()
    assert (saved_path / "improvement_prompt.md").exists()


@patch("evalmonkey.evals.asset_generator.call_llm")
def test_save_traces_json_has_correct_shape(mock_llm, tmp_path):
    mock_llm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps([])))]
    )
    gen = EvalAssetGenerator()
    gen.record_failure(_make_trace(40, chaos="client_empty_payload"))

    saved = gen.save(str(tmp_path / "out"))
    traces = json.loads((Path(saved) / "traces.json").read_text())

    assert len(traces) == 1
    t = traces[0]
    assert t["scenario"] == "gsm8k"
    assert t["score"] == 40
    assert t["chaos_profile"] == "client_empty_payload"
    assert "input_payload" in t
    assert "agent_output" in t


@patch("evalmonkey.evals.asset_generator.call_llm")
def test_save_improvement_prompt_contains_key_sections(mock_llm, tmp_path):
    mock_llm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps([])))]
    )
    gen = EvalAssetGenerator()
    gen.record_failure(_make_trace(15))

    saved = gen.save(str(tmp_path / "out"))
    prompt = (Path(saved) / "improvement_prompt.md").read_text()

    assert "## Summary" in prompt
    assert "## What went wrong" in prompt
    assert "## New Evaluation Scenarios to Fix" in prompt
    assert "Coding Agent Prompt" in prompt


# ─── export_to_langfuse ──────────────────────────────────────────────────────

def test_export_to_langfuse_skipped_when_no_keys():
    gen = EvalAssetGenerator()
    gen.record_failure(_make_trace(20))
    # Remove keys from env just in case
    env = {k: v for k, v in os.environ.items() if k not in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")}
    with patch.dict(os.environ, env, clear=True):
        result = gen.export_to_langfuse("my_dataset")
    assert result is False


@patch("evalmonkey.evals.asset_generator.call_llm")
@patch("requests.post")
def test_export_to_langfuse_calls_api_when_keys_set(mock_post, mock_llm, tmp_path):
    evals = [{"id": "e1", "description": "d", "input_payload": {"question": "q"}, "expected_behavior_rubric": "r"}]
    mock_llm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(evals)))]
    )
    mock_post.return_value = MagicMock(status_code=200)

    gen = EvalAssetGenerator()
    gen.record_failure(_make_trace(20))

    env_patch = {
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
    }
    with patch.dict(os.environ, env_patch):
        result = gen.export_to_langfuse("test_dataset")

    assert result is True
    # One call to create dataset + one per eval item
    assert mock_post.call_count >= 2


# ─── build_output_dir ────────────────────────────────────────────────────────

def test_build_output_dir_contains_scenario():
    d = build_output_dir("gsm8k")
    assert "gsm8k" in d
    assert d.startswith("output/")


def test_build_output_dir_unique_per_call():
    import time
    d1 = build_output_dir("arc")
    time.sleep(1)
    d2 = build_output_dir("arc")
    assert d1 != d2


# ─── FailingTrace.to_dict ────────────────────────────────────────────────────

def test_failing_trace_to_dict_roundtrip():
    t = _make_trace(30, chaos="client_typo_injection")
    d = t.to_dict()
    assert d["score"] == 30
    assert d["chaos_profile"] == "client_typo_injection"
    assert d["scenario"] == "gsm8k"
    assert "timestamp" in d
    # Verify it JSON-serialises cleanly
    json.dumps(d)
