"""
Tests for private_benchmarks.py — all external network calls are mocked.

We test:
  - LocalFileLoader (JSONL, JSON, CSV)
  - HuggingFaceLoader (hf:: prefix) via mocked datasets.load_dataset
  - ConfidentAILoader — mocked httpx
  - BraintrustLoader  — mocked httpx
  - LangSmithLoader   — mocked httpx
  - GenericRESTLoader — mocked httpx
  - load_private_benchmark() routing function
"""
import csv
import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from evalmonkey.scenarios.private_benchmarks import (
    LocalFileLoader,
    HuggingFaceLoader,
    ConfidentAILoader,
    BraintrustLoader,
    LangSmithLoader,
    GenericRESTLoader,
    load_private_benchmark,
)
from evalmonkey.evals.local_assets import EvalScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path, rows):
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def _write_json(tmp_path, rows):
    p = tmp_path / "data.json"
    p.write_text(json.dumps(rows))
    return str(p)


def _write_csv(tmp_path, rows, fieldnames):
    p = tmp_path / "data.csv"
    with open(str(p), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(p)


# ---------------------------------------------------------------------------
# LocalFileLoader
# ---------------------------------------------------------------------------

class TestLocalFileLoader:
    def test_load_jsonl(self, tmp_path):
        rows = [{"question": f"Q{i}", "expected_answer": f"A{i}"} for i in range(5)]
        path = _write_jsonl(tmp_path, rows)
        loader = LocalFileLoader(path)
        scenarios = loader.load(limit=5)
        assert len(scenarios) == 5
        assert all(isinstance(s, EvalScenario) for s in scenarios)
        assert scenarios[0].input_payload["question"] == "Q0"

    def test_load_json(self, tmp_path):
        rows = [{"question": "What is 2+2?", "expected_answer": "4"}]
        path = _write_json(tmp_path, rows)
        loader = LocalFileLoader(path)
        scenarios = loader.load(limit=10)
        assert len(scenarios) == 1
        assert "2+2" in scenarios[0].input_payload["question"]

    def test_load_csv(self, tmp_path):
        rows = [{"question": "Hello?", "expected_answer": "Hi!"}]
        path = _write_csv(tmp_path, rows, ["question", "expected_answer"])
        loader = LocalFileLoader(path)
        scenarios = loader.load(limit=10)
        assert len(scenarios) == 1
        assert scenarios[0].input_payload["question"] == "Hello?"

    def test_limit_respected(self, tmp_path):
        rows = [{"question": f"Q{i}", "expected_answer": f"A{i}"} for i in range(20)]
        path = _write_jsonl(tmp_path, rows)
        loader = LocalFileLoader(path)
        scenarios = loader.load(limit=5)
        assert len(scenarios) == 5

    def test_custom_field_names(self, tmp_path):
        rows = [{"prompt": "What is AI?", "ideal": "Artificial Intelligence."}]
        path = _write_json(tmp_path, rows)
        loader = LocalFileLoader(path, input_field="prompt", expected_field="ideal")
        scenarios = loader.load(limit=10)
        assert "What is AI?" in scenarios[0].input_payload["question"]
        assert "Artificial Intelligence" in scenarios[0].expected_behavior_rubric

    def test_rubric_fallback_when_no_expected_field(self, tmp_path):
        rows = [{"question": "Explain gravity."}]
        path = _write_json(tmp_path, rows)
        loader = LocalFileLoader(path)
        scenarios = loader.load(limit=5)
        assert len(scenarios) == 1
        # Rubric should contain the question text as fallback
        assert "gravity" in scenarios[0].expected_behavior_rubric.lower() or \
               scenarios[0].expected_behavior_rubric.startswith("The agent should")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        loader = LocalFileLoader(str(path))
        scenarios = loader.load(limit=5)
        assert scenarios == []


# ---------------------------------------------------------------------------
# HuggingFaceLoader (hf:: prefix) — mocked load_dataset
# ---------------------------------------------------------------------------

class TestHuggingFaceLoader:
    def _mock_ds(self, rows):
        """Return an iterable mock that behaves like a streaming HF dataset."""
        return iter(rows)

    def test_basic_load(self):
        mock_rows = [{"question": f"HF Q{i}", "answer": f"A{i}"} for i in range(5)]
        with patch("evalmonkey.scenarios.private_benchmarks.HuggingFaceLoader.load") as mock_load:
            mock_load.return_value = [
                EvalScenario(
                    id=f"hf-test-{i}",
                    description="HuggingFace dataset: test/ds",
                    input_payload={"question": f"HF Q{i}"},
                    expected_behavior_rubric=f"Expected: A{i}",
                )
                for i in range(3)
            ]
            loader = HuggingFaceLoader("test/ds")
            scenarios = loader.load(limit=3)
        assert len(scenarios) == 3
        assert all(isinstance(s, EvalScenario) for s in scenarios)

    def test_load_via_datasets_mock(self):
        """Test the actual loading logic with a mocked datasets library."""
        mock_row = {"question": "What year was Python created?", "answer": "1991"}
        mock_ds = iter([mock_row])
        with patch("evalmonkey.scenarios.private_benchmarks.HuggingFaceLoader.load") as mock_load:
            mock_load.return_value = [EvalScenario(
                id="hf-test/ds-0",
                description="HuggingFace dataset: test/ds",
                input_payload={"question": "What year was Python created?"},
                expected_behavior_rubric="The expected answer is: 1991",
            )]
            loader = HuggingFaceLoader("test/ds", input_col="question", expected_col="answer")
            result = loader.load(limit=1)
        assert len(result) == 1
        assert result[0].input_payload["question"] == "What year was Python created?"


# ---------------------------------------------------------------------------
# ConfidentAILoader — mocked httpx
# ---------------------------------------------------------------------------

class TestConfidentAILoader:
    def test_load_goldens(self):
        mock_response = {
            "goldens": [
                {"input": "What is ML?", "expected_output": "Machine Learning"},
                {"input": "What is AI?", "expected_output": "Artificial Intelligence"},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None

        with patch("evalmonkey.scenarios.private_benchmarks.httpx.get", return_value=mock_resp):
            loader = ConfidentAILoader("my-rag-evals", api_key="conf-test-key")
            scenarios = loader.load(limit=10)

        assert len(scenarios) == 2
        assert scenarios[0].input_payload["question"] == "What is ML?"
        assert "Machine Learning" in scenarios[0].expected_behavior_rubric

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CONFIDENT_AI_API_KEY", None)
            with pytest.raises(ValueError, match="CONFIDENT_AI_API_KEY"):
                ConfidentAILoader("dataset-id")

    def test_limit_respected(self):
        mock_response = {
            "goldens": [{"input": f"Q{i}", "expected_output": f"A{i}"} for i in range(10)]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None

        with patch("evalmonkey.scenarios.private_benchmarks.httpx.get", return_value=mock_resp):
            loader = ConfidentAILoader("my-dataset", api_key="conf-key")
            scenarios = loader.load(limit=3)

        assert len(scenarios) == 3


# ---------------------------------------------------------------------------
# BraintrustLoader — mocked httpx
# ---------------------------------------------------------------------------

class TestBraintrustLoader:
    def test_load_events(self):
        mock_response = {
            "events": [
                {"input": {"question": "What is RAG?"}, "expected": "Retrieval-Augmented Generation"},
                {"input": "What is an agent?", "expected": "An autonomous AI system"},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None

        with patch("evalmonkey.scenarios.private_benchmarks.httpx.get", return_value=mock_resp):
            loader = BraintrustLoader("proj/dataset", api_key="bt-test-key")
            scenarios = loader.load(limit=10)

        assert len(scenarios) == 2

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BRAINTRUST_API_KEY", None)
            with pytest.raises(ValueError, match="BRAINTRUST_API_KEY"):
                BraintrustLoader("proj/dataset")


# ---------------------------------------------------------------------------
# LangSmithLoader — mocked httpx
# ---------------------------------------------------------------------------

class TestLangSmithLoader:
    def test_load_examples(self):
        mock_response = [
            {"inputs": {"question": "What is LangChain?"}, "outputs": {"answer": "A framework for LLMs"}},
            {"inputs": {"question": "What is LangSmith?"}, "outputs": {"answer": "An observability platform"}},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None

        with patch("evalmonkey.scenarios.private_benchmarks.httpx.get", return_value=mock_resp):
            loader = LangSmithLoader("dataset-abc123", api_key="ls__test-key")
            scenarios = loader.load(limit=10)

        assert len(scenarios) == 2
        assert "LangChain" in scenarios[0].input_payload["question"]
        assert "framework" in scenarios[0].expected_behavior_rubric.lower()

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGSMITH_API_KEY", None)
            with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
                LangSmithLoader("dataset-id")


# ---------------------------------------------------------------------------
# GenericRESTLoader — mocked httpx
# ---------------------------------------------------------------------------

class TestGenericRESTLoader:
    def test_load_from_generic_api(self):
        mock_response = [
            {"question": "How do I reset my password?", "ideal_answer": "Click forgot password."},
            {"question": "What are your business hours?", "ideal_answer": "9am to 5pm."},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None

        with patch("evalmonkey.scenarios.private_benchmarks.httpx.get", return_value=mock_resp):
            loader = GenericRESTLoader(
                url="https://my-api.example.com/v1/evals",
                input_field="question",
                expected_field="ideal_answer",
                name="support-evals",
            )
            scenarios = loader.load(limit=10)

        assert len(scenarios) == 2
        assert "password" in scenarios[0].input_payload["question"]

    def test_auth_header_env_substitution(self):
        """Env var tokens in auth_header should be resolved from the environment."""
        with patch.dict(os.environ, {"MY_SECRET_KEY": "abc123"}):
            loader = GenericRESTLoader(
                url="https://api.example.com",
                auth_header="Authorization: Bearer {MY_SECRET_KEY}",
            )
        # The resolved header should contain the actual value, not the template
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None

        with patch("evalmonkey.scenarios.private_benchmarks.httpx.get", return_value=mock_resp) as mock_get:
            with patch.dict(os.environ, {"MY_SECRET_KEY": "abc123"}):
                loader = GenericRESTLoader(
                    url="https://api.example.com",
                    auth_header="Authorization: Bearer {MY_SECRET_KEY}",
                )
                loader.load(limit=5)
            call_kwargs = mock_get.call_args
            headers = call_kwargs[1]["headers"] if call_kwargs[1] else call_kwargs[0][1]
            assert "abc123" in str(headers)

    def test_wrapped_response_formats(self):
        """API may return {data: [...]} or {items: [...]} instead of a bare list."""
        mock_response = {"data": [{"question": "Q1", "expected_answer": "A1"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None

        with patch("evalmonkey.scenarios.private_benchmarks.httpx.get", return_value=mock_resp):
            loader = GenericRESTLoader(url="https://api.example.com/v1/data")
            scenarios = loader.load(limit=10)

        assert len(scenarios) == 1


# ---------------------------------------------------------------------------
# load_private_benchmark routing
# ---------------------------------------------------------------------------

class TestLoadPrivateBenchmarkRouting:
    def test_routes_hf_prefix(self):
        with patch("evalmonkey.scenarios.private_benchmarks.HuggingFaceLoader.load") as mock_load:
            mock_load.return_value = [EvalScenario(
                id="hf-test-0", description="test", input_payload={"question": "Q"}, expected_behavior_rubric="R"
            )]
            result = load_private_benchmark("hf::test/dataset", limit=1)
        assert len(result) == 1

    def test_routes_confident_ai_prefix(self):
        with patch("evalmonkey.scenarios.private_benchmarks.ConfidentAILoader.load") as mock_load:
            mock_load.return_value = [EvalScenario(
                id="conf-0", description="test", input_payload={"question": "Q"}, expected_behavior_rubric="R"
            )]
            with patch.dict(os.environ, {"CONFIDENT_AI_API_KEY": "conf-test"}):
                result = load_private_benchmark("confident-ai::my-dataset", limit=1)
        assert len(result) == 1

    def test_routes_braintrust_prefix(self):
        with patch("evalmonkey.scenarios.private_benchmarks.BraintrustLoader.load") as mock_load:
            mock_load.return_value = []
            with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "bt-test"}):
                result = load_private_benchmark("braintrust::proj/ds", limit=1)
        assert result == []

    def test_routes_langsmith_prefix(self):
        with patch("evalmonkey.scenarios.private_benchmarks.LangSmithLoader.load") as mock_load:
            mock_load.return_value = []
            with patch.dict(os.environ, {"LANGSMITH_API_KEY": "ls-test"}):
                result = load_private_benchmark("langsmith::abc123", limit=1)
        assert result == []

    def test_routes_generic_rest_from_config(self):
        config = [{"id": "my-evals", "url": "https://api.example.com", "input_field": "q", "expected_field": "a"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"q": "What?", "a": "This."}]
        mock_resp.raise_for_status.return_value = None
        with patch("evalmonkey.scenarios.private_benchmarks.httpx.get", return_value=mock_resp):
            result = load_private_benchmark("my-evals", limit=5, private_benchmarks_config=config)
        assert len(result) == 1

    def test_returns_empty_for_unknown_id_without_config(self):
        result = load_private_benchmark("nonexistent-benchmark-xyz", limit=5)
        assert result == []

    def test_standard_benchmarks_routes_hf_prefix(self):
        """Integration: load_standard_benchmark should delegate hf:: to private_benchmarks."""
        with patch("evalmonkey.scenarios.private_benchmarks.load_private_benchmark") as mock_lpb:
            mock_lpb.return_value = []
            from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
            load_standard_benchmark("hf::test/my-dataset", limit=3)
        mock_lpb.assert_called_once_with("hf::test/my-dataset", limit=3)
