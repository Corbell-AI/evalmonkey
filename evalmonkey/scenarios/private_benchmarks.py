"""
External and Private Dataset Support
=====================================
EvalMonkey supports three ways to bring your own evaluation data:

1. Local files  — `evalmonkey run-benchmark --dataset my_cases.jsonl`
2. HuggingFace  — `--scenario hf::org/dataset-name` (any public or gated HF dataset)
3. Generic REST — configure a URL in evalmonkey.yaml under `private_benchmarks`
4. Eval platforms you already use (Confident AI, Braintrust, LangSmith) — see below

EvalMonkey acts as a harness: it fetches your data, normalizes it to EvalScenario,
then runs chaos injection + LLM scoring. Your data stays on your machine.

Benchmark ID convention:
  - hf::<org>/<dataset>            → any HuggingFace dataset
  - confident-ai::<dataset_id>     → Confident AI (DeepEval) dataset
  - braintrust::<project>/<dataset>→ Braintrust dataset
  - langsmith::<dataset_id>        → LangSmith dataset
  - <configured-id>                → private_benchmarks entry in evalmonkey.yaml
"""
from __future__ import annotations

import os
import json
import csv
import io
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import httpx

from evalmonkey.evals.local_assets import EvalScenario


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PrivateBenchmarkLoader(ABC):
    """Abstract base class for all private/external benchmark loaders."""

    @abstractmethod
    def load(self, limit: int = 10) -> List[EvalScenario]:
        """Fetch and normalise dataset rows into EvalScenario objects."""


# ---------------------------------------------------------------------------
# Local-file loader (JSONL / JSON / CSV)
# ---------------------------------------------------------------------------

class LocalFileLoader(PrivateBenchmarkLoader):
    """
    Load from a local file (JSONL, JSON, or CSV).

    Expected field names (flexible — any name works for input_field):
        input_field        → becomes input_payload[request_key]
        expected_field     → becomes expected_behavior_rubric
        id_field           → (optional) scenario ID
        description_field  → (optional) human-readable description

    Example JSONL row:
        {"question": "What is 2+2?", "expected_answer": "4"}

    Example CSV:
        question,expected_answer
        "What is 2+2?","4"
    """

    def __init__(
        self,
        filepath: str,
        input_field: str = "question",
        expected_field: str = "expected_answer",
        id_field: str = "id",
        description_field: str = "description",
    ):
        self.filepath = filepath
        self.input_field = input_field
        self.expected_field = expected_field
        self.id_field = id_field
        self.description_field = description_field

    def load(self, limit: int = 10) -> List[EvalScenario]:
        rows = self._read_file()
        return self._normalise(rows, limit)

    def _read_file(self) -> List[Dict[str, Any]]:
        fp = self.filepath
        if fp.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        elif fp.endswith(".json"):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        elif fp.endswith(".csv"):
            with open(fp, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]
        else:
            # Try JSONL as fallback
            with open(fp, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            try:
                return [json.loads(l) for l in lines]
            except json.JSONDecodeError:
                raise ValueError(f"Unsupported file format: {fp}. Supported: .jsonl, .json, .csv")

    def _normalise(self, rows: List[Dict], limit: int) -> List[EvalScenario]:
        scenarios = []
        for i, row in enumerate(rows[:limit]):
            question = row.get(self.input_field, str(row))
            rubric = row.get(self.expected_field, "")
            if not rubric:
                rubric = f"The agent should correctly answer: {question}"
            scenario_id = str(row.get(self.id_field, f"local-{i}"))
            description = str(row.get(self.description_field, f"Local eval #{i}"))
            scenarios.append(EvalScenario(
                id=scenario_id,
                description=description,
                input_payload={"question": question},
                expected_behavior_rubric=rubric,
            ))
        return scenarios


# ---------------------------------------------------------------------------
# HuggingFace direct loader  (hf::<org>/<dataset>)
# ---------------------------------------------------------------------------

class HuggingFaceLoader(PrivateBenchmarkLoader):
    """
    Load any HuggingFace dataset by its repository ID.

    Usage:  --scenario hf::org/dataset-name
    Options configurable via loader kwargs:
        split         default "train"
        input_col     column name for the question/input
        expected_col  column name for the expected answer (optional)
        config_name   HF dataset config name (optional)
    """

    def __init__(
        self,
        dataset_id: str,
        split: str = "train",
        input_col: str = "question",
        expected_col: Optional[str] = None,
        config_name: Optional[str] = None,
    ):
        self.dataset_id = dataset_id
        self.split = split
        self.input_col = input_col
        self.expected_col = expected_col
        self.config_name = config_name

    def load(self, limit: int = 10) -> List[EvalScenario]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("HuggingFace 'datasets' package required. Run: pip install datasets")

        kwargs: Dict[str, Any] = {
            "split": self.split,
            "streaming": True,
            "trust_remote_code": False,
        }
        if self.config_name:
            kwargs["name"] = self.config_name

        ds = load_dataset(self.dataset_id, **kwargs)

        scenarios: List[EvalScenario] = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            # Try to find a sensible input column
            question = self._get_col(row, self.input_col) or self._first_string_col(row)
            rubric_val = self._get_col(row, self.expected_col) if self.expected_col else None
            rubric = (
                f"The expected answer is: {rubric_val}"
                if rubric_val
                else f"The agent should correctly answer the following question: {question}"
            )
            scenarios.append(EvalScenario(
                id=f"hf-{self.dataset_id.replace('/', '-')}-{i}",
                description=f"HuggingFace dataset: {self.dataset_id} (row {i})",
                input_payload={"question": str(question)},
                expected_behavior_rubric=rubric,
            ))
        return scenarios

    @staticmethod
    def _get_col(row: dict, col: Optional[str]) -> Optional[str]:
        if col and col in row:
            return str(row[col])
        return None

    @staticmethod
    def _first_string_col(row: dict) -> str:
        for v in row.values():
            if isinstance(v, str) and len(v) > 5:
                return v
        return str(list(row.values())[0])


# ---------------------------------------------------------------------------
# Confident AI (DeepEval) loader   (confident-ai::<dataset_id>)
# ---------------------------------------------------------------------------

class ConfidentAILoader(PrivateBenchmarkLoader):
    """
    Load a dataset from Confident AI (DeepEval cloud platform).

    Requires: CONFIDENT_AI_API_KEY in .env
    Dataset ID: the name or UUID of a dataset in your Confident AI workspace.

    Usage:  --scenario confident-ai::my-rag-evals
    """

    BASE_URL = "https://api.confident-ai.com/v1"

    def __init__(self, dataset_id: str, api_key: Optional[str] = None):
        self.dataset_id = dataset_id
        self.api_key = api_key or os.getenv("CONFIDENT_AI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "CONFIDENT_AI_API_KEY not set. Add it to your .env file.\n"
                "Get your key from: https://app.confident-ai.com → Settings → API Keys"
            )

    def load(self, limit: int = 10) -> List[EvalScenario]:
        url = f"{self.BASE_URL}/datasets/{self.dataset_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = httpx.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        goldens = data.get("goldens", data.get("data", []))
        return self._normalise(goldens, limit)

    def _normalise(self, goldens: list, limit: int) -> List[EvalScenario]:
        scenarios = []
        for i, g in enumerate(goldens[:limit]):
            question = g.get("input", g.get("query", str(g)))
            expected = g.get("expected_output", g.get("expected", ""))
            rubric = (
                f"The expected answer is: {expected}. Grade how accurately the agent addresses this."
                if expected
                else "Grade how well the agent addresses the question."
            )
            scenarios.append(EvalScenario(
                id=f"confident-ai-{self.dataset_id}-{i}",
                description=f"Confident AI dataset: {self.dataset_id}",
                input_payload={"question": str(question)},
                expected_behavior_rubric=rubric,
            ))
        return scenarios


# ---------------------------------------------------------------------------
# Braintrust loader   (braintrust::<project>/<dataset>)
# ---------------------------------------------------------------------------

class BraintrustLoader(PrivateBenchmarkLoader):
    """
    Load a dataset from Braintrust.

    Requires: BRAINTRUST_API_KEY in .env
    Dataset ref: "<project_id_or_name>/<dataset_name>"  (slash-separated)

    Usage:  --scenario braintrust::my-project/golden-set
    """

    BASE_URL = "https://api.braintrustdata.com/v1"

    def __init__(self, dataset_ref: str, api_key: Optional[str] = None):
        self.dataset_ref = dataset_ref
        self.api_key = api_key or os.getenv("BRAINTRUST_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "BRAINTRUST_API_KEY not set. Add it to your .env file.\n"
                "Get your key from: https://www.braintrustdata.com → Settings"
            )

    def load(self, limit: int = 10) -> List[EvalScenario]:
        # Braintrust uses a dataset UUID for fetch; try treating the ref as UUID first
        url = f"{self.BASE_URL}/dataset/{self.dataset_ref}/fetch"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = httpx.get(url, headers=headers, params={"limit": limit}, timeout=30)
        resp.raise_for_status()
        events = resp.json().get("events", [])
        return self._normalise(events, limit)

    def _normalise(self, events: list, limit: int) -> List[EvalScenario]:
        scenarios = []
        for i, event in enumerate(events[:limit]):
            inp = event.get("input", {})
            question = inp if isinstance(inp, str) else inp.get("question", str(inp))
            expected = event.get("expected", "")
            rubric = (
                f"The expected answer is: {expected}."
                if expected
                else "Grade how well the agent addresses the question."
            )
            scenarios.append(EvalScenario(
                id=f"braintrust-{i}",
                description=f"Braintrust dataset: {self.dataset_ref}",
                input_payload={"question": str(question)},
                expected_behavior_rubric=rubric,
            ))
        return scenarios


# ---------------------------------------------------------------------------
# LangSmith loader   (langsmith::<dataset_id>)
# ---------------------------------------------------------------------------

class LangSmithLoader(PrivateBenchmarkLoader):
    """
    Load examples from a LangSmith dataset.

    Requires: LANGSMITH_API_KEY in .env
    Dataset ID: the UUID or name of a dataset in your LangSmith org.

    Usage:  --scenario langsmith::my-dataset-id
    """

    BASE_URL = "https://api.smith.langchain.com"

    def __init__(self, dataset_id: str, api_key: Optional[str] = None):
        self.dataset_id = dataset_id
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "LANGSMITH_API_KEY not set. Add it to your .env file.\n"
                "Get your key from: https://smith.langchain.com → Settings → API Keys"
            )

    def load(self, limit: int = 10) -> List[EvalScenario]:
        url = f"{self.BASE_URL}/datasets/{self.dataset_id}/examples"
        headers = {"x-api-key": self.api_key}
        resp = httpx.get(url, headers=headers, params={"limit": limit}, timeout=30)
        resp.raise_for_status()
        examples = resp.json()
        if isinstance(examples, dict):
            examples = examples.get("examples", examples.get("data", []))
        return self._normalise(examples, limit)

    def _normalise(self, examples: list, limit: int) -> List[EvalScenario]:
        scenarios = []
        for i, ex in enumerate(examples[:limit]):
            inputs = ex.get("inputs", {})
            outputs = ex.get("outputs", {})
            question = inputs.get("question", inputs.get("input", str(inputs)))
            expected = outputs.get("answer", outputs.get("output", outputs.get("expected", "")))
            rubric = (
                f"The expected answer is: {expected}."
                if expected
                else "Grade how well the agent addresses the question."
            )
            scenarios.append(EvalScenario(
                id=f"langsmith-{self.dataset_id}-{i}",
                description=f"LangSmith dataset: {self.dataset_id}",
                input_payload={"question": str(question)},
                expected_behavior_rubric=rubric,
            ))
        return scenarios


# ---------------------------------------------------------------------------
# Generic REST loader   (configured in evalmonkey.yaml private_benchmarks)
# ---------------------------------------------------------------------------

class GenericRESTLoader(PrivateBenchmarkLoader):
    """
    Load from any REST endpoint that returns a JSON array of eval rows.

    Configuration in evalmonkey.yaml:
        private_benchmarks:
          - id: "my-support-evals"
            name: "Customer Support Golden Set"
            url: "https://my-api.company.com/v1/eval-dataset"
            auth_header: "Authorization: Bearer {MY_API_KEY}"
            input_field: "question"
            expected_field: "ideal_answer"
            category: "Customer Support"

    Any {VAR_NAME} tokens in auth_header are resolved from environment variables.
    """

    def __init__(
        self,
        url: str,
        auth_header: Optional[str] = None,
        input_field: str = "question",
        expected_field: str = "expected_answer",
        name: str = "custom",
    ):
        self.url = url
        self.auth_header = self._resolve_env(auth_header) if auth_header else None
        self.input_field = input_field
        self.expected_field = expected_field
        self.name = name

    @staticmethod
    def _resolve_env(template: str) -> str:
        """Replace {VAR_NAME} tokens with values from the environment."""
        def _replace(m: re.Match) -> str:
            return os.getenv(m.group(1), m.group(0))
        return re.sub(r"\{([A-Z0-9_]+)\}", _replace, template)

    def load(self, limit: int = 10) -> List[EvalScenario]:
        headers = {}
        if self.auth_header:
            key, _, val = self.auth_header.partition(":")
            headers[key.strip()] = val.strip()

        resp = httpx.get(self.url, headers=headers, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, dict):
            rows = rows.get("data", rows.get("items", rows.get("results", [])))

        scenarios = []
        for i, row in enumerate(rows[:limit]):
            question = row.get(self.input_field, str(row))
            expected = row.get(self.expected_field, "")
            rubric = (
                f"The expected answer is: {expected}."
                if expected
                else f"Grade how well the agent addresses: {question}"
            )
            scenarios.append(EvalScenario(
                id=f"{self.name}-{i}",
                description=f"Private dataset: {self.name}",
                input_payload={"question": str(question)},
                expected_behavior_rubric=rubric,
            ))
        return scenarios


# ---------------------------------------------------------------------------
# Top-level factory function
# ---------------------------------------------------------------------------

def load_private_benchmark(
    benchmark_id: str,
    limit: int = 10,
    private_benchmarks_config: Optional[List[Dict]] = None,
) -> List[EvalScenario]:
    """
    Route a benchmark_id to the correct private/external loader.

    Handles these prefixes:
        hf::<org/dataset>           → HuggingFaceLoader
        confident-ai::<dataset_id>  → ConfidentAILoader
        braintrust::<ref>           → BraintrustLoader
        langsmith::<dataset_id>     → LangSmithLoader
        <id>                        → GenericRESTLoader (from private_benchmarks_config)

    Returns an empty list if the id is not recognised (caller falls back to local evals).
    """
    if benchmark_id.startswith("hf::"):
        dataset_id = benchmark_id[4:]
        loader: PrivateBenchmarkLoader = HuggingFaceLoader(dataset_id)

    elif benchmark_id.startswith("confident-ai::"):
        dataset_id = benchmark_id[len("confident-ai::"):]
        loader = ConfidentAILoader(dataset_id)

    elif benchmark_id.startswith("braintrust::"):
        dataset_ref = benchmark_id[len("braintrust::"):]
        loader = BraintrustLoader(dataset_ref)

    elif benchmark_id.startswith("langsmith::"):
        dataset_id = benchmark_id[len("langsmith::"):]
        loader = LangSmithLoader(dataset_id)

    else:
        # Look up in private_benchmarks_config list from evalmonkey.yaml
        cfg_list = private_benchmarks_config or []
        match = next((b for b in cfg_list if b.get("id") == benchmark_id), None)
        if not match:
            return []
        loader = GenericRESTLoader(
            url=match["url"],
            auth_header=match.get("auth_header"),
            input_field=match.get("input_field", "question"),
            expected_field=match.get("expected_field", "expected_answer"),
            name=match.get("name", benchmark_id),
        )

    return loader.load(limit=limit)
