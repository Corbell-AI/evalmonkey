"""
Tests for coding agent benchmarks and chaos profiles.
"""
import pytest
from unittest.mock import patch, MagicMock


# ── Category Filtering ──────────────────────────────────────────────────────

def test_get_benchmarks_by_category_coding():
    from evalmonkey.scenarios.standard_benchmarks import get_benchmarks_by_category
    coding = get_benchmarks_by_category("Coding")
    assert len(coding) > 0
    assert "human-eval" in coding
    assert "mbpp" in coding
    assert "apps" in coding
    assert "swe-bench" in coding
    # Non-coding should not appear
    assert "gsm8k" not in coding
    assert "mmlu" not in coding


def test_get_benchmarks_by_category_case_insensitive():
    from evalmonkey.scenarios.standard_benchmarks import get_benchmarks_by_category
    lower = get_benchmarks_by_category("coding")
    upper = get_benchmarks_by_category("CODING")
    mixed = get_benchmarks_by_category("Coding")
    assert set(lower.keys()) == set(upper.keys()) == set(mixed.keys())


def test_get_benchmarks_by_category_reasoning():
    from evalmonkey.scenarios.standard_benchmarks import get_benchmarks_by_category
    reasoning = get_benchmarks_by_category("Reasoning")
    assert "gsm8k" in reasoning
    assert "arc" in reasoning
    assert "bbh" in reasoning
    assert "hella-swag" in reasoning


def test_get_benchmarks_by_category_unknown_returns_empty():
    from evalmonkey.scenarios.standard_benchmarks import get_benchmarks_by_category
    result = get_benchmarks_by_category("NonExistentCategory")
    assert result == {}


def test_coding_benchmarks_in_supported():
    from evalmonkey.scenarios.standard_benchmarks import SUPPORTED_BENCHMARKS
    for bid in ["human-eval", "mbpp", "apps", "swe-bench"]:
        assert bid in SUPPORTED_BENCHMARKS
        assert SUPPORTED_BENCHMARKS[bid]["agent_category"] == "Coding"


def test_catalogue_has_22_benchmarks():
    """Ensure the total count is 22."""
    from evalmonkey.scenarios.standard_benchmarks import get_supported_benchmarks
    cat = get_supported_benchmarks()
    assert len(cat) == 22


# ── Coding Chaos Profiles ───────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_code_context_strip(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "def foo(): pass"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    # Payload with code block and function def
    res = await gen.run_scenario(
        "human-eval_0",
        {"question": "Complete the following Python function:\n\n```python\ndef add(a, b):\n```"},
        chaos_profile="code_context_strip",
    )
    assert res["status"] == "success"
    # Verify code was stripped from the sent payload
    sent_json = mock_post.call_args[1]["json"]
    assert "```" not in sent_json.get("question", "")


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_code_wrong_language(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "function foo() {}"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "mbpp_0",
        {"question": "Write a Python function to sort a list"},
        chaos_profile="code_wrong_language",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    assert "JavaScript" in sent_json.get("question", "")


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_code_syntax_break(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "human-eval_1",
        {"question": "def add(a, b):\n    return a + b"},
        chaos_profile="code_syntax_break",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    q = sent_json.get("question", "")
    assert "deff " in q or "returnn " in q  # at least one keyword was broken


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_code_test_poison(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "mbpp_1",
        {"question": "Write a function that adds two numbers"},
        chaos_profile="code_test_poison",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    q = sent_json.get("question", "")
    assert "assert result == None" in q


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_code_incomplete_signature(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    original_q = "Write a Python function that sorts a list of integers in ascending order using bubble sort algorithm"
    res = await gen.run_scenario(
        "apps_0",
        {"question": original_q},
        chaos_profile="code_incomplete_signature",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    q = sent_json.get("question", "")
    # Should be truncated and include the truncation marker
    assert "SPECIFICATION TRUNCATED" in q
    assert len(q) < len(original_q) + 100  # + marker length


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_code_conflicting_constraints(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "human-eval_2",
        {"question": "Write a function to find max of list"},
        chaos_profile="code_conflicting_constraints",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    q = sent_json.get("question", "")
    assert "MUST NOT use any loops" in q
    assert "O(1)" in q
    assert "O(n)" in q


# ── CLI list-benchmarks with --category ────────────────────────────────────

from typer.testing import CliRunner
from scripts.cli import app

runner = CliRunner()


def test_cli_list_benchmarks_all():
    result = runner.invoke(app, ["list-benchmarks"])
    assert result.exit_code == 0
    assert "human-eval" in result.stdout
    assert "mbpp" in result.stdout
    assert "gsm8k" in result.stdout
    # Category column should now show
    assert "Coding" in result.stdout
    assert "Reasoning" in result.stdout


def test_cli_list_benchmarks_category_coding():
    result = runner.invoke(app, ["list-benchmarks", "--category", "Coding"])
    assert result.exit_code == 0
    assert "human-eval" in result.stdout
    assert "mbpp" in result.stdout
    assert "apps" in result.stdout
    assert "swe-bench" in result.stdout
    # Non-coding should NOT appear
    assert "gsm8k" not in result.stdout


def test_cli_list_benchmarks_category_unknown():
    result = runner.invoke(app, ["list-benchmarks", "--category", "WizardMagic"])
    assert result.exit_code == 0
    assert "No benchmarks found for category" in result.stdout


def test_cli_list_benchmarks_category_reasoning():
    result = runner.invoke(app, ["list-benchmarks", "--category", "Reasoning"])
    assert result.exit_code == 0
    assert "gsm8k" in result.stdout
    # Coding should not appear
    assert "human-eval" not in result.stdout


# ── Backend API category filter ─────────────────────────────────────────────

def test_backend_list_benchmarks_no_filter():
    from ui.backend.main import list_benchmarks
    result = list_benchmarks(category=None)
    ids = [b.id for b in result]
    assert "human-eval" in ids
    assert "gsm8k" in ids
    assert len(ids) == 22


def test_backend_list_benchmarks_coding_filter():
    from ui.backend.main import list_benchmarks
    result = list_benchmarks(category="Coding")
    ids = [b.id for b in result]
    assert "human-eval" in ids
    assert "mbpp" in ids
    assert "swe-bench" in ids
    assert "apps" in ids
    # Non-coding should not appear
    assert "gsm8k" not in ids
    for b in result:
        assert b.category == "Coding"


def test_backend_list_benchmarks_case_insensitive():
    from ui.backend.main import list_benchmarks
    upper = list_benchmarks(category="CODING")
    lower = list_benchmarks(category="coding")
    mixed = list_benchmarks(category="Coding")
    assert {b.id for b in upper} == {b.id for b in lower} == {b.id for b in mixed}


def test_backend_list_benchmarks_unknown_category_returns_empty():
    from ui.backend.main import list_benchmarks
    result = list_benchmarks(category="UnknownXYZ")
    assert result == []


# ── Dedicated coding loader rubric quality ──────────────────────────────────

def test_humaneval_loader_builds_coding_rubric():
    """Verify the humaneval loader produces code-specific rubrics (not generic Q&A)."""
    from unittest.mock import patch, MagicMock
    mock_item = {
        "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n",
        "canonical_solution": "    return a + b\n",
        "entry_point": "add",
        "test": "assert add(1, 2) == 3",
    }
    mock_dataset = [mock_item]

    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter(mock_dataset)
        from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
        scenarios = load_standard_benchmark("human-eval", limit=1)

    assert len(scenarios) == 1
    s = scenarios[0]
    assert s.id == "human-eval_0"
    assert "Complete the following Python function" in s.input_payload["question"]
    assert "add" in s.expected_behavior_rubric
    assert "syntactically correct" in s.expected_behavior_rubric.lower() or "valid Python" in s.expected_behavior_rubric


def test_mbpp_loader_includes_test_cases_in_rubric():
    """Verify mbpp loader embeds test assertions in rubric."""
    mock_item = {
        "text": "Write a function to find the sum of a list",
        "test_list": ["assert sum_list([1, 2, 3]) == 6", "assert sum_list([]) == 0"],
        "code": "def sum_list(lst): return sum(lst)",
    }
    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter([mock_item])
        from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
        scenarios = load_standard_benchmark("mbpp", limit=1)

    assert len(scenarios) == 1
    s = scenarios[0]
    assert "sum_list" in s.input_payload["question"]
    # Rubric must contain the test assertions
    assert "sum_list([1, 2, 3]) == 6" in s.expected_behavior_rubric


def test_apps_loader_produces_code_rubric():
    """Verify apps loader produces code-correctness rubric."""
    mock_item = {
        "question": "Given N integers, find the maximum.",
        "solutions": '["def solve():\\n    n = int(input())\\n    print(max(map(int, input().split())))"]',
        "input_output": "{}",
    }
    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter([mock_item])
        from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
        scenarios = load_standard_benchmark("apps", limit=1)

    assert len(scenarios) == 1
    s = scenarios[0]
    assert "executable Python code" in s.expected_behavior_rubric


def test_swebench_loader_produces_patch_rubric():
    """Verify swe-bench loader embeds repo and patch context in rubric."""
    mock_item = {
        "problem_statement": "Fix the off-by-one error in parser.py",
        "repo": "psf/requests",
        "patch": "--- a/parser.py\n+++ b/parser.py\n@@ -10 +10 @@\n-    idx = n\n+    idx = n - 1",
    }
    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter([mock_item])
        from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
        scenarios = load_standard_benchmark("swe-bench", limit=1)

    assert len(scenarios) == 1
    s = scenarios[0]
    assert "psf/requests" in s.input_payload["question"]
    assert "psf/requests" in s.expected_behavior_rubric
    assert "patch" in s.expected_behavior_rubric.lower()
