import pytest
import asyncio
from unittest.mock import patch, MagicMock

from evalmonkey.simulator.load_gen import LoadGenerator
from evalmonkey.evals.runner import LLMJudgeProvider
from evalmonkey.reporting.markdown import print_benchmark_score
from typer.testing import CliRunner
from scripts.cli import app

runner = CliRunner()

# ----------- TEST LLM JUDGE RUNNER (MOCKING LITELLM) -----------

@patch("evalmonkey.evals.runner.call_llm")
def test_llm_judge_scoring(mock_litellm):
    # Mock Litellm to return a fake JSON parsing payload
    mock_msg = MagicMock()
    mock_msg.message.content = '{"score": 90, "reasoning": "Excellent."}'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg.message
    mock_litellm.return_value = MagicMock(choices=[mock_choice])

    judge = LLMJudgeProvider()
    result = judge.score_run("Must do X", "X was done.")
    
    assert result["score"] == 90
    assert result["reasoning"] == "Excellent."


@patch("evalmonkey.evals.runner.call_llm")
def test_llm_judge_malformed(mock_litellm):
    # Mock a completely broken LLM response
    mock_msg = MagicMock()
    mock_msg.message.content = 'Not valid JSON at all.'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg.message
    mock_litellm.return_value = MagicMock(choices=[mock_choice])

    judge = LLMJudgeProvider()
    result = judge.score_run("Must do X", "X was done.")
    
    # Should fallback gracefully
    assert result["score"] == 0
    assert "Evaluation failed due to error" in result["reasoning"]


# ----------- TEST LOAD GENERATOR -----------

@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_load_generator_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "42"}
    mock_post.return_value = mock_response

    generator = LoadGenerator("http://fake.api/solve")
    res = await generator.run_scenario("gsm8k", {"question": "life?"})
    
    assert res["status"] == "success"
    assert res["data"] == "42"


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_load_generator_timeout(mock_post):
    import httpx
    mock_post.side_effect = httpx.TimeoutException("Timeout")

    generator = LoadGenerator("http://fake.api/solve")
    res = await generator.run_scenario("gsm8k", {"question": "life?"})
    
    assert res["status"] == "error"
    assert "Timeout" in res["error_message"]


# ----------- TEST CLI -----------

@patch("scripts.cli.get_supported_benchmarks")
def test_cli_list_benchmarks(mock_get_benchmarks):
    mock_get_benchmarks.return_value = {"test_scen": "Just a test"}
    result = runner.invoke(app, ["list-benchmarks"])
    assert result.exit_code == 0
    assert "test_scen" in result.stdout
    assert "EvalMonkey Natively Supported Benchmarks" in result.stdout


def test_cli_no_target_url():
    # If no target URL and no sample agent, it should complain.
    result = runner.invoke(app, ["run-benchmark", "--scenario", "gsm8k"])
    assert "No agent URL found" in result.stdout

