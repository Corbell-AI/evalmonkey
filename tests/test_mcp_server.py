import pytest
import asyncio
from evalmonkey.mcp_server import run_benchmark, run_chaos, get_benchmark_history

@pytest.mark.asyncio
async def test_mcp_run_benchmark(monkeypatch):
    # Mock load_standard_benchmark to avoid huggingface calls
    def mock_load_standard_benchmark(scenario, limit):
        from evalmonkey.evals.local_assets import EvalScenario
        return [EvalScenario(id=scenario, description="test", input_payload={"question":"a"}, expected_behavior_rubric="test")]
    import evalmonkey.mcp_server
    monkeypatch.setattr(evalmonkey.mcp_server, "load_standard_benchmark", mock_load_standard_benchmark)
    
    # Mock generator
    class MockGenerator:
        def __init__(self, *args, **kwargs):
            pass
        async def run_scenario(self, *args, **kwargs):
            return {"status": "success", "data": "mock_data"}
    monkeypatch.setattr(evalmonkey.mcp_server, "LoadGenerator", MockGenerator)
    
    # Mock Judge
    class MockJudge:
        def score_run(self, rubric, agent_output_text):
            return {"score": 99, "reasoning": "mock_reason"}
    monkeypatch.setattr(evalmonkey.mcp_server, "LLMJudgeProvider", MockJudge)
    
    # Mock Record Run
    monkeypatch.setattr(evalmonkey.mcp_server, "record_run", lambda *args, **kwargs: None)
    
    result = await run_benchmark("test_id", "http://test")
    assert "Score: 99/100" in result
    assert "Reasoning: mock_reason" in result

@pytest.mark.asyncio
async def test_mcp_run_chaos(monkeypatch):
    def mock_load_standard_benchmark(scenario, limit):
        from evalmonkey.evals.local_assets import EvalScenario
        return [EvalScenario(id=scenario, description="test", input_payload={"question":"a"}, expected_behavior_rubric="test")]
    import evalmonkey.mcp_server
    monkeypatch.setattr(evalmonkey.mcp_server, "load_standard_benchmark", mock_load_standard_benchmark)
    
    class MockGenerator:
        def __init__(self, *args, **kwargs):
            pass
        async def run_scenario(self, *args, **kwargs):
            assert kwargs.get("chaos_profile") == "test_chaos"
            return {"status": "error", "error_message": "Chaos injected failure!"}
    monkeypatch.setattr(evalmonkey.mcp_server, "LoadGenerator", MockGenerator)
    
    class MockJudge:
        def score_run(self, rubric, agent_output_text):
            assert "Chaos injected failure!" in agent_output_text
            return {"score": 10, "reasoning": "failed"}
    monkeypatch.setattr(evalmonkey.mcp_server, "LLMJudgeProvider", MockJudge)
    
    monkeypatch.setattr(evalmonkey.mcp_server, "record_run", lambda *args, **kwargs: None)
    
    result = await run_chaos("test_id", "http://test", chaos_profile="test_chaos")
    assert "Score: 10/100" in result
    assert "Chaos Profile: test_chaos" in result

def test_mcp_get_history(monkeypatch):
    import evalmonkey.mcp_server
    monkeypatch.setattr(evalmonkey.mcp_server, "get_history", lambda s: [{"timestamp": "2026", "run_type": "BASELINE", "score": 85}])
    result = get_benchmark_history("test_id")
    assert "2026" in result
    assert "BASELINE" in result
    assert "85/100" in result
