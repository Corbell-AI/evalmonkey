"""
Tests for Voice Agent benchmarks and chaos profiles.
"""
import pytest
from unittest.mock import patch, MagicMock


# ── Category Filtering ──────────────────────────────────────────────────────

def test_get_benchmarks_by_category_voice():
    from evalmonkey.scenarios.standard_benchmarks import get_benchmarks_by_category
    voice = get_benchmarks_by_category("Voice")
    assert len(voice) == 3
    assert "daily-dialog" in voice
    assert "multiwoz" in voice
    assert "spokentext-cleanup" in voice


def test_voice_benchmarks_in_supported():
    from evalmonkey.scenarios.standard_benchmarks import SUPPORTED_BENCHMARKS
    for bid in ["daily-dialog", "multiwoz", "spokentext-cleanup"]:
        assert bid in SUPPORTED_BENCHMARKS
        assert SUPPORTED_BENCHMARKS[bid]["agent_category"] == "Voice"


# ── Voice Benchmark Loaders (Mocked) ────────────────────────────────────────

def test_load_standard_benchmark_daily_dialog_mocked():
    from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
    mock_item = {
        "dialog": ["hello there", "how can I help you?", "I need a flight"]
    }
    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter([mock_item])
        scenarios = load_standard_benchmark("daily-dialog", limit=1)

    assert isinstance(scenarios, list)
    assert len(scenarios) == 1
    assert "hello there" in scenarios[0].input_payload["question"]
    assert "how can I help you?" in scenarios[0].input_payload["question"]
    assert "I need a flight" in scenarios[0].expected_behavior_rubric


def test_load_standard_benchmark_multiwoz_mocked():
    from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
    mock_item = {
        "turns": {
            "speaker": ["USER", "SYSTEM", "USER"],
            "utterance": ["hi", "hello, how are you?", "can you find a restaurant?"]
        }
    }
    with patch("datasets.load_dataset") as mock_ld:
        mock_ld.return_value = iter([mock_item])
        scenarios = load_standard_benchmark("multiwoz", limit=1)

    assert isinstance(scenarios, list)
    assert len(scenarios) == 1
    assert "User: hi" in scenarios[0].input_payload["question"]
    assert "Assistant: hello, how are you?" in scenarios[0].input_payload["question"]
    assert "can you find a restaurant?" in scenarios[0].expected_behavior_rubric


def test_load_standard_benchmark_spokentext_cleanup():
    from evalmonkey.scenarios.standard_benchmarks import load_standard_benchmark
    scenarios = load_standard_benchmark("spokentext-cleanup", limit=2)
    assert isinstance(scenarios, list)
    assert len(scenarios) == 2
    assert "Please clean up this spoken transcription" in scenarios[0].input_payload["question"]
    assert "Agent MUST clean" in scenarios[0].expected_behavior_rubric


# ── Voice Chaos Profiles ────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_voice_asr_noise(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "Ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "daily-dialog_0",
        {"question": "Is weather there nice today? Please write to see if you can accept."},
        chaos_profile="voice_asr_noise",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    sent_question = sent_json.get("question", "")
    # Check lowercased and no punctuation
    assert sent_question.islower()
    assert "?" not in sent_question
    # Check homophone replacement: weather -> whether, there -> their, write -> right, see -> sea, accept -> except
    assert "whether" in sent_question
    assert "their" in sent_question
    assert "right" in sent_question
    assert "sea" in sent_question
    assert "except" in sent_question


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_voice_filler_words(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "Ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "daily-dialog_0",
        {"question": "book a taxi to the station"},
        chaos_profile="voice_filler_words",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    sent_question = sent_json.get("question", "")
    assert "uh," in sent_question or "um," in sent_question
    assert "like," in sent_question or "you know," in sent_question


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_voice_background_noise_sim(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "Ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "daily-dialog_0",
        {"question": "book a taxi"},
        chaos_profile="voice_background_noise_sim",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    sent_question = sent_json.get("question", "")
    assert "[background chatter]" in sent_question
    assert "[static]" in sent_question
    assert "[dog barking]" in sent_question


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_voice_truncated_speech(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "Ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "daily-dialog_0",
        {"question": "I would like to book a flight to Paris right now"},
        chaos_profile="voice_truncated_speech",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    sent_question = sent_json.get("question", "")
    assert "[audio cut off / silence]" in sent_question


@pytest.mark.asyncio
@patch("evalmonkey.simulator.load_gen.httpx.AsyncClient.post")
async def test_chaos_voice_dialect_shift(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value={"data": "Ok"})
    )
    from evalmonkey.simulator.load_gen import LoadGenerator
    gen = LoadGenerator("http://fake/solve")
    res = await gen.run_scenario(
        "daily-dialog_0",
        {"question": "Yes, I want to see you all. Let me know when ok."},
        chaos_profile="voice_dialect_shift",
    )
    assert res["status"] == "success"
    sent_json = mock_post.call_args[1]["json"]
    sent_question = sent_json.get("question", "")
    assert "yeah" in sent_question.lower()
    assert "wanna" in sent_question.lower()
    assert "y'all" in sent_question.lower()
    assert "lemme" in sent_question.lower()
    assert "uh-huh" in sent_question.lower()


# ── Backend API category filter ─────────────────────────────────────────────

def test_backend_list_benchmarks_voice_filter():
    from ui.backend.main import list_benchmarks
    result = list_benchmarks(category="Voice")
    ids = [b.id for b in result]
    assert len(ids) == 3
    assert "daily-dialog" in ids
    assert "multiwoz" in ids
    assert "spokentext-cleanup" in ids
