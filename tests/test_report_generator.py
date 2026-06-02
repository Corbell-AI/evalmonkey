"""Tests for report_generator.py (Agent Card)."""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from evalmonkey.reporting.report_generator import generate_report, _badge_url, _badge_color


# ---------------------------------------------------------------------------
# Badge helpers
# ---------------------------------------------------------------------------

class TestBadgeHelpers:
    def test_badge_color_green_above_80(self):
        assert _badge_color(80) == "brightgreen"
        assert _badge_color(100) == "brightgreen"

    def test_badge_color_yellow_60_to_79(self):
        assert _badge_color(60) == "yellow"
        assert _badge_color(79) == "yellow"

    def test_badge_color_red_below_60(self):
        assert _badge_color(59) == "red"
        assert _badge_color(0) == "red"

    def test_badge_url_contains_score(self):
        url = _badge_url(75)
        assert "75" in url
        assert "shields.io" in url
        assert "EvalMonkey" in url

    def test_badge_url_has_correct_color(self):
        green_url = _badge_url(85)
        assert "brightgreen" in green_url
        yellow_url = _badge_url(65)
        assert "yellow" in yellow_url
        red_url = _badge_url(40)
        assert "red" in red_url


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def _make_history(records: list, tmp_path) -> str:
    """Write a fake history.json and return its path."""
    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps(records))
    return str(history_file)


def _record(scenario, score, run_type="baseline"):
    return {
        "scenario": scenario,
        "run_type": run_type,
        "score": score,
        "timestamp": "2025-01-01T09:00:00",
        "details": {},
    }


class TestGenerateReport:
    def test_creates_output_file(self, tmp_path):
        history_path = _make_history([_record("gsm8k", 80)], tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path, agent_name="Test Agent")
        assert os.path.exists(output_path)

    def test_report_contains_agent_name(self, tmp_path):
        history_path = _make_history([_record("gsm8k", 80)], tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path, agent_name="My Research Bot")
        assert "My Research Bot" in content

    def test_report_contains_scenario_name(self, tmp_path):
        history_path = _make_history([_record("gsm8k", 82)], tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path)
        assert "gsm8k" in content

    def test_report_contains_baseline_score(self, tmp_path):
        history_path = _make_history([_record("gsm8k", 82)], tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path)
        assert "82" in content

    def test_report_contains_shields_badge(self, tmp_path):
        history_path = _make_history([_record("gsm8k", 82)], tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path)
        assert "shields.io" in content
        assert "EvalMonkey" in content

    def test_report_with_baseline_and_chaos(self, tmp_path):
        records = [
            _record("gsm8k", 82, "baseline"),
            _record("gsm8k", 65, "chaos"),
        ]
        history_path = _make_history(records, tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path)
        assert "82" in content
        assert "65" in content

    def test_report_empty_history(self, tmp_path):
        history_path = _make_history([], tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path)
        # Should still produce a valid file with placeholder row
        assert "no runs recorded" in content

    def test_report_multiple_scenarios(self, tmp_path):
        records = [
            _record("gsm8k", 82, "baseline"),
            _record("mmlu", 75, "baseline"),
            _record("arc", 90, "baseline"),
        ]
        history_path = _make_history(records, tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path)
        assert "gsm8k" in content
        assert "mmlu" in content
        assert "arc" in content

    def test_report_includes_production_reliability_column(self, tmp_path):
        records = [
            _record("gsm8k", 80, "baseline"),
            _record("gsm8k", 60, "chaos"),
        ]
        history_path = _make_history(records, tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path)
        assert "Production Reliability" in content

    def test_report_includes_badge_markdown_snippet(self, tmp_path):
        history_path = _make_history([_record("gsm8k", 80)], tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            content = generate_report(output_path=output_path)
        # Should include the copy-paste badge snippet
        assert "```markdown" in content
        assert "[![EvalMonkey" in content

    def test_report_returns_string_content(self, tmp_path):
        history_path = _make_history([_record("gsm8k", 80)], tmp_path)
        output_path = str(tmp_path / "report.md")
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = generate_report(output_path=output_path)
        assert isinstance(result, str)
        assert len(result) > 0
