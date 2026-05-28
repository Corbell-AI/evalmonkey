"""Tests for detect_regression() and the guard command logic."""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from evalmonkey.reporting.history import detect_regression, record_run, get_history


# ---------------------------------------------------------------------------
# detect_regression unit tests
# ---------------------------------------------------------------------------

def _history_file_with(records: list, tmp_path):
    """Write a fake history.json and return its path."""
    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps(records))
    return str(history_file)


class TestDetectRegression:
    def _make_record(self, scenario, score, run_type="baseline", ts="2025-01-01T00:00:00"):
        return {"scenario": scenario, "run_type": run_type, "score": score, "timestamp": ts}

    def test_returns_none_when_only_one_baseline(self, tmp_path):
        records = [self._make_record("gsm8k", 80)]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("gsm8k", 80, threshold=5)
        assert result is None

    def test_returns_none_when_no_history(self, tmp_path):
        history_path = _history_file_with([], tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("gsm8k", 75, threshold=5)
        assert result is None

    def test_detects_regression_above_threshold(self, tmp_path):
        records = [
            self._make_record("gsm8k", 82, ts="2025-01-01T00:00:00"),
            self._make_record("gsm8k", 60, ts="2025-01-02T00:00:00"),  # current (already recorded)
        ]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("gsm8k", 60, threshold=5)
        assert result is not None
        assert result["prev_score"] == 82
        assert result["current_score"] == 60
        assert result["drop"] == 22
        assert result["scenario"] == "gsm8k"

    def test_no_regression_when_drop_below_threshold(self, tmp_path):
        records = [
            self._make_record("gsm8k", 80, ts="2025-01-01T00:00:00"),
            self._make_record("gsm8k", 77, ts="2025-01-02T00:00:00"),  # drop of 3, threshold 5
        ]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("gsm8k", 77, threshold=5)
        assert result is None

    def test_no_regression_when_score_improved(self, tmp_path):
        records = [
            self._make_record("gsm8k", 70, ts="2025-01-01T00:00:00"),
            self._make_record("gsm8k", 85, ts="2025-01-02T00:00:00"),
        ]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("gsm8k", 85, threshold=5)
        assert result is None

    def test_regression_at_exact_threshold(self, tmp_path):
        """A drop exactly equal to threshold should trigger."""
        records = [
            self._make_record("mmlu", 75, ts="2025-01-01T00:00:00"),
            self._make_record("mmlu", 70, ts="2025-01-02T00:00:00"),  # drop = 5 = threshold
        ]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("mmlu", 70, threshold=5)
        assert result is not None
        assert result["drop"] == 5

    def test_ignores_chaos_runs_when_comparing_baselines(self, tmp_path):
        """Chaos run records should not affect the baseline regression comparison."""
        records = [
            self._make_record("gsm8k", 82, run_type="baseline", ts="2025-01-01T00:00:00"),
            self._make_record("gsm8k", 45, run_type="chaos", ts="2025-01-01T12:00:00"),
            self._make_record("gsm8k", 60, run_type="baseline", ts="2025-01-02T00:00:00"),
        ]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("gsm8k", 60, threshold=5)
        # Should compare baseline 82 vs 60, NOT chaos 45
        assert result is not None
        assert result["prev_score"] == 82
        assert result["drop"] == 22

    def test_scenario_isolation(self, tmp_path):
        """Regression for one scenario should not bleed into another."""
        records = [
            self._make_record("gsm8k", 90, ts="2025-01-01T00:00:00"),
            self._make_record("gsm8k", 50, ts="2025-01-02T00:00:00"),
            self._make_record("mmlu", 70, ts="2025-01-01T00:00:00"),
            self._make_record("mmlu", 72, ts="2025-01-02T00:00:00"),
        ]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            mmlu_result = detect_regression("mmlu", 72, threshold=5)
        assert mmlu_result is None  # mmlu improved, should not regress

    def test_custom_threshold_zero(self, tmp_path):
        """Threshold of 0 means any drop at all triggers regression."""
        records = [
            self._make_record("arc", 80, ts="2025-01-01T00:00:00"),
            self._make_record("arc", 79, ts="2025-01-02T00:00:00"),
        ]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("arc", 79, threshold=0)
        assert result is not None
        assert result["drop"] == 1

    def test_with_many_baselines_compares_last_two(self, tmp_path):
        """With 5 baselines, should compare the 5th vs 4th."""
        records = [
            self._make_record("truthfulqa", 60, ts="2025-01-01T00:00:00"),
            self._make_record("truthfulqa", 65, ts="2025-01-02T00:00:00"),
            self._make_record("truthfulqa", 70, ts="2025-01-03T00:00:00"),
            self._make_record("truthfulqa", 80, ts="2025-01-04T00:00:00"),
            self._make_record("truthfulqa", 50, ts="2025-01-05T00:00:00"),  # big drop
        ]
        history_path = _history_file_with(records, tmp_path)
        with patch("evalmonkey.reporting.history.HISTORY_FILE", history_path):
            result = detect_regression("truthfulqa", 50, threshold=5)
        assert result is not None
        assert result["prev_score"] == 80
        assert result["drop"] == 30


# ---------------------------------------------------------------------------
# print_regression_warning smoke test (just checks it doesn't raise)
# ---------------------------------------------------------------------------

class TestPrintRegressionWarning:
    def test_no_exception_raised(self):
        from evalmonkey.reporting.markdown import print_regression_warning
        # Should render without any exception
        print_regression_warning("gsm8k", prev_score=82, curr_score=60, drop=22)
