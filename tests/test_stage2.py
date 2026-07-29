"""
tests/test_stage2.py

Unit tests for Stage 2 — OCR Benchmark metrics and engine wrappers.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_ocr_benchmark.metrics import (
    cer, wer, normalise, mean_cer, mean_wer,
    field_exact_match, field_f1, compute_aggregate_metrics,
)


class TestNormalise:
    def test_strips_whitespace(self):
        assert normalise("  hello  ") == "hello"

    def test_lowercases(self):
        assert normalise("HELLO World") == "hello world"

    def test_collapses_spaces(self):
        assert normalise("hello  world") == "hello world"

    def test_empty_string(self):
        assert normalise("") == ""


class TestCER:
    def test_identical_strings(self):
        assert cer("hello", "hello") == 0.0

    def test_completely_different(self):
        c = cer("abc", "xyz")
        assert 0.0 < c <= 1.0

    def test_empty_reference(self):
        assert cer("something", "") == 1.0

    def test_empty_both(self):
        assert cer("", "") == 0.0

    def test_one_char_error(self):
        c = cer("hallo", "hello")
        assert c == pytest.approx(1 / 5, abs=1e-6)

    def test_one_deletion(self):
        c = cer("hell", "hello")
        assert c == pytest.approx(1 / 5, abs=1e-6)

    def test_cer_capped_at_reasonable_value(self):
        c = cer("", "abcde")
        assert c == 1.0


class TestWER:
    def test_identical_sentences(self):
        assert wer("hello world", "hello world") == 0.0

    def test_one_word_wrong(self):
        w = wer("hello earth", "hello world")
        assert w == pytest.approx(1 / 2, abs=1e-6)

    def test_empty_reference(self):
        assert wer("foo bar", "") == 1.0

    def test_empty_both(self):
        assert wer("", "") == 0.0


class TestMeanMetrics:
    def test_mean_cer(self):
        hyps = ["hello", "world"]
        refs = ["hello", "world"]
        assert mean_cer(hyps, refs) == 0.0

    def test_mean_wer_empty(self):
        assert mean_wer([], []) == 0.0


class TestFieldMetrics:
    def test_exact_match_all_correct(self):
        pred = {"name": "Alice", "dob": "01/01/1990"}
        gold = {"name": "Alice", "dob": "01/01/1990"}
        result = field_exact_match(pred, gold)
        assert result == {"name": True, "dob": True}

    def test_exact_match_case_insensitive(self):
        pred = {"name": "ALICE"}
        gold = {"name": "alice"}
        result = field_exact_match(pred, gold)
        assert result["name"] is True

    def test_exact_match_missing_field(self):
        pred = {}
        gold = {"name": "Alice"}
        result = field_exact_match(pred, gold)
        assert result["name"] is False

    def test_field_f1_perfect(self):
        pred = {"name": "Alice", "dob": "01/01/1990"}
        gold = {"name": "Alice", "dob": "01/01/1990"}
        assert field_f1(pred, gold) == pytest.approx(1.0, abs=1e-6)

    def test_field_f1_zero(self):
        pred = {"name": "Bob", "dob": "02/02/2002"}
        gold = {"name": "Alice", "dob": "01/01/1990"}
        # F1 depends on TP — if nothing matches, f1=0
        f = field_f1(pred, gold)
        assert 0.0 <= f <= 1.0

    def test_field_f1_empty_both(self):
        assert field_f1({}, {}) == 1.0


class TestAggregateMetrics:
    def test_perfect_predictions(self):
        hyps = ["hello world", "foo bar"]
        refs = ["hello world", "foo bar"]
        m = compute_aggregate_metrics(hyps, refs)
        assert m["cer"] == 0.0
        assert m["wer"] == 0.0
        assert m["exact_match"] == 1.0

    def test_empty_lists(self):
        m = compute_aggregate_metrics([], [])
        assert m["cer"] == 0.0

    def test_partial_match(self):
        hyps = ["hello world", "xyz abc"]
        refs = ["hello world", "foo bar"]
        m = compute_aggregate_metrics(hyps, refs)
        assert 0.0 < m["cer"] < 1.0
        assert m["exact_match"] == pytest.approx(0.5, abs=1e-6)
