"""
tests/test_metrics.py

Integration tests for the shared metrics utilities
and the utils.io / utils.seed modules.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.seed import set_global_seed
from utils.io import save_json, load_json, append_jsonl, load_jsonl


class TestSeed:
    def test_set_seed_no_torch(self):
        """set_global_seed should not crash even without torch installed."""
        set_global_seed(42)   # Should not raise

    def test_seed_reproducibility(self):
        import random
        set_global_seed(42)
        a = random.random()
        set_global_seed(42)
        b = random.random()
        assert a == b


class TestIO:
    def test_save_and_load_json(self, tmp_path):
        data = {"name": "Alice", "scores": [1, 2, 3]}
        p = tmp_path / "test.json"
        save_json(data, p)
        loaded = load_json(p)
        assert loaded == data

    def test_save_json_creates_parents(self, tmp_path):
        data = {"key": "value"}
        deep = tmp_path / "a" / "b" / "c" / "test.json"
        save_json(data, deep)
        assert deep.exists()
        assert load_json(deep) == data

    def test_append_jsonl(self, tmp_path):
        p = tmp_path / "test.jsonl"
        records = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
        for rec in records:
            append_jsonl(rec, p)
        loaded = list(load_jsonl(p))
        assert loaded == records

    def test_jsonl_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        loaded = list(load_jsonl(p))
        assert loaded == []

    def test_save_json_with_non_serialisable(self, tmp_path):
        """Non-serialisable objects should be converted via default=str."""
        from datetime import date
        data = {"date": date(2024, 1, 1)}
        p = tmp_path / "test.json"
        save_json(data, p)   # Should not raise
        loaded = load_json(p)
        assert "date" in loaded


class TestPromptsModule:
    def test_build_extraction_prompt_no_type(self):
        from stage3_vlm_sft.prompts import build_extraction_prompt
        prompt = build_extraction_prompt()
        assert "JSON" in prompt
        assert len(prompt) > 10

    def test_build_extraction_prompt_with_type(self):
        from stage3_vlm_sft.prompts import build_extraction_prompt
        prompt = build_extraction_prompt("aadhaar")
        assert "aadhaar" in prompt.lower() or "uid" in prompt.lower()

    def test_build_extraction_response_valid_json(self):
        from stage3_vlm_sft.prompts import build_extraction_response
        fields = {"name": "Alice", "dob": "01/01/1990"}
        resp   = build_extraction_response(fields, "aadhaar")
        parsed = json.loads(resp)
        assert parsed["name"] == "Alice"
        assert parsed["card_type"] == "aadhaar"

    def test_build_chosen_response(self):
        from stage3_vlm_sft.prompts import build_chosen_response
        fields = {"name": "Alice"}
        chosen = build_chosen_response(fields, "aadhaar", confidence="high")
        assert "HIGH" in chosen
        assert "Alice" in chosen

    def test_build_rejected_response_corrupts_fields(self):
        from stage3_vlm_sft.prompts import build_rejected_response
        fields = {"name": "Alice Smith", "id_number": "1234-5678-9012"}
        rejected = build_rejected_response(fields, "aadhaar", "overconfident_wrong")
        # The rejected response should still contain the card_type key
        assert "aadhaar" in rejected
        # And should claim HIGH confidence
        assert "HIGH" in rejected

    def test_build_rejected_hallucinated_fields(self):
        from stage3_vlm_sft.prompts import build_rejected_response
        fields = {"name": "Bob"}
        rejected = build_rejected_response(fields, "driving_licence", "hallucinated_fields")
        parsed_part = rejected.split("\n\n")[0].strip()
        parsed = json.loads(parsed_part)
        # Should have extra hallucinated keys
        assert "passport_number" in parsed or "visa_status" in parsed
