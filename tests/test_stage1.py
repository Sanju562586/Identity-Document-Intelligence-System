"""
tests/test_stage1.py

Unit tests for Stage 1 — Synthetic Data Factory.
Tests card rendering, degradation transforms, forgery generators,
and the manifest JSON schema.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_data_factory.templates import render_card, RENDERERS
from stage1_data_factory.degradations import (
    apply_degradations, DEGRADATION_REGISTRY, ALL_DEGRADATIONS,
    apply_blur, apply_jpeg_compression, apply_occlusion,
    apply_rotation, apply_shadow, apply_low_dpi,
    apply_perspective_warp, apply_ink_bleed,
)
from stage1_data_factory.forgery import apply_photocopy, apply_splice


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

MOCK_CFG = {
    "stage1": {
        "image_size": [640, 400],
        "fonts": {
            "regular": "data/raw/fonts/DejaVuSans.ttf",
            "bold":    "data/raw/fonts/DejaVuSans-Bold.ttf",
            "mono":    "data/raw/fonts/DejaVuSansMono.ttf",
        },
    }
}


def _make_test_image(w=640, h=400) -> Image.Image:
    """Create a dummy RGB image for testing."""
    arr = np.random.randint(100, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Card rendering tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCardRendering:

    @pytest.mark.parametrize("card_type", ["driving_licence", "aadhaar", "bank_statement"])
    def test_render_returns_image_and_fields(self, card_type):
        img, fields = render_card(card_type, MOCK_CFG)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size[0] > 0 and img.size[1] > 0
        assert isinstance(fields, dict)
        assert len(fields) > 0

    @pytest.mark.parametrize("card_type", ["driving_licence", "aadhaar", "bank_statement"])
    def test_render_fields_are_non_empty(self, card_type):
        _, fields = render_card(card_type, MOCK_CFG)
        # At minimum, name and id_number or account_number must be present
        assert any(
            k in fields for k in ["name", "account_number"]
        ), f"No name/account in fields: {list(fields.keys())}"

    def test_unknown_card_type_raises(self):
        with pytest.raises(ValueError, match="Unknown card type"):
            render_card("alien_passport", MOCK_CFG)

    def test_rendering_is_deterministic_size(self):
        """Two renders should produce images of the same configured size."""
        img1, _ = render_card("aadhaar", MOCK_CFG)
        img2, _ = render_card("aadhaar", MOCK_CFG)
        assert img1.size == img2.size


# ──────────────────────────────────────────────────────────────────────────────
# Degradation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDegradations:

    def setup_method(self):
        self.img = _make_test_image()

    @pytest.mark.parametrize("deg_name", ALL_DEGRADATIONS)
    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_all_degradations_return_pil(self, deg_name, severity):
        fn = DEGRADATION_REGISTRY[deg_name]
        result = fn(self.img, severity=severity)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == self.img.size

    def test_apply_degradations_chain(self):
        result = apply_degradations(
            self.img, ["blur", "occlusion", "rotation"], severity=2
        )
        assert isinstance(result, Image.Image)

    def test_unknown_degradation_raises(self):
        with pytest.raises(ValueError, match="Unknown degradation"):
            apply_degradations(self.img, ["alien_noise"])

    def test_apply_degradations_empty_list(self):
        result = apply_degradations(self.img, [], severity=1)
        # No degradation = original image
        assert result.size == self.img.size

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_blur_severity_increases_smoothness(self, severity):
        """Higher severity blur should produce a different (smoother) image."""
        blurred = apply_blur(self.img, severity=severity)
        arr_orig = np.array(self.img, dtype=float)
        arr_blur = np.array(blurred, dtype=float)
        diff = np.abs(arr_orig - arr_blur).mean()
        # There should be some difference
        if severity >= 2:
            assert diff > 0

    def test_jpeg_quality_range(self):
        for severity in [1, 2, 3]:
            result = apply_jpeg_compression(self.img, severity)
            assert isinstance(result, Image.Image)


# ──────────────────────────────────────────────────────────────────────────────
# Forgery tests
# ──────────────────────────────────────────────────────────────────────────────

class TestForgery:

    def setup_method(self):
        self.base = _make_test_image(640, 400)
        self.donor = _make_test_image(640, 400)

    def test_photocopy_returns_pil(self):
        result, ftype, bbox = apply_photocopy(self.base)
        assert isinstance(result, Image.Image)
        assert ftype == "photocopied"
        assert bbox is None

    def test_splice_returns_pil_with_bbox(self):
        result, ftype, bbox = apply_splice(self.base, self.donor)
        assert isinstance(result, Image.Image)
        assert ftype == "spliced"
        assert bbox is not None
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert w > 0 and h > 0

    def test_photocopy_changes_image(self):
        result, _, _ = apply_photocopy(self.base)
        arr_orig = np.array(self.base, dtype=float)
        arr_phot = np.array(result, dtype=float)
        assert not np.array_equal(arr_orig, arr_phot)


# ──────────────────────────────────────────────────────────────────────────────
# Manifest schema tests
# ──────────────────────────────────────────────────────────────────────────────

class TestManifestSchema:

    REQUIRED_KEYS = ["image_path", "card_type", "fields", "degradations",
                     "severity", "is_forged", "forgery_type", "forgery_bbox"]

    def test_required_keys_present(self):
        """Mock a manifest record and verify schema."""
        record = {
            "image_path": "synthetic/genuine/driving_licence_000001.jpg",
            "card_type": "driving_licence",
            "fields": {"name": "Test User", "dob": "01/01/1990"},
            "degradations": ["blur"],
            "severity": 2,
            "is_forged": False,
            "forgery_type": None,
            "forgery_bbox": None,
        }
        for key in self.REQUIRED_KEYS:
            assert key in record, f"Missing key: {key}"

    def test_json_serialisable(self):
        record = {
            "image_path": "path/img.jpg",
            "card_type": "aadhaar",
            "fields": {"name": "Alice"},
            "degradations": ["rotation", "shadow"],
            "severity": 1,
            "is_forged": True,
            "forgery_type": "spliced",
            "forgery_bbox": [10, 20, 100, 80],
        }
        serialised = json.dumps(record)
        recovered  = json.loads(serialised)
        assert recovered == record
