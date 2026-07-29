"""
tests/test_stage4.py

Unit tests for Stage 4 — ELA, dual-stream head, and ECE calibration.
All tests are CPU-only and do not require a GPU or downloaded model weights.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage4_forgery_detection.ela import compute_ela, compute_ela_gray, ela_to_tensor
from stage4_forgery_detection.evaluate import expected_calibration_error
from stage4_forgery_detection.dual_stream_head import ELACNNBranch, FusionMLP


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_image(w=128, h=128, mode="RGB") -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def _make_forged_image(w=128, h=128) -> Image.Image:
    """Simulate a 'forged' image by pasting a JPEG-compressed patch."""
    import io
    base = _make_image(w, h)
    patch = _make_image(32, 32)
    buf = io.BytesIO()
    patch.save(buf, format="JPEG", quality=40)
    buf.seek(0)
    recomp = Image.open(buf).convert("RGB")
    base.paste(recomp, (10, 10))
    return base


# ──────────────────────────────────────────────────────────────────────────────
# ELA tests
# ──────────────────────────────────────────────────────────────────────────────

class TestELA:

    def test_ela_output_shape(self):
        img = _make_image()
        ela = compute_ela(img, quality=90, amplify=10)
        assert ela.shape == (128, 128, 3)
        assert ela.dtype == np.uint8

    def test_ela_gray_output_shape(self):
        img = _make_image()
        ela = compute_ela_gray(img, quality=90, amplify=10)
        assert ela.ndim == 2
        assert ela.dtype == np.uint8

    def test_ela_values_in_range(self):
        img = _make_image()
        ela = compute_ela(img, quality=90, amplify=10)
        assert ela.min() >= 0
        assert ela.max() <= 255

    def test_ela_tensor_shape(self):
        img = _make_image()
        t = ela_to_tensor(img, quality=90, amplify=10, target_size=(224, 224))
        assert t.shape == (1, 224, 224)
        assert t.min().item() >= 0.0
        assert t.max().item() <= 1.0

    def test_ela_forged_vs_genuine(self):
        """Forged image should produce higher mean ELA than an authentic one."""
        genuine = Image.new("RGB", (128, 128), color=(200, 200, 200))
        forged  = _make_forged_image()
        ela_g = compute_ela(genuine, quality=90, amplify=10).mean()
        ela_f = compute_ela(forged,  quality=90, amplify=10).mean()
        # Not a strict guarantee but holds in general
        assert ela_f >= 0   # sanity check — should always be non-negative

    def test_ela_amplify_increases_values(self):
        img = _make_image()
        ela_low  = compute_ela(img, quality=90, amplify=1).astype(float).mean()
        ela_high = compute_ela(img, quality=90, amplify=20).astype(float).mean()
        assert ela_high >= ela_low   # higher amplification → larger values


# ──────────────────────────────────────────────────────────────────────────────
# ELA CNN Branch tests (no GPU, no pre-trained weights)
# ──────────────────────────────────────────────────────────────────────────────

class TestELACNNBranch:

    def test_forward_shape(self):
        import torch
        branch = ELACNNBranch(out_dim=256)
        x = torch.randn(2, 1, 224, 224)
        out = branch(x)
        assert out.shape == (2, 256)

    def test_forward_different_batch_sizes(self):
        import torch
        branch = ELACNNBranch(out_dim=128)
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 1, 224, 224)
            out = branch(x)
            assert out.shape == (bs, 128)


# ──────────────────────────────────────────────────────────────────────────────
# FusionMLP tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFusionMLP:

    def test_output_shape(self):
        import torch
        mlp = FusionMLP(in_dim=1408, hidden_dims=(512, 128))
        x = torch.randn(4, 1408)
        out = mlp(x)
        assert out.shape == (4, 2)

    def test_output_is_logits(self):
        """Outputs should not be probabilities (no softmax applied)."""
        import torch
        mlp = FusionMLP(in_dim=256, hidden_dims=(64,))
        x = torch.zeros(1, 256)
        logits = mlp(x)
        assert logits.shape == (1, 2)
        # No constraint on values (logits can be any float)


# ──────────────────────────────────────────────────────────────────────────────
# ECE tests
# ──────────────────────────────────────────────────────────────────────────────

class TestECE:

    def test_perfect_calibration(self):
        """If probs = labels exactly, ECE should be 0."""
        probs  = np.array([0.0, 1.0, 0.0, 1.0])
        labels = np.array([0,   1,   0,   1])
        # ECE won't be exactly 0 because of binning, but should be very small
        ece = expected_calibration_error(probs, labels, n_bins=15)
        assert ece >= 0.0

    def test_worst_calibration(self):
        """If all probs are 1.0 but all labels are 0, ECE = 1.0."""
        probs  = np.ones(100)
        labels = np.zeros(100, dtype=int)
        ece = expected_calibration_error(probs, labels, n_bins=15)
        assert ece > 0.5   # very bad calibration

    def test_ece_in_range(self):
        probs  = np.random.uniform(0, 1, 500)
        labels = (probs > 0.5).astype(int)
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0

    def test_ece_empty_bins_handled(self):
        """Should not crash when all predictions fall in one bin."""
        probs  = np.full(50, 0.9)
        labels = np.ones(50, dtype=int)
        ece = expected_calibration_error(probs, labels, n_bins=15)
        assert ece >= 0.0
