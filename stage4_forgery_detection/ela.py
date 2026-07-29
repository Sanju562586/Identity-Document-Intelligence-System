"""
stage4_forgery_detection/ela.py

Error Level Analysis (ELA) feature extractor.

ELA reveals JPEG compression inconsistencies introduced by image manipulation:
  1. Re-save the image at a known JPEG quality level.
  2. Compute the absolute pixel-level difference (amplified).
  3. Regions that were re-compressed (e.g., spliced patches) show
     higher error levels than authentic regions compressed at that quality.

References:
  Krawetz, N. (2007). A Picture's Worth... Hacking Illustrated.
"""
import io
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Core ELA computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_ela(
    image: Image.Image,
    quality: int = 90,
    amplify: int = 10,
) -> np.ndarray:
    """
    Compute Error Level Analysis map for an image.

    Args:
        image:    Input PIL.Image (RGB or RGBA).
        quality:  JPEG re-save quality (lower = more amplification of authentic regions).
        amplify:  Scalar multiplier applied to the difference image.

    Returns:
        ELA map as np.ndarray of shape (H, W, 3), dtype uint8.
        Bright regions indicate higher error level (potential tampering).
    """
    image = image.convert("RGB")

    # Re-save at target quality
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_arr  = np.array(image,        dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)

    diff = np.abs(orig_arr - recomp_arr) * amplify
    ela_map = np.clip(diff, 0, 255).astype(np.uint8)
    return ela_map


def compute_ela_gray(
    image: Image.Image,
    quality: int = 90,
    amplify: int = 10,
) -> np.ndarray:
    """
    Returns a single-channel (grayscale) ELA map of shape (H, W), dtype uint8.
    Useful as the second input channel for the forgery detection head.
    """
    ela_rgb = compute_ela(image, quality=quality, amplify=amplify)
    return cv2.cvtColor(ela_rgb, cv2.COLOR_RGB2GRAY)


# ──────────────────────────────────────────────────────────────────────────────
# Batch + tensor utilities
# ──────────────────────────────────────────────────────────────────────────────

def ela_to_tensor(
    image: Image.Image,
    quality: int = 90,
    amplify: int = 10,
    target_size: Tuple[int, int] = (224, 224),
):
    """
    Compute ELA, resize to target_size, normalise to [0, 1], return torch.Tensor.
    Shape: (1, H, W)  — single channel, ready to concat with RGB features.
    """
    import torch
    ela_gray = compute_ela_gray(image, quality=quality, amplify=amplify)
    ela_resized = cv2.resize(ela_gray, target_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(ela_resized).float() / 255.0
    return tensor.unsqueeze(0)   # (1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def visualise_ela(
    image: Image.Image,
    quality: int = 90,
    amplify: int = 10,
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Create a side-by-side visualisation: original | ELA heatmap.
    Optionally saves to disk.
    """
    ela_rgb = compute_ela(image, quality=quality, amplify=amplify)

    # Apply a colour map for better visual contrast
    ela_gray = cv2.cvtColor(ela_rgb, cv2.COLOR_RGB2GRAY)
    ela_heat = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    ela_heat = cv2.cvtColor(ela_heat, cv2.COLOR_BGR2RGB)

    orig_np = np.array(image.convert("RGB"))
    # Resize both to same height
    H = max(orig_np.shape[0], ela_heat.shape[0])

    def _resize_to_height(arr, h):
        ratio = h / arr.shape[0]
        return cv2.resize(arr, (int(arr.shape[1] * ratio), h))

    orig_r = _resize_to_height(orig_np,   H)
    ela_r  = _resize_to_height(ela_heat,  H)
    side_by_side = np.hstack([orig_r, ela_r])
    vis_img = Image.fromarray(side_by_side)

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        vis_img.save(save_path)

    return vis_img
