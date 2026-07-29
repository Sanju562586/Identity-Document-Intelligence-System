"""
stage1_data_factory/degradations.py

8 degradation transforms applied to rendered ID card images.
Each function accepts a PIL.Image and a severity level (1=mild, 2=medium, 3=severe)
and returns a degraded PIL.Image.

Degradation IDs:
  D1  blur              Gaussian blur
  D2  jpeg_compression  JPEG re-save at low quality
  D3  perspective_warp  Four-corner perspective distortion
  D4  ink_bleed         Text region morphological dilation
  D5  low_dpi           Downsample → nearest-upsample
  D6  occlusion         Random opaque patches
  D7  rotation          Small-angle rotation with border fill
  D8  shadow            Polygon gradient shadow overlay
"""
import io
import random
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────────────────────────────────────
# Severity parameter tables
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: (min_val, max_val) indexed by severity 1, 2, 3
_BLUR_SIGMA      = {1: (0.5, 1.2), 2: (1.5, 3.0), 3: (3.5, 6.0)}
_JPEG_QUALITY    = {1: (65, 80),   2: (30, 55),   3: (5, 25)}
_PERSP_SCALE     = {1: (0.01, 0.04), 2: (0.04, 0.09), 3: (0.09, 0.16)}
_INK_KERNEL      = {1: 1, 2: 2, 3: 3}
_LOW_DPI_SCALE   = {1: 0.75, 2: 0.5, 3: 0.35}
_OCC_PATCHES     = {1: (1, 2), 2: (2, 4), 3: (4, 7)}
_OCC_SIZE        = {1: (0.03, 0.08), 2: (0.06, 0.15), 3: (0.12, 0.25)}
_ROT_ANGLE       = {1: (-5, 5), 2: (-10, 10), 3: (-15, 15)}
_SHADOW_ALPHA    = {1: (0.10, 0.20), 2: (0.20, 0.40), 3: (0.40, 0.65)}


def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────────
# D1 — Gaussian Blur
# ──────────────────────────────────────────────────────────────────────────────

def apply_blur(img: Image.Image, severity: int = 2) -> Image.Image:
    lo, hi = _BLUR_SIGMA[severity]
    sigma = random.uniform(lo, hi)
    arr = _to_np(img)
    ksize = max(1, int(sigma * 3) | 1)   # odd kernel
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigma)
    return _to_pil(blurred)


# ──────────────────────────────────────────────────────────────────────────────
# D2 — JPEG Compression
# ──────────────────────────────────────────────────────────────────────────────

def apply_jpeg_compression(img: Image.Image, severity: int = 2) -> Image.Image:
    lo, hi = _JPEG_QUALITY[severity]
    quality = random.randint(lo, hi)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ──────────────────────────────────────────────────────────────────────────────
# D3 — Perspective Warp
# ──────────────────────────────────────────────────────────────────────────────

def apply_perspective_warp(img: Image.Image, severity: int = 2) -> Image.Image:
    arr = _to_np(img)
    H, W = arr.shape[:2]
    lo, hi = _PERSP_SCALE[severity]

    def jitter():
        return int(random.uniform(lo, hi) * min(W, H))

    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([
        [jitter(), jitter()],
        [W - jitter(), jitter()],
        [W - jitter(), H - jitter()],
        [jitter(), H - jitter()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (W, H), borderValue=(240, 240, 240))
    return _to_pil(warped)


# ──────────────────────────────────────────────────────────────────────────────
# D4 — Ink Bleed (text region morphological dilation)
# ──────────────────────────────────────────────────────────────────────────────

def apply_ink_bleed(img: Image.Image, severity: int = 2) -> Image.Image:
    arr = _to_np(img)
    k = _INK_KERNEL[severity]
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    # Dilate dark pixels only (ink regions)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    dark_mask = (gray < 80).astype(np.uint8) * 255
    dilated_mask = cv2.dilate(dark_mask, kernel, iterations=1)
    # Apply dilation only where text is dark
    bleed = arr.copy()
    spread = dilated_mask > 0
    bleed[spread] = np.clip(arr[spread].astype(int) - 30, 0, 255).astype(np.uint8)
    return _to_pil(bleed)


# ──────────────────────────────────────────────────────────────────────────────
# D5 — Low DPI (downsample + upsample)
# ──────────────────────────────────────────────────────────────────────────────

def apply_low_dpi(img: Image.Image, severity: int = 2) -> Image.Image:
    scale = _LOW_DPI_SCALE[severity]
    W, H = img.size
    small_w = max(32, int(W * scale))
    small_h = max(32, int(H * scale))
    downsampled = img.resize((small_w, small_h), Image.NEAREST)
    upsampled   = downsampled.resize((W, H), Image.NEAREST)
    return upsampled


# ──────────────────────────────────────────────────────────────────────────────
# D6 — Occlusion Patches
# ──────────────────────────────────────────────────────────────────────────────

def apply_occlusion(img: Image.Image, severity: int = 2) -> Image.Image:
    arr = _to_np(img).copy()
    H, W = arr.shape[:2]
    n_lo, n_hi = _OCC_PATCHES[severity]
    s_lo, s_hi = _OCC_SIZE[severity]
    n_patches = random.randint(n_lo, n_hi)
    for _ in range(n_patches):
        pw = int(random.uniform(s_lo, s_hi) * W)
        ph = int(random.uniform(s_lo, s_hi) * H)
        px = random.randint(0, max(0, W - pw))
        py = random.randint(0, max(0, H - ph))
        color = random.choice([(0, 0, 0), (255, 255, 255), (128, 128, 128)])
        arr[py:py + ph, px:px + pw] = color
    return _to_pil(arr)


# ──────────────────────────────────────────────────────────────────────────────
# D7 — Rotation
# ──────────────────────────────────────────────────────────────────────────────

def apply_rotation(img: Image.Image, severity: int = 2) -> Image.Image:
    lo, hi = _ROT_ANGLE[severity]
    angle = random.uniform(lo, hi)
    arr = _to_np(img)
    H, W = arr.shape[:2]
    cx, cy = W / 2, H / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return _to_pil(rotated)


# ──────────────────────────────────────────────────────────────────────────────
# D8 — Shadow Overlay (polygon gradient)
# ──────────────────────────────────────────────────────────────────────────────

def apply_shadow(img: Image.Image, severity: int = 2) -> Image.Image:
    arr = _to_np(img).astype(np.float32)
    H, W = arr.shape[:2]
    lo, hi = _SHADOW_ALPHA[severity]
    alpha = random.uniform(lo, hi)

    # Shadow polygon: one side of image darkened
    shadow = np.ones((H, W), dtype=np.float32)
    # Pick a random direction: left, right, top, bottom
    direction = random.choice(["left", "right", "top", "bottom"])
    shadow_width = int(random.uniform(0.2, 0.7) * (W if direction in ("left", "right") else H))

    if direction == "left":
        for i in range(shadow_width):
            shadow[:, i] = 1.0 - alpha * (1.0 - i / shadow_width)
    elif direction == "right":
        for i in range(shadow_width):
            shadow[:, W - 1 - i] = 1.0 - alpha * (1.0 - i / shadow_width)
    elif direction == "top":
        for i in range(shadow_width):
            shadow[i, :] = 1.0 - alpha * (1.0 - i / shadow_width)
    else:
        for i in range(shadow_width):
            shadow[H - 1 - i, :] = 1.0 - alpha * (1.0 - i / shadow_width)

    shadowed = arr * shadow[:, :, np.newaxis]
    return _to_pil(np.clip(shadowed, 0, 255))


# ──────────────────────────────────────────────────────────────────────────────
# Registry & Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

DEGRADATION_REGISTRY = {
    "blur":              apply_blur,
    "jpeg_compression":  apply_jpeg_compression,
    "perspective_warp":  apply_perspective_warp,
    "ink_bleed":         apply_ink_bleed,
    "low_dpi":           apply_low_dpi,
    "occlusion":         apply_occlusion,
    "rotation":          apply_rotation,
    "shadow":            apply_shadow,
}

ALL_DEGRADATIONS = list(DEGRADATION_REGISTRY.keys())


def apply_degradations(
    img: Image.Image,
    degradation_names: List[str],
    severity: int = 2,
) -> Image.Image:
    """Apply a sequence of named degradations at the given severity."""
    for name in degradation_names:
        fn = DEGRADATION_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown degradation: {name}")
        img = fn(img, severity)
    return img


def sample_degradations(max_count: int = 3) -> List[str]:
    """Randomly sample 0..max_count degradations (without replacement)."""
    n = random.randint(0, max_count)
    return random.sample(ALL_DEGRADATIONS, k=min(n, len(ALL_DEGRADATIONS)))
