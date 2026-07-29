"""
stage1_data_factory/forgery.py

Forgery generators for two tamper types:
  1. spliced    — copy-move: a region from card B is pasted onto card A
                  (mimics replacing a name, photo, or ID number)
  2. photocopied — full card is put through a simulated scanner/printer chain

Both functions accept a PIL.Image (the base card) and return:
  - tampered PIL.Image
  - forgery_type str
  - forgery_bbox (x, y, w, h) tuple  |  None for photocopied
"""
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def _random_region(W: int, H: int, min_ratio: float = 0.10, max_ratio: float = 0.30):
    """Return a random (x, y, w, h) bounding box."""
    rw = int(random.uniform(min_ratio, max_ratio) * W)
    rh = int(random.uniform(min_ratio, max_ratio) * H)
    rx = random.randint(0, W - rw)
    ry = random.randint(0, H - rh)
    return rx, ry, rw, rh


def _poisson_blend(src: np.ndarray, dst: np.ndarray, mask: np.ndarray,
                   center: Tuple[int, int]) -> np.ndarray:
    """Use cv2 seamless clone for realistic splice blending."""
    try:
        result = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        return result
    except cv2.error:
        # Fallback: direct paste if seamlessClone fails (e.g., region too small)
        return dst


# ──────────────────────────────────────────────────────────────────────────────
# 1. Splice (Copy-Move) Forgery
# ──────────────────────────────────────────────────────────────────────────────

def apply_splice(
    base_img: Image.Image,
    donor_img: Image.Image,
) -> Tuple[Image.Image, str, Tuple[int, int, int, int]]:
    """
    Copy a random region from donor_img and paste it onto base_img.
    Uses Poisson blending for realistic compositing.

    Returns: (tampered_image, "spliced", (x, y, w, h))
    """
    W, H = base_img.size

    # Source region from donor (same size card expected)
    donor_img = donor_img.resize((W, H), Image.LANCZOS)
    dst = _to_np(base_img)
    src = _to_np(donor_img)

    # Pick a semantically plausible target region (upper 60% — where text lives)
    rx, ry, rw, rh = _random_region(W, H, min_ratio=0.12, max_ratio=0.28)
    ry = max(0, min(ry, int(H * 0.60) - rh))   # keep in text area

    # Extract patch from donor
    patch = src[ry:ry + rh, rx:rx + rw]

    # Destination region (may differ slightly for copy-move realism)
    offset_x = random.randint(-20, 20)
    offset_y = random.randint(-10, 10)
    dx = max(0, min(rx + offset_x, W - rw))
    dy = max(0, min(ry + offset_y, H - rh))

    # Build mask
    mask = np.ones((rh, rw, 3), dtype=np.uint8) * 255
    # Feather edges slightly
    feather_k = max(3, min(rw, rh) // 8) | 1
    gray_mask = cv2.GaussianBlur(mask[:, :, 0], (feather_k, feather_k), 0)
    mask = np.stack([gray_mask] * 3, axis=-1)

    # Expand dst to fit patch
    full_mask = np.zeros((H, W, 3), dtype=np.uint8)
    full_mask[dy:dy + rh, dx:dx + rw] = 255

    src_full = dst.copy()
    src_full[dy:dy + rh, dx:dx + rw] = patch

    center = (dx + rw // 2, dy + rh // 2)
    result = _poisson_blend(src_full, dst, full_mask[:, :, 0], center)

    # Subtle compression mismatch on the spliced region to introduce ELA artifact
    patch_img = Image.fromarray(result[dy:dy + rh, dx:dx + rw])
    import io
    buf = io.BytesIO()
    patch_img.save(buf, format="JPEG", quality=random.randint(50, 75))
    buf.seek(0)
    recomp = np.array(Image.open(buf).convert("RGB"))
    result[dy:dy + rh, dx:dx + rw] = recomp

    return _to_pil(result), "spliced", (dx, dy, rw, rh)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Photocopied Forgery
# ──────────────────────────────────────────────────────────────────────────────

def apply_photocopy(img: Image.Image) -> Tuple[Image.Image, str, None]:
    """
    Simulate the quality degradation of scanning a printed document then
    reprinting and rescanning it.

    Pipeline:
      1. Reduce contrast + add slight yellowing (print)
      2. Add scanner line noise
      3. Slight skew
      4. Desaturate partially (photocopy grey)
      5. Salt-and-pepper noise + halftone hint
    """
    arr = _to_np(img).astype(np.float32)
    H, W = arr.shape[:2]

    # --- Step 1: Contrast reduction + yellowing (simulate aged paper) ---
    contrast_factor = random.uniform(0.70, 0.85)
    arr = arr * contrast_factor + (1 - contrast_factor) * 200
    # Add yellowing: boost R & G slightly relative to B
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.05, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.02, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.90, 0, 255)

    # --- Step 2: Scanner line noise (horizontal banding) ---
    n_lines = random.randint(3, 12)
    for _ in range(n_lines):
        y = random.randint(0, H - 1)
        thickness = random.randint(1, 3)
        alpha = random.uniform(0.02, 0.10)
        arr[y:y + thickness, :] = arr[y:y + thickness, :] * (1 - alpha) + 255 * alpha

    # --- Step 3: Slight skew ---
    angle = random.uniform(-2.0, 2.0)
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    arr_u8 = cv2.warpAffine(arr_u8, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    arr = arr_u8.astype(np.float32)

    # --- Step 4: Partial desaturation ---
    gray = np.mean(arr, axis=2, keepdims=True)
    desat = random.uniform(0.30, 0.60)
    arr = arr * (1 - desat) + gray * desat

    # --- Step 5: Halftone-like noise ---
    noise = np.random.normal(0, random.uniform(4, 12), arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    # --- Step 6: Salt-and-pepper ---
    s_p_ratio = random.uniform(0.002, 0.008)
    n_sp = int(H * W * s_p_ratio)
    for _ in range(n_sp):
        y, x = random.randint(0, H - 1), random.randint(0, W - 1)
        arr[y, x] = random.choice([0.0, 255.0])

    result = np.clip(arr, 0, 255).astype(np.uint8)
    return _to_pil(result), "photocopied", None


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def apply_forgery(
    base_img: Image.Image,
    forgery_type: str,
    donor_img: Optional[Image.Image] = None,
) -> Tuple[Image.Image, str, Optional[Tuple]]:
    """
    Apply the requested forgery to base_img.

    Args:
        base_img:     The source ID card image.
        forgery_type: "spliced" | "photocopied"
        donor_img:    Required for "spliced" — provides the donor region.

    Returns:
        (tampered_image, forgery_type_str, forgery_bbox_or_None)
    """
    if forgery_type == "spliced":
        if donor_img is None:
            raise ValueError("donor_img is required for splice forgery")
        return apply_splice(base_img, donor_img)
    elif forgery_type == "photocopied":
        return apply_photocopy(base_img)
    else:
        raise ValueError(f"Unknown forgery type: {forgery_type}")
