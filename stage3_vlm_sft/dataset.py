"""
stage3_vlm_sft/dataset.py

HuggingFace Dataset builder for VLM SFT.
Converts the Stage 1 manifest into image-instruction-response triples.

Output schema per record:
  {
    "image":      PIL.Image,
    "prompt":     str,
    "completion": str,
    "card_type":  str,
    "is_forged":  bool,
    "severity":   int,
  }
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage3_vlm_sft.prompts import (
    build_extraction_prompt,
    build_extraction_response,
    format_paligemma_sample,
    format_qwen_sample,
    FIELD_EXTRACTION_SYSTEM,
)
from utils.io import load_config, load_json
from utils.logger import get_logger

log = get_logger("stage3.dataset")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset record builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_record(rec: Dict, model_family: str = "paligemma") -> Optional[Dict]:
    """
    Convert a single manifest record into a training sample.
    Returns None if image not found.
    """
    img_path = Path(".") / rec["image_path"]
    if not img_path.exists():
        return None

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        log.warning(f"Cannot open {img_path}: {e}")
        return None

    card_type = rec.get("card_type", "unknown")
    fields    = rec.get("fields", {})
    severity  = rec.get("severity", 1)

    user_text = build_extraction_prompt(card_type)
    response  = build_extraction_response(fields, card_type)

    if model_family == "qwen":
        messages = format_qwen_sample(user_text, response, FIELD_EXTRACTION_SYSTEM)
        return {
            "image":    img,
            "messages": messages,
            "card_type": card_type,
            "is_forged": rec.get("is_forged", False),
            "severity":  severity,
        }
    else:  # paligemma default
        sample = format_paligemma_sample(user_text, response)
        return {
            "image":      img,
            "prompt":     sample["prompt"],
            "completion": sample["completion"],
            "card_type":  card_type,
            "is_forged":  rec.get("is_forged", False),
            "severity":   severity,
        }


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_hf_dataset(
    cfg: Dict,
    split: str = "train",
    model_family: str = "paligemma",
    max_samples: Optional[int] = None,
):
    """
    Build a HuggingFace Dataset from the split CSV.

    Args:
        cfg:          Pipeline config dict.
        split:        "train" | "val" | "test"
        model_family: "paligemma" | "qwen"
        max_samples:  If set, cap the dataset size (useful for debugging).

    Returns:
        datasets.Dataset
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise RuntimeError("datasets library not installed. Run: pip install datasets")

    splits_dir = Path(cfg["paths"]["splits"])
    csv_path   = splits_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {csv_path}. Run stage1/split.py first."
        )

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    if max_samples:
        records = records[:max_samples]

    log.info(f"Building '{split}' dataset ({len(records)} records, model={model_family}) …")

    samples = []
    skipped = 0
    for rec in records:
        sample = _build_record(rec, model_family)
        if sample is not None:
            samples.append(sample)
        else:
            skipped += 1

    log.info(f"  {len(samples)} valid samples ({skipped} skipped)")

    # Remove PIL Image from dataset (store as paths or encode)
    # We keep a lightweight version: text fields only; images loaded in collator
    def _strip_image(s):
        out = {k: v for k, v in s.items() if k != "image"}
        return out

    # Store the image_path so the collator can load lazily
    for i, (rec, s) in enumerate(zip(records, samples)):
        samples[i]["image_path"] = str(Path(".") / rec["image_path"])

    text_samples = [_strip_image(s) for s in samples]
    ds = Dataset.from_list(text_samples)
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Quick verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    ds = build_hf_dataset(cfg, split="train", max_samples=5)
    print(f"Dataset columns: {ds.column_names}")
    print(f"First sample prompt:\n{ds[0]['prompt']}")
    print(f"First sample completion:\n{ds[0]['completion']}")
