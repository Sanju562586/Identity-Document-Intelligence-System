"""
stage5_dpo/preference_dataset.py

Builds the preference dataset (chosen / rejected pairs) for DPO training.

Strategy (no external API needed):
  1. Run the SFT model on all training images to obtain its predictions.
  2. For correct predictions on clean/mildly-degraded images → "chosen"
     with calibrated HIGH/MEDIUM/LOW confidence based on severity.
  3. Construct "rejected" responses via controlled corruption:
       a) Wrong fields + HIGH confidence  (overconfident mistake)
       b) Hallucinated extra fields       (hallucination)
  4. For images where SFT CER > threshold → chosen = correct GT,
     rejected = raw SFT output with HIGH confidence attached.

Output: data/dpo/preference_pairs.jsonl
Each line: {prompt, chosen, rejected, image_path, card_type, severity}
"""
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage3_vlm_sft.prompts import (
    build_chosen_response,
    build_rejected_response,
    build_extraction_prompt,
)
from stage2_ocr_benchmark.metrics import cer as compute_cer
from utils.io import load_config, append_jsonl, ensure_dir
from utils.logger import get_logger
from utils.seed import set_global_seed

log = get_logger("stage5.preference_dataset")


# ──────────────────────────────────────────────────────────────────────────────
# Confidence mapping (severity → calibration label)
# ──────────────────────────────────────────────────────────────────────────────

def _severity_to_confidence(severity: int) -> str:
    if severity == 1:
        return "high"
    elif severity == 2:
        return "medium"
    else:
        return "low"


# ──────────────────────────────────────────────────────────────────────────────
# Pair builders
# ──────────────────────────────────────────────────────────────────────────────

def _make_pair_from_gt(rec: Dict, rejected_type: str = "overconfident_wrong") -> Dict:
    """
    Build a preference pair using GT fields as chosen and corrupted version as rejected.
    Does NOT require running the model.
    """
    prompt    = build_extraction_prompt(rec.get("card_type"))
    fields    = rec.get("fields", {})
    card_type = rec.get("card_type", "unknown")
    severity  = rec.get("severity", 1)
    confidence = _severity_to_confidence(severity)

    chosen   = build_chosen_response(fields, card_type, confidence=confidence)
    rejected = build_rejected_response(fields, card_type, error_type=rejected_type)

    return {
        "prompt":     f"<image>\n{prompt}\n",
        "chosen":     chosen,
        "rejected":   rejected,
        "image_path": rec.get("image_path", ""),
        "card_type":  card_type,
        "severity":   severity,
        "is_forged":  rec.get("is_forged", False),
        "pair_type":  rejected_type,
    }


def _make_pair_from_sft_output(
    rec: Dict,
    sft_pred_json: str,
) -> Optional[Dict]:
    """
    When the SFT model produces a bad output (high CER), use:
      chosen   = GT with LOW confidence
      rejected = SFT's wrong output + HIGH confidence

    Returns None if the pair can't be constructed.
    """
    fields    = rec.get("fields", {})
    card_type = rec.get("card_type", "unknown")
    prompt    = build_extraction_prompt(card_type)

    # GT text for CER computation
    gt_text   = " ".join(str(v) for v in fields.values() if v)
    pred_text = sft_pred_json

    c = compute_cer(pred_text, gt_text)
    if c < 0.1:   # SFT was correct — not a good rejected example
        return None

    chosen = build_chosen_response(fields, card_type, confidence="low")
    rejected = sft_pred_json + "\n\nConfidence: HIGH — all fields extracted with high certainty."

    return {
        "prompt":     f"<image>\n{prompt}\n",
        "chosen":     chosen,
        "rejected":   rejected,
        "image_path": rec.get("image_path", ""),
        "card_type":  card_type,
        "severity":   rec.get("severity", 1),
        "is_forged":  rec.get("is_forged", False),
        "pair_type":  "sft_overconfident",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_preference_dataset(
    cfg: Dict,
    sft_inference_csv: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> List[Dict]:
    """
    Build the full preference dataset.

    Args:
        cfg:                 Pipeline config.
        sft_inference_csv:   Path to Stage 3 inference results CSV.
                             If provided, includes SFT-overconfident pairs.
        max_pairs:           Cap on total pairs.

    Returns:
        List of preference pair dicts (also saved to JSONL).
    """
    splits_dir = Path(cfg["paths"]["splits"])
    train_df   = pd.read_csv(splits_dir / "train.csv")
    records    = train_df.to_dict(orient="records")

    out_dir  = ensure_dir(Path(cfg["paths"]["data_root"]) / "dpo")
    out_path = out_dir / "preference_pairs.jsonl"

    # Remove existing file to start fresh
    if out_path.exists():
        out_path.unlink()

    pairs: List[Dict] = []

    # ── Strategy A: GT-based pairs (always available)
    error_types = ["overconfident_wrong", "hallucinated_fields"]
    for rec in records:
        et = random.choice(error_types)
        pair = _make_pair_from_gt(rec, rejected_type=et)
        pairs.append(pair)
        append_jsonl(pair, out_path)

    log.info(f"A: {len(pairs)} GT-based pairs generated")

    # ── Strategy B: SFT-output pairs (when inference results available)
    if sft_inference_csv and Path(sft_inference_csv).exists():
        inf_df = pd.read_csv(sft_inference_csv)
        # Match on image_path
        inf_map = {r["image_path"]: r.get("pred", "{}") for r in inf_df.to_dict("records")}

        cfg_threshold = cfg["stage5"]["preference"]["cer_rejection_threshold"]
        sft_count = 0

        for rec in records:
            ip  = rec.get("image_path", "")
            pred = inf_map.get(ip)
            if pred is None:
                continue
            pair = _make_pair_from_sft_output(rec, pred)
            if pair is not None:
                pairs.append(pair)
                append_jsonl(pair, out_path)
                sft_count += 1

        log.info(f"B: {sft_count} SFT-overconfident pairs generated")

    if max_pairs:
        pairs = pairs[:max_pairs]

    log.info(f"Total preference pairs: {len(pairs)}  →  {out_path}")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# HF Dataset builder for DPOTrainer
# ──────────────────────────────────────────────────────────────────────────────

def load_dpo_dataset(cfg: Dict):
    """
    Load the preference JSONL as a HuggingFace Dataset.
    Schema required by TRL DPOTrainer: {prompt, chosen, rejected}
    """
    from datasets import Dataset
    from utils.io import load_jsonl

    dpo_path = Path(cfg["paths"]["data_root"]) / "dpo" / "preference_pairs.jsonl"
    if not dpo_path.exists():
        raise FileNotFoundError(
            f"Preference dataset not found at {dpo_path}. "
            "Run build_preference_dataset() first."
        )

    records = list(load_jsonl(dpo_path))
    # DPOTrainer only needs: prompt, chosen, rejected
    minimal = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
               for r in records]
    return Dataset.from_list(minimal)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 5 — Build Preference Dataset")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--sft-csv", default=None,
                        help="Path to Stage 3 inference CSV (optional)")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])
    build_preference_dataset(cfg, sft_inference_csv=args.sft_csv, max_pairs=args.max_pairs)
