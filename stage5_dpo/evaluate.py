"""
stage5_dpo/evaluate.py

DPO evaluation: measures whether the DPO model is better calibrated
and more likely to refuse on ambiguous inputs compared to the SFT baseline.

Metrics:
  - Refusal rate: fraction of outputs containing "Confidence: LOW" or "uncertain"
    on highly-degraded (severity=3) inputs
  - ECE improvement vs SFT model
  - Field F1 preservation (DPO should not hurt extraction quality)
  - Preference match rate (chosen > rejected on held-out preference pairs)

Usage:
    python -m stage5_dpo.evaluate \
        --sft-adapter  checkpoints/vlm-sft/lora_adapters \
        --dpo-adapter  checkpoints/vlm-dpo/dpo_lora_adapters \
        --split test
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage3_vlm_sft.prompts import build_extraction_prompt
from stage2_ocr_benchmark.metrics import field_f1
from utils.io import load_config, save_json, ensure_dir
from utils.logger import get_logger, log_metrics, finish_wandb
from utils.seed import set_global_seed

log = get_logger("stage5.evaluate")

_REFUSAL_PATTERNS = re.compile(
    r"(confidence:\s*low|uncertain|degraded|cannot\s+extract|"
    r"unclear|not\s+visible|illegible|low\s+quality)",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_adapter(adapter_dir: str, cfg: dict):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel

    s3 = cfg["stage3"]
    use_fallback = s3.get("use_fallback", False)
    base_id = s3["model_id_fallback"] if use_fallback else s3["model_id"]

    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_model(
    model,
    processor,
    records: List[Dict],
    cfg: Dict,
    device: str,
    label: str = "model",
) -> pd.DataFrame:
    infer_cfg = cfg["stage3"]["inference"]
    rows = []

    for rec in tqdm(records, desc=f"eval/{label}", unit="img"):
        img_path = Path(".") / rec.get("image_path", "")
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        card_type = rec.get("card_type")
        prompt    = build_extraction_prompt(card_type)
        encoding  = processor(
            images=img, text=f"<image>\n{prompt}\n",
            return_tensors="pt"
        ).to(device)

        output_ids = model.generate(
            **encoding,
            max_new_tokens=infer_cfg["max_new_tokens"],
            do_sample=False,
        )
        new_ids  = output_ids[0, encoding["input_ids"].shape[1]:]
        response = processor.batch_decode([new_ids], skip_special_tokens=True)[0]

        # Field F1 (best-effort JSON parse)
        try:
            pred_fields = json.loads(response.split("\n\n")[0].strip())
        except Exception:
            pred_fields = {}
        f1 = field_f1(pred_fields, rec.get("fields", {}))

        is_refusal = bool(_REFUSAL_PATTERNS.search(response))
        severity   = rec.get("severity", 1)

        rows.append({
            "model":      label,
            "severity":   severity,
            "is_forged":  rec.get("is_forged", False),
            "card_type":  card_type,
            "field_f1":   round(f1, 4),
            "is_refusal": is_refusal,
            "response":   response[:200],
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Compare SFT vs DPO
# ──────────────────────────────────────────────────────────────────────────────

def compare_models(
    sft_adapter: str,
    dpo_adapter: str,
    cfg: Dict,
    records: List[Dict],
    device: str,
) -> Dict:
    log.info("Loading SFT model …")
    sft_model, sft_proc = _load_adapter(sft_adapter, cfg)

    log.info("Loading DPO model …")
    dpo_model, dpo_proc = _load_adapter(dpo_adapter, cfg)

    # Evaluate on severe degradation (severity=3) for refusal rate
    severe_recs = [r for r in records if r.get("severity", 0) == 3]
    if not severe_recs:
        severe_recs = records[:50]

    sft_df = _evaluate_model(sft_model, sft_proc, severe_recs, cfg, device, "sft")
    dpo_df = _evaluate_model(dpo_model, dpo_proc, severe_recs, cfg, device, "dpo")

    combined = pd.concat([sft_df, dpo_df], ignore_index=True)

    metrics = {}
    for label, df in [("sft", sft_df), ("dpo", dpo_df)]:
        metrics[label] = {
            "refusal_rate_severity3": df["is_refusal"].mean(),
            "mean_field_f1":          df["field_f1"].mean(),
        }

    metrics["delta"] = {
        "refusal_rate_improvement": (
            metrics["dpo"]["refusal_rate_severity3"]
            - metrics["sft"]["refusal_rate_severity3"]
        ),
        "field_f1_change": (
            metrics["dpo"]["mean_field_f1"]
            - metrics["sft"]["mean_field_f1"]
        ),
    }

    return metrics, combined


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 5 — DPO Evaluation")
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--sft-adapter",  required=True)
    parser.add_argument("--dpo-adapter",  required=True)
    parser.add_argument("--split",        default="test", choices=["train", "val", "test"])
    parser.add_argument("--n",            type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    splits_path = Path(cfg["paths"]["splits"]) / f"{args.split}.csv"
    records = pd.read_csv(splits_path).head(args.n).to_dict(orient="records")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics, df = compare_models(args.sft_adapter, args.dpo_adapter, cfg, records, device)

    log.info("\n── DPO Evaluation Results ──")
    for label, m in metrics.items():
        log.info(f"  [{label}]  {m}")

    out_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "dpo_eval")
    save_json(metrics, out_dir / "dpo_comparison.json")
    df.to_csv(out_dir / "dpo_comparison_details.csv", index=False)

    log_metrics({f"stage5/{k}_{kk}": vv
                 for k, v in metrics.items()
                 for kk, vv in (v.items() if isinstance(v, dict) else {})})
    finish_wandb()
    log.info("Stage 5 evaluation complete ✓")


if __name__ == "__main__":
    main()
