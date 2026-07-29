"""
stage3_vlm_sft/inference.py

Batch inference with the fine-tuned VLM.
Outputs structured JSON per image, with regex fallback parsing if JSON is malformed.

Usage:
    python -m stage3_vlm_sft.inference --adapter checkpoints/vlm-sft/lora_adapters
    python -m stage3_vlm_sft.inference --adapter ... --split test --n 100
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage3_vlm_sft.prompts import build_extraction_prompt
from utils.io import load_config, save_json, ensure_dir
from utils.logger import get_logger
from utils.seed import set_global_seed

log = get_logger("stage3.inference")


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing with fallback
# ──────────────────────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_response(text: str) -> Optional[Dict]:
    """
    Attempt to parse the model's text output as JSON.
    Tries:
      1. Direct json.loads
      2. First JSON-like block extracted by regex
    Returns None if parsing fails.
    """
    text = text.strip()
    # Remove trailing confidence note
    if "\n\n" in text:
        text = text.split("\n\n")[0].strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex extraction
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def load_model(adapter_dir: str, cfg: Dict):
    """Load base model + LoRA adapter + processor."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel

    s3 = cfg["stage3"]
    use_fallback = s3.get("use_fallback", False)
    base_id = s3["model_id_fallback"] if use_fallback else s3["model_id"]

    log.info(f"Loading base model: {base_id}")
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    log.info("Model loaded with LoRA adapters ✓")
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Single image inference
# ──────────────────────────────────────────────────────────────────────────────

def infer_single(
    model,
    processor,
    image: Image.Image,
    card_type: Optional[str] = None,
    cfg: Optional[Dict] = None,
    device: str = "cpu",
) -> Tuple[Dict, float]:
    """
    Run field extraction on one image.
    Returns (parsed_json_or_empty_dict, latency_ms).
    """
    infer_cfg = (cfg or {}).get("stage3", {}).get("inference", {})
    max_new_tokens = infer_cfg.get("max_new_tokens", 512)
    temperature    = infer_cfg.get("temperature", 0.0)

    prompt = build_extraction_prompt(card_type)
    image  = image.convert("RGB")

    encoding = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        output_ids = model.generate(**encoding, **gen_kwargs)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Decode only the new tokens
    input_len = encoding["input_ids"].shape[1]
    new_ids   = output_ids[0, input_len:]
    text      = processor.batch_decode([new_ids], skip_special_tokens=True)[0]

    parsed = _parse_response(text) or {}
    return parsed, elapsed_ms


# ──────────────────────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────────────────────

def run_batch_inference(
    model,
    processor,
    records: List[Dict],
    cfg: Dict,
    device: str,
    max_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run inference on a list of manifest records.
    Returns a DataFrame with predictions and metrics.
    """
    from stage2_ocr_benchmark.metrics import field_f1

    results = []
    sample = records[:max_n] if max_n else records

    for rec in tqdm(sample, desc="VLM inference", unit="img"):
        img_path = Path(".") / rec["image_path"]
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        pred, lat = infer_single(model, processor, img,
                                 card_type=rec.get("card_type"),
                                 cfg=cfg, device=device)
        gt_fields = rec.get("fields", {})
        f1 = field_f1(pred, gt_fields) if gt_fields else 0.0
        json_ok = len(pred) > 0

        results.append({
            "image_path":  rec["image_path"],
            "card_type":   rec.get("card_type"),
            "severity":    rec.get("severity", 0),
            "is_forged":   rec.get("is_forged", False),
            "field_f1":    round(f1, 4),
            "json_valid":  json_ok,
            "latency_ms":  round(lat, 2),
            "pred":        json.dumps(pred),
            "gt":          json.dumps(gt_fields),
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3 — VLM Inference")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--adapter", required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=None,
                        help="Max images to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    from utils.io import load_json
    import pandas as pd

    splits_path = Path(cfg["paths"]["splits"]) / f"{args.split}.csv"
    if not splits_path.exists():
        log.error(f"Split file not found: {splits_path}")
        sys.exit(1)

    records = pd.read_csv(splits_path).to_dict(orient="records")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_model(args.adapter, cfg)

    df = run_batch_inference(model, processor, records, cfg, device, max_n=args.n)

    out_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "vlm_inference")
    out_path = out_dir / f"inference_{args.split}.csv"
    df.to_csv(out_path, index=False)

    log.info(f"\n── Inference Results ({args.split}) ──")
    log.info(f"Mean Field F1:      {df['field_f1'].mean():.4f}")
    log.info(f"JSON Valid Rate:    {df['json_valid'].mean():.4f}")
    log.info(f"Median Latency ms:  {df['latency_ms'].median():.1f}")
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
