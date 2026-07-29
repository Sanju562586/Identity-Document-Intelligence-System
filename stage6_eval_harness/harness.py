"""
stage6_eval_harness/harness.py

Adversarial Evaluation Harness.
Tests models across all 24 conditions: 8 degradations × 3 severity levels.

For each condition:
  - Applies the degradation at the given severity to N test images
  - Runs field extraction (Base VLM / SFT / SFT+DPO)
  - Runs forgery detection head
  - Collects: field F1, forgery AUROC, ECE, latency

Usage:
    python -m stage6_eval_harness.harness \
        --sft-adapter  checkpoints/vlm-sft/lora_adapters \
        --dpo-adapter  checkpoints/vlm-dpo/dpo_lora_adapters \
        --forgery-ckpt checkpoints/forgery-head/best_forgery_head.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_data_factory.degradations import DEGRADATION_REGISTRY
from stage3_vlm_sft.prompts import build_extraction_prompt
from stage4_forgery_detection.dual_stream_head import build_detector
from stage4_forgery_detection.ela import compute_ela_gray
from stage2_ocr_benchmark.metrics import field_f1
from stage6_eval_harness.metrics import compute_condition_metrics
from utils.io import load_config, save_json, ensure_dir
from utils.logger import get_logger, init_wandb, log_metrics, finish_wandb
from utils.seed import set_global_seed

log = get_logger("stage6.harness")

_RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

def _load_vlm(adapter_dir: Optional[str], cfg: Dict, device: str):
    """Load a VLM from adapter dir (None = base model, no adapters)."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    s3 = cfg["stage3"]
    use_fallback = s3.get("use_fallback", False)
    base_id = s3["model_id_fallback"] if use_fallback else s3["model_id"]

    processor = AutoProcessor.from_pretrained(
        adapter_dir if adapter_dir else base_id, trust_remote_code=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
    )
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base

    model.eval()
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Single image evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_image(
    img: Image.Image,
    rec: Dict,
    vlm_model,
    vlm_proc,
    forgery_model: Optional[torch.nn.Module],
    cfg: Dict,
    device: str,
) -> Dict:
    """Evaluate one image through VLM + forgery head. Returns metric dict."""
    card_type = rec.get("card_type")
    gt_fields = rec.get("fields", {})
    is_forged = int(rec.get("is_forged", False))

    # ── VLM field extraction
    prompt   = build_extraction_prompt(card_type)
    encoding = vlm_proc(
        images=img, text=f"<image>\n{prompt}\n", return_tensors="pt"
    ).to(device)

    t0 = time.perf_counter()
    out_ids = vlm_model.generate(
        **encoding,
        max_new_tokens=cfg["stage3"]["inference"]["max_new_tokens"],
        do_sample=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    new_ids  = out_ids[0, encoding["input_ids"].shape[1]:]
    response = vlm_proc.batch_decode([new_ids], skip_special_tokens=True)[0]

    try:
        pred_fields = json.loads(response.split("\n\n")[0].strip())
    except Exception:
        pred_fields = {}

    f1 = field_f1(pred_fields, gt_fields)

    # ── Forgery detection
    forgery_prob = None
    if forgery_model is not None:
        s4 = cfg["stage4"]
        pv  = _RGB_TRANSFORM(img).unsqueeze(0).to(device)
        ela = compute_ela_gray(img, quality=s4["ela_quality"], amplify=s4["ela_amplify"])
        import cv2
        ela_t = torch.from_numpy(
            cv2.resize(ela, (224, 224))
        ).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

        logits = forgery_model(pv, ela_t)
        probs  = torch.softmax(logits, dim=-1)
        forgery_prob = probs[0, 1].item()

    return {
        "field_f1":   round(f1, 4),
        "latency_ms": round(latency_ms, 2),
        "is_forged":  is_forged,
        "forgery_prob": forgery_prob,
    }


# ──────────────────────────────────────────────────────────────────────────────
# One condition (degradation × severity) evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _eval_condition(
    degradation: str,
    severity: int,
    records: List[Dict],
    vlm_model,
    vlm_proc,
    forgery_model,
    cfg: Dict,
    device: str,
    model_label: str,
) -> Dict:
    """Evaluate N images under a single condition."""
    degrade_fn = DEGRADATION_REGISTRY.get(degradation)
    field_f1s, forgery_probs, forgery_labels, latencies = [], [], [], []

    for rec in records:
        img_path = Path(".") / rec.get("image_path", "")
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Apply the target degradation
        img_degraded = degrade_fn(img, severity=severity) if degrade_fn else img

        result = _eval_image(img_degraded, rec, vlm_model, vlm_proc,
                             forgery_model, cfg, device)
        field_f1s.append(result["field_f1"])
        latencies.append(result["latency_ms"])
        if result["forgery_prob"] is not None:
            forgery_probs.append(result["forgery_prob"])
            forgery_labels.append(result["is_forged"])

    metrics = compute_condition_metrics(
        field_f1_scores=field_f1s,
        forgery_probs=forgery_probs if forgery_probs else None,
        forgery_labels=forgery_labels if forgery_labels else None,
        latency_ms=latencies,
    )
    metrics.update({
        "model":       model_label,
        "degradation": degradation,
        "severity":    severity,
        "n_images":    len(field_f1s),
    })
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Full harness run
# ──────────────────────────────────────────────────────────────────────────────

def run_harness(
    cfg: Dict,
    sft_adapter:    Optional[str] = None,
    dpo_adapter:    Optional[str] = None,
    forgery_ckpt:   Optional[str] = None,
    smoke_test:     bool = False,
) -> pd.DataFrame:
    s6 = cfg["stage6"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_per_cond = 5 if smoke_test else s6["n_per_condition"]

    # Test set
    test_df = pd.read_csv(Path(cfg["paths"]["splits"]) / "test.csv")
    records = test_df.to_dict(orient="records")

    # Forgery model
    forgery_model = None
    if forgery_ckpt and Path(forgery_ckpt).exists():
        forgery_model = build_detector(cfg).to(device)
        forgery_model.load_state_dict(
            torch.load(forgery_ckpt, map_location=device)
        )
        forgery_model.eval()
        log.info("Forgery detection model loaded ✓")

    # Models to compare
    model_configs = [("base_vlm", None)]
    if sft_adapter:
        model_configs.append(("sft", sft_adapter))
    if dpo_adapter:
        model_configs.append(("sft_dpo", dpo_adapter))

    all_results = []
    degradations = s6["degradation_types"] if not smoke_test else ["blur", "jpeg_compression"]
    severities   = s6["severity_levels"]   if not smoke_test else [1, 2]

    for model_label, adapter_dir in model_configs:
        log.info(f"\n══ Evaluating model: {model_label} ══")
        vlm_model, vlm_proc = _load_vlm(adapter_dir, cfg, device)

        for deg in degradations:
            for sev in severities:
                log.info(f"  degradation={deg:<20} severity={sev}")
                # Sample N records for this condition
                sample = records[:n_per_cond]
                result = _eval_condition(
                    deg, sev, sample, vlm_model, vlm_proc,
                    forgery_model, cfg, device, model_label,
                )
                all_results.append(result)
                log.info(f"    field_f1={result.get('field_f1_mean', 'N/A'):.4f}  "
                         f"auroc={result.get('forgery_auroc', 'N/A')}")

        # Free GPU memory between models
        del vlm_model, vlm_proc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    out_dir = ensure_dir(Path(s6["output_dir"]))
    results_df.to_csv(out_dir / "harness_results.csv", index=False)
    log.info(f"Harness results saved → {out_dir / 'harness_results.csv'}")

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 6 — Adversarial Eval Harness")
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--sft-adapter",  default=None)
    parser.add_argument("--dpo-adapter",  default=None)
    parser.add_argument("--forgery-ckpt", default=None)
    parser.add_argument("--smoke-test",   action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    init_wandb(cfg, stage="stage6", run_name="eval-harness", tags=["eval", "harness"])

    df = run_harness(
        cfg,
        sft_adapter=args.sft_adapter,
        dpo_adapter=args.dpo_adapter,
        forgery_ckpt=args.forgery_ckpt,
        smoke_test=args.smoke_test,
    )

    # Log summary to W&B
    for _, row in df.iterrows():
        prefix = f"{row.get('model')}/{row.get('degradation')}/sev{row.get('severity')}"
        log_metrics({
            f"harness/{prefix}/field_f1":      row.get("field_f1_mean", 0),
            f"harness/{prefix}/forgery_auroc": row.get("forgery_auroc", 0),
            f"harness/{prefix}/latency_ms":    row.get("latency_median_ms", 0),
        })

    finish_wandb()
    log.info("Stage 6 eval harness complete ✓")


if __name__ == "__main__":
    main()
