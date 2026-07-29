"""
stage4_forgery_detection/evaluate.py

Comprehensive evaluation of the forgery detection head:
  - AUROC
  - F1 @ 0.5 threshold
  - Precision-Recall curve
  - ECE (Expected Calibration Error, 15 bins)
  - Grad-CAM visualisations (50 forged samples)

Usage:
    python -m stage4_forgery_detection.evaluate --checkpoint checkpoints/forgery-head/best_forgery_head.pt
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage4_forgery_detection.dual_stream_head import build_detector
from stage4_forgery_detection.train import ForgeryDataset
from stage4_forgery_detection.gradcam import visualise_gradcam_batch
from utils.io import load_config, save_json, ensure_dir
from utils.logger import get_logger, init_wandb, log_metrics, finish_wandb
from utils.seed import set_global_seed

log = get_logger("stage4.evaluate")


# ──────────────────────────────────────────────────────────────────────────────
# ECE (Expected Calibration Error)
# ──────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE = Σ |B_m| / N * |acc(B_m) - conf(B_m)|
    where bins are equal-width over [0, 1].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(probs)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_probs  = probs[mask]
        bin_labels = labels[mask]
        acc  = bin_labels.mean()
        conf = bin_probs.mean()
        ece += len(bin_probs) / n_total * abs(acc - conf)

    return float(ece)


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    records: List[Dict],
    cfg: Dict,
    device: str,
) -> Dict:
    s4 = cfg["stage4"]
    dataset = ForgeryDataset(records, ela_quality=s4["ela_quality"],
                             ela_amplify=s4["ela_amplify"])
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_probs:  List[float] = []
    all_labels: List[int]   = []

    model.eval()
    for batch in tqdm(loader, desc="evaluating", unit="batch"):
        pv     = batch["pixel_values"].to(device)
        ela    = batch["ela_maps"].to(device)
        labels = batch["labels"]

        logits = model(pv, ela)
        probs  = torch.softmax(logits, dim=-1)[:, 1]

        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr  = (probs_arr >= 0.5).astype(int)

    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        precision_recall_curve, average_precision_score,
    )

    try:
        auroc = roc_auc_score(labels_arr, probs_arr)
    except Exception:
        auroc = 0.0

    f1        = f1_score(labels_arr, preds_arr, zero_division=0)
    precision = precision_score(labels_arr, preds_arr, zero_division=0)
    recall    = recall_score(labels_arr, preds_arr, zero_division=0)
    avg_prec  = average_precision_score(labels_arr, probs_arr)
    ece       = expected_calibration_error(probs_arr, labels_arr)
    accuracy  = (labels_arr == preds_arr).mean()

    metrics = {
        "auroc":          round(auroc, 4),
        "f1":             round(f1, 4),
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "avg_precision":  round(avg_prec, 4),
        "ece":            round(ece, 4),
        "accuracy":       round(accuracy, 4),
        "n_samples":      len(all_labels),
        "n_forged":       int(labels_arr.sum()),
        "n_genuine":      int((labels_arr == 0).sum()),
    }

    log.info("── Forgery Detection Evaluation ──")
    for k, v in metrics.items():
        log.info(f"  {k:<20}: {v}")

    return metrics, probs_arr, labels_arr


# ──────────────────────────────────────────────────────────────────────────────
# PR Curve to W&B
# ──────────────────────────────────────────────────────────────────────────────

def _log_pr_curve(probs: np.ndarray, labels: np.ndarray, metrics: Dict) -> None:
    try:
        import wandb
        from sklearn.metrics import precision_recall_curve

        if wandb.run is None:
            return

        prec, rec, thresholds = precision_recall_curve(labels, probs)
        pr_data = [[r, p] for r, p in zip(rec.tolist(), prec.tolist())]
        table   = wandb.Table(data=pr_data, columns=["recall", "precision"])
        wandb.log({
            "stage4/pr_curve": wandb.plot.line(
                table, x="recall", y="precision", title="Precision-Recall Curve"
            ),
            **{f"stage4/{k}": v for k, v in metrics.items()},
        })
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4 — Forgery Detection Evaluation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved model state dict (.pt)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--gradcam", action="store_true",
                        help="Also generate Grad-CAM visualisations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_detector(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    splits_path = Path(cfg["paths"]["splits"]) / f"{args.split}.csv"
    records = pd.read_csv(splits_path).to_dict(orient="records")

    init_wandb(cfg, stage="stage4", run_name="forgery-eval", tags=["evaluation"])
    metrics, probs, labels = run_evaluation(model, records, cfg, device)
    _log_pr_curve(probs, labels, metrics)

    # Save metrics
    out_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "forgery_eval")
    save_json(metrics, out_dir / f"metrics_{args.split}.json")

    if args.gradcam:
        log.info("Generating Grad-CAM visualisations …")
        gradcam_dir = out_dir / "gradcam"
        visualise_gradcam_batch(
            model, records, cfg,
            out_dir=str(gradcam_dir),
            n=cfg["stage4"]["gradcam_samples"],
            device=device,
        )

    finish_wandb()
    log.info("Stage 4 evaluation complete ✓")


if __name__ == "__main__":
    main()
