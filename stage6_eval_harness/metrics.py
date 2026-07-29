"""
stage6_eval_harness/metrics.py

Unified metrics for the adversarial evaluation harness.
Combines all previous metric implementations into one clean interface.
"""
from typing import Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Re-export from stage-specific modules
# ──────────────────────────────────────────────────────────────────────────────

from stage2_ocr_benchmark.metrics import cer, wer, field_f1
from stage4_forgery_detection.evaluate import expected_calibration_error


# ──────────────────────────────────────────────────────────────────────────────
# Aggregated condition metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_condition_metrics(
    field_f1_scores:   List[float],
    forgery_probs:     Optional[List[float]],
    forgery_labels:    Optional[List[int]],
    latency_ms:        List[float],
) -> Dict[str, float]:
    """
    Compute all metrics for a single test condition
    (one degradation type × one severity level).

    Args:
        field_f1_scores:  Per-image field extraction F1 scores.
        forgery_probs:    Per-image P(forged) from detection head. None if unavailable.
        forgery_labels:   Ground-truth forgery labels (0/1). None if unavailable.
        latency_ms:       Per-image inference latency in ms.

    Returns:
        Dict of metric name → value.
    """
    metrics: Dict[str, float] = {}

    # Field extraction
    if field_f1_scores:
        arr = np.array(field_f1_scores)
        metrics["field_f1_mean"]   = float(arr.mean())
        metrics["field_f1_std"]    = float(arr.std())
        metrics["field_f1_p25"]    = float(np.percentile(arr, 25))
        metrics["field_f1_p75"]    = float(np.percentile(arr, 75))

    # Forgery detection
    if forgery_probs is not None and forgery_labels is not None:
        p = np.array(forgery_probs)
        y = np.array(forgery_labels)

        from sklearn.metrics import roc_auc_score, f1_score
        try:
            metrics["forgery_auroc"] = float(roc_auc_score(y, p))
        except Exception:
            metrics["forgery_auroc"] = 0.0

        preds = (p >= 0.5).astype(int)
        metrics["forgery_f1"] = float(f1_score(y, preds, zero_division=0))
        metrics["forgery_ece"] = float(expected_calibration_error(p, y))

    # Latency
    if latency_ms:
        lat = np.array(latency_ms)
        metrics["latency_median_ms"] = float(np.median(lat))
        metrics["latency_p95_ms"]    = float(np.percentile(lat, 95))

    return {k: round(v, 4) for k, v in metrics.items()}


def summarise_grid(
    grid_results: List[Dict],
    group_by: List[str] = ["degradation", "severity"],
) -> Dict:
    """
    Summarise a list of condition result dicts grouped by keys.
    """
    import pandas as pd
    df = pd.DataFrame(grid_results)
    numeric_cols = [c for c in df.columns if c not in group_by and df[c].dtype in [float, int]]
    summary = df.groupby(group_by)[numeric_cols].mean().reset_index()
    return summary.to_dict(orient="records")
