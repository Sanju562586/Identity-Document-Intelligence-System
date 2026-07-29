"""
stage2_ocr_benchmark/benchmark.py

Full OCR benchmark runner.
For each engine × degradation severity × card type, runs OCR on N images,
computes CER/WER, logs to W&B, and saves a results CSV.

Usage:
    python -m stage2_ocr_benchmark.benchmark
    python -m stage2_ocr_benchmark.benchmark --engines tesseract easyocr --n 50
    python -m stage2_ocr_benchmark.benchmark --smoke-test
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_ocr_benchmark.engines import build_engines
from stage2_ocr_benchmark.metrics import compute_aggregate_metrics
from utils.io import load_config, load_json, ensure_dir
from utils.logger import get_logger, init_wandb, log_metrics, finish_wandb
from utils.seed import set_global_seed

log = get_logger("stage2.benchmark")


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth text extractor
# ──────────────────────────────────────────────────────────────────────────────

def _fields_to_text(fields: Dict) -> str:
    """Flatten GT field dict to a single space-joined string for CER/WER."""
    return " ".join(str(v) for v in fields.values() if v)


# ──────────────────────────────────────────────────────────────────────────────
# Single engine evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_engine(
    engine,
    records: List[Dict],
    data_root: str,
    max_n: int = 500,
) -> pd.DataFrame:
    rows = []
    sample = records[:max_n]
    for rec in tqdm(sample, desc=engine.name, unit="img", leave=False):
        img_path = Path(data_root) / rec["image_path"]
        if not img_path.exists():
            log.warning(f"Image not found: {img_path}")
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(f"Failed to load {img_path}: {e}")
            continue

        result = engine.run_timed(img)
        gt_text = _fields_to_text(rec.get("fields", {}))

        from stage2_ocr_benchmark.metrics import cer, wer
        rows.append({
            "engine":       engine.name,
            "card_type":    rec.get("card_type", "unknown"),
            "severity":     rec.get("severity", 0),
            "is_forged":    rec.get("is_forged", False),
            "degradations": json.dumps(rec.get("degradations", [])),
            "cer":          cer(result["text"], gt_text),
            "wer":          wer(result["text"], gt_text),
            "latency_ms":   result["latency_ms"],
            "pred_len":     len(result["text"]),
            "gt_len":       len(gt_text),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Full benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(cfg: Dict, engine_names: List[str], max_n: int) -> pd.DataFrame:
    synthetic_dir = Path(cfg["paths"]["synthetic"])
    manifest_path = synthetic_dir / "manifest.json"
    if not manifest_path.exists():
        log.error(f"Manifest not found at {manifest_path}. Run stage1 generator first.")
        sys.exit(1)

    manifest = load_json(manifest_path)
    log.info(f"Loaded {len(manifest)} records. Benchmarking {len(engine_names)} engines …")

    engines = build_engines(engine_names)
    if not engines:
        log.error("No engines could be initialised. Check dependencies.")
        sys.exit(1)

    all_dfs = []
    for name, engine in engines.items():
        log.info(f"\n── Evaluating: {name} ──")
        df = evaluate_engine(engine, manifest, data_root=".", max_n=max_n)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save results CSV
    out_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "ocr_benchmark")
    out_path = out_dir / "benchmark_results.csv"
    combined.to_csv(out_path, index=False)
    log.info(f"Results saved → {out_path}")

    # Summary by engine × severity
    summary = (
        combined.groupby(["engine", "severity"])
        .agg(mean_cer=("cer", "mean"), mean_wer=("wer", "mean"),
             median_latency_ms=("latency_ms", "median"), n=("cer", "count"))
        .reset_index()
    )
    summary_path = out_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path}")
    log.info("\n" + summary.to_string(index=False))

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# W&B logging
# ──────────────────────────────────────────────────────────────────────────────

def _log_to_wandb(df: pd.DataFrame, cfg: Dict) -> None:
    run = init_wandb(cfg, stage="stage2", run_name="ocr-benchmark",
                     tags=["ocr", "benchmark"])
    if run is None:
        return
    try:
        # pyrefly: ignore [missing-import]
        import wandb

        # Summary metrics per engine
        for engine in df["engine"].unique():
            sub = df[df["engine"] == engine]
            log_metrics({
                f"{engine}/mean_cer":        sub["cer"].mean(),
                f"{engine}/mean_wer":        sub["wer"].mean(),
                f"{engine}/median_latency":  sub["latency_ms"].median(),
            })

        # Full results table
        if cfg.get("wandb", {}).get("log_tables", True):
            table = wandb.Table(dataframe=df.head(1000))
            run.log({"benchmark_results": table})

        # Per-severity breakdown chart
        sev_df = (
            df.groupby(["engine", "severity"])["cer"]
            .mean()
            .reset_index()
            .rename(columns={"cer": "mean_cer"})
        )
        table2 = wandb.Table(dataframe=sev_df)
        run.log({
            "cer_by_severity": wandb.plot.line(
                table2, x="severity", y="mean_cer", stroke="engine",
                title="Mean CER by Severity Level"
            )
        })
    finally:
        finish_wandb()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2 — OCR Benchmark")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--engines", nargs="+", default=None,
                        help="Engine names to run (default: from config)")
    parser.add_argument("--n", type=int, default=None,
                        help="Max images per engine (default: from config)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on only 20 images per engine")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    engine_names = args.engines or cfg["stage2"]["engines"]
    max_n = 20 if args.smoke_test else (args.n or cfg["stage2"]["benchmark_sample_size"])

    df = run_benchmark(cfg, engine_names, max_n)
    _log_to_wandb(df, cfg)
    log.info("Stage 2 benchmark complete ✓")


if __name__ == "__main__":
    main()
