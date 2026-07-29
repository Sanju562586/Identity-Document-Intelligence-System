"""
stage1_data_factory/generator.py

Main orchestrator for the Synthetic Data Factory.
Generates N identity card images across card types and forgery classes,
exports per-image ground-truth JSON, and writes a master manifest.

Usage:
    python -m stage1_data_factory.generator              # uses config/config.yaml
    python -m stage1_data_factory.generator --n 100      # quick smoke test
    python -m stage1_data_factory.generator --smoke-test # 10 images only
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_data_factory.degradations import sample_degradations, apply_degradations, ALL_DEGRADATIONS
from stage1_data_factory.forgery import apply_forgery
from stage1_data_factory.templates import render_card, RENDERERS
from utils.io import load_config, save_json, ensure_dir
from utils.logger import get_logger, init_wandb, log_metrics, finish_wandb
from utils.seed import set_global_seed

log = get_logger("stage1.generator")

CARD_TYPES = ["driving_licence", "aadhaar", "bank_statement"]


# ──────────────────────────────────────────────────────────────────────────────
# Single sample generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_genuine(
    idx: int,
    card_type: str,
    cfg: Dict,
    out_dir: Path,
) -> Dict:
    img, fields = render_card(card_type, cfg)
    max_deg = cfg["stage1"]["max_degradations_per_image"]
    severity = random.choice(cfg["stage1"]["degradation_severity_levels"])
    degradation_names = sample_degradations(max_count=max_deg)
    img = apply_degradations(img, degradation_names, severity)

    fname = f"{card_type}_{idx:06d}.jpg"
    fpath = out_dir / "genuine" / card_type / fname
    ensure_dir(fpath.parent)
    img.save(fpath, format="JPEG", quality=92)

    return {
        "image_path": str(fpath.relative_to(out_dir.parent)),
        "card_type":  card_type,
        "fields":     fields,
        "degradations": degradation_names,
        "severity":   severity,
        "is_forged":  False,
        "forgery_type": None,
        "forgery_bbox": None,
    }


def _generate_forged(
    idx: int,
    card_type: str,
    forgery_type: str,
    cfg: Dict,
    out_dir: Path,
    donor_pool: List[Image.Image],
) -> Dict:
    img, fields = render_card(card_type, cfg)
    max_deg = cfg["stage1"]["max_degradations_per_image"]
    severity = random.choice(cfg["stage1"]["degradation_severity_levels"])
    degradation_names = sample_degradations(max_count=max_deg)
    img = apply_degradations(img, degradation_names, severity)

    donor: Optional[Image.Image] = None
    if forgery_type == "spliced" and donor_pool:
        donor = random.choice(donor_pool).copy()

    try:
        tampered, ftype, bbox = apply_forgery(img, forgery_type, donor_img=donor)
    except Exception as e:
        log.warning(f"Forgery failed for idx={idx}: {e}. Falling back to genuine.")
        tampered = img
        ftype = "none"
        bbox = None

    fname = f"{card_type}_{forgery_type}_{idx:06d}.jpg"
    fpath = out_dir / forgery_type / card_type / fname
    ensure_dir(fpath.parent)
    tampered.save(fpath, format="JPEG", quality=92)

    return {
        "image_path": str(fpath.relative_to(out_dir.parent)),
        "card_type":  card_type,
        "fields":     fields,
        "degradations": degradation_names,
        "severity":   severity,
        "is_forged":  True,
        "forgery_type": ftype,
        "forgery_bbox": bbox,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Donor pool builder (genuine images used as splice donors)
# ──────────────────────────────────────────────────────────────────────────────

def _build_donor_pool(n: int, cfg: Dict) -> List[Image.Image]:
    pool = []
    for _ in range(n):
        ct = random.choice(CARD_TYPES)
        img, _ = render_card(ct, cfg)
        pool.append(img)
    return pool


# ──────────────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────────────

def run_generation(cfg: Dict, n_override: Optional[int] = None) -> List[Dict]:
    s1 = cfg["stage1"]
    dist = s1["class_distribution"]
    n_total = n_override or s1["n_total"]

    n_genuine    = int(n_total * dist["genuine"])
    n_spliced    = int(n_total * dist["spliced"])
    n_photocopied = n_total - n_genuine - n_spliced

    synthetic_dir = Path(cfg["paths"]["synthetic"])
    ensure_dir(synthetic_dir)

    log.info(f"Generating {n_total} images  (genuine={n_genuine}, "
             f"spliced={n_spliced}, photocopied={n_photocopied})")

    # Pre-build a donor pool for splice operations
    log.info("Building donor pool …")
    donor_pool = _build_donor_pool(min(50, n_spliced + 10), cfg)

    manifest: List[Dict] = []
    global_idx = 0
    t0 = time.time()

    # ── Genuine
    with tqdm(total=n_genuine, desc="genuine", unit="img") as pbar:
        for i in range(n_genuine):
            ct = CARD_TYPES[i % len(CARD_TYPES)]
            record = _generate_genuine(global_idx, ct, cfg, synthetic_dir)
            manifest.append(record)
            global_idx += 1
            pbar.update(1)

    # ── Spliced
    with tqdm(total=n_spliced, desc="spliced", unit="img") as pbar:
        for i in range(n_spliced):
            ct = CARD_TYPES[i % len(CARD_TYPES)]
            record = _generate_forged(global_idx, ct, "spliced", cfg,
                                      synthetic_dir, donor_pool)
            manifest.append(record)
            global_idx += 1
            pbar.update(1)

    # ── Photocopied
    with tqdm(total=n_photocopied, desc="photocopied", unit="img") as pbar:
        for i in range(n_photocopied):
            ct = CARD_TYPES[i % len(CARD_TYPES)]
            record = _generate_forged(global_idx, ct, "photocopied", cfg,
                                      synthetic_dir, donor_pool)
            manifest.append(record)
            global_idx += 1
            pbar.update(1)

    elapsed = time.time() - t0
    log.info(f"Generated {len(manifest)} images in {elapsed:.1f}s  "
             f"({len(manifest)/elapsed:.1f} img/s)")

    # Save manifest
    manifest_path = synthetic_dir / "manifest.json"
    save_json(manifest, manifest_path)
    log.info(f"Manifest saved → {manifest_path}")

    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# W&B summary
# ──────────────────────────────────────────────────────────────────────────────

def _log_generation_stats(manifest: List[Dict], cfg: Dict) -> None:
    run = init_wandb(cfg, stage="stage1", run_name="synthetic-data-factory",
                     tags=["data-generation"])
    if run is None:
        return

    import wandb
    n_genuine = sum(1 for r in manifest if not r["is_forged"])
    n_forged  = len(manifest) - n_genuine
    log_metrics({
        "total_images": len(manifest),
        "genuine": n_genuine,
        "forged":  n_forged,
        "spliced": sum(1 for r in manifest if r.get("forgery_type") == "spliced"),
        "photocopied": sum(1 for r in manifest if r.get("forgery_type") == "photocopied"),
    })

    # Sample table
    if cfg.get("wandb", {}).get("log_tables", True):
        rows = [[r["image_path"], r["card_type"], r["is_forged"],
                 r["forgery_type"], r["severity"], str(r["degradations"])]
                for r in manifest[:200]]
        table = wandb.Table(
            columns=["image_path", "card_type", "is_forged",
                     "forgery_type", "severity", "degradations"],
            data=rows,
        )
        run.log({"manifest_sample": table})
    finish_wandb()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1 — Synthetic Data Factory")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n", type=int, default=None,
                        help="Override n_total from config")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Generate only 10 images for a quick sanity check")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["project"]["seed"]
    set_global_seed(seed)

    n = 10 if args.smoke_test else args.n
    manifest = run_generation(cfg, n_override=n)
    _log_generation_stats(manifest, cfg)
    log.info("Stage 1 complete ✓")


if __name__ == "__main__":
    main()
