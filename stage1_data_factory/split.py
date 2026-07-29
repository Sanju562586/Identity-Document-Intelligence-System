"""
stage1_data_factory/split.py

Stratified train/val/test split of the synthetic manifest.
Stratification key: (card_type, is_forged, severity)

Outputs three CSV files:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_config, load_json, ensure_dir
from utils.logger import get_logger
from utils.seed import set_global_seed

log = get_logger("stage1.split")


def make_splits(
    manifest: List[Dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Stratified split preserving the distribution of
    (card_type, is_forged, severity) across splits.
    """
    # Build stratification key
    for rec in manifest:
        rec["_stratum"] = f"{rec['card_type']}_{int(rec['is_forged'])}_{rec['severity']}"

    df = pd.DataFrame(manifest)

    # Some strata might be too small for stratified split; handle gracefully
    stratum_counts = df["_stratum"].value_counts()
    small_strata = stratum_counts[stratum_counts < 3].index.tolist()
    df_small = df[df["_stratum"].isin(small_strata)]
    df_large = df[~df["_stratum"].isin(small_strata)]

    # Split large strata with stratification
    test_val_ratio = 1.0 - train_ratio
    if len(df_large) > 0:
        train_df, temp_df = train_test_split(
            df_large, test_size=test_val_ratio,
            stratify=df_large["_stratum"], random_state=seed,
        )
        relative_val = val_ratio / test_val_ratio
        val_df, test_df = train_test_split(
            temp_df, test_size=1.0 - relative_val,
            stratify=temp_df["_stratum"], random_state=seed,
        )
    else:
        train_df = val_df = test_df = pd.DataFrame()

    # Dump small strata entirely into training set
    if len(df_small) > 0:
        train_df = pd.concat([train_df, df_small], ignore_index=True)

    # Clean up helper column
    for df_ in [train_df, val_df, test_df]:
        if "_stratum" in df_.columns:
            df_.drop(columns=["_stratum"], inplace=True)

    splits = {
        "train": train_df.to_dict(orient="records"),
        "val":   val_df.to_dict(orient="records"),
        "test":  test_df.to_dict(orient="records"),
    }

    log.info(
        f"Split sizes — train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )
    return splits


def save_splits(splits: Dict[str, List[Dict]], out_dir: str) -> None:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    for name, records in splits.items():
        df = pd.DataFrame(records)
        fpath = out_dir / f"{name}.csv"
        df.to_csv(fpath, index=False)
        log.info(f"Saved {name} split → {fpath}  ({len(records)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 — Train/Val/Test Split")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    manifest_path = Path(cfg["paths"]["synthetic"]) / "manifest.json"
    if not manifest_path.exists():
        log.error(f"Manifest not found at {manifest_path}. Run generator.py first.")
        sys.exit(1)

    manifest = load_json(manifest_path)
    log.info(f"Loaded {len(manifest)} records from manifest")

    splits = make_splits(
        manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=cfg["project"]["seed"],
    )
    save_splits(splits, cfg["paths"]["splits"])
    log.info("Stage 1 split complete ✓")


if __name__ == "__main__":
    main()
