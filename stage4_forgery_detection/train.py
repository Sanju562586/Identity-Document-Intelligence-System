"""
stage4_forgery_detection/train.py

Training loop for the dual-stream forgery detection head.
Only the ELA CNN branch and fusion MLP are trained; vision encoder is frozen.

Usage:
    python -m stage4_forgery_detection.train
    python -m stage4_forgery_detection.train --smoke-test
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage4_forgery_detection.dual_stream_head import build_detector
from stage4_forgery_detection.ela import compute_ela_gray
from utils.io import load_config, ensure_dir
from utils.logger import get_logger, init_wandb, log_metrics, finish_wandb
from utils.seed import set_global_seed

log = get_logger("stage4.train")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ForgeryDataset(Dataset):
    """
    Loads ID card images and computes:
      - Normalised RGB tensor  (3, 224, 224)
      - ELA single-channel map (1, 224, 224)
      - Binary label: 0=genuine, 1=forged
    """

    IMG_SIZE = (224, 224)

    _rgb_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, records: List[Dict], ela_quality: int = 90, ela_amplify: int = 10):
        self.records     = records
        self.ela_quality = ela_quality
        self.ela_amplify = ela_amplify

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        label = int(rec.get("is_forged", False))

        img_path = Path(".") / rec["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", self.IMG_SIZE, (200, 200, 200))

        rgb_tensor = self._rgb_transform(img)

        ela_gray = compute_ela_gray(img, quality=self.ela_quality, amplify=self.ela_amplify)
        import cv2
        ela_resized = cv2.resize(ela_gray, self.IMG_SIZE)
        ela_tensor = torch.from_numpy(ela_resized).float().unsqueeze(0) / 255.0

        return {
            "pixel_values": rgb_tensor,   # (3, 224, 224)
            "ela_maps":     ela_tensor,    # (1, 224, 224)
            "labels":       torch.tensor(label, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Weighted sampler for class imbalance
# ──────────────────────────────────────────────────────────────────────────────

def _make_sampler(records: List[Dict]) -> WeightedRandomSampler:
    labels = [int(r.get("is_forged", False)) for r in records]
    n_genuine = labels.count(0)
    n_forged  = labels.count(1)
    n_total   = len(labels)

    w_genuine = n_total / (2 * max(n_genuine, 1))
    w_forged  = n_total / (2 * max(n_forged, 1))
    weights   = [w_forged if l == 1 else w_genuine for l in labels]
    return WeightedRandomSampler(weights, num_samples=n_total, replacement=True)


# ──────────────────────────────────────────────────────────────────────────────
# Training epoch
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimiser, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="train", leave=False):
        pv     = batch["pixel_values"].to(device)
        ela    = batch["ela_maps"].to(device)
        labels = batch["labels"].to(device)

        optimiser.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(pv, ela)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            logits = model(pv, ela)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for batch in tqdm(loader, desc="val", leave=False):
        pv     = batch["pixel_values"].to(device)
        ela    = batch["ela_maps"].to(device)
        labels = batch["labels"].to(device)

        logits = model(pv, ela)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs  = torch.softmax(logits, dim=-1)[:, 1]
        preds  = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    from sklearn.metrics import roc_auc_score
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = 0.0

    return total_loss / total, correct / total, auroc


# ──────────────────────────────────────────────────────────────────────────────
# Main training
# ──────────────────────────────────────────────────────────────────────────────

def run_training(cfg: Dict, smoke_test: bool = False) -> None:
    s4        = cfg["stage4"]
    train_cfg = s4["training"]
    output_dir = Path(train_cfg["output_dir"])
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Training forgery head on {device}")

    # Load splits
    splits_dir = Path(cfg["paths"]["splits"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")

    if smoke_test:
        train_df = train_df.head(40)
        val_df   = val_df.head(20)

    train_recs = train_df.to_dict(orient="records")
    val_recs   = val_df.to_dict(orient="records")

    # Datasets + loaders
    train_ds = ForgeryDataset(train_recs, ela_quality=s4["ela_quality"],
                              ela_amplify=s4["ela_amplify"])
    val_ds   = ForgeryDataset(val_recs,   ela_quality=s4["ela_quality"],
                              ela_amplify=s4["ela_amplify"])

    sampler    = _make_sampler(train_recs)
    batch_size = 4 if smoke_test else train_cfg["batch_size"]

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2,
                              shuffle=False, num_workers=0)

    # Model
    model = build_detector(cfg).to(device)
    # Only train non-encoder parameters
    trainable_params = [p for n, p in model.named_parameters()
                        if "vision_encoder" not in n and p.requires_grad]
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Loss with label smoothing
    class_weights = torch.tensor(train_cfg["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=train_cfg["label_smoothing"],
    )

    optimiser = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=train_cfg["num_epochs"]
    )
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    epochs = 2 if smoke_test else train_cfg["num_epochs"]

    best_auroc = 0.0
    best_path  = output_dir / "best_forgery_head.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_epoch(
            model, train_loader, optimiser, criterion, device, scaler
        )
        val_loss, val_acc, val_auroc = _val_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} AUROC={val_auroc:.4f}"
        )
        log_metrics({
            "stage4/epoch":        epoch,
            "stage4/train_loss":   train_loss,
            "stage4/train_acc":    train_acc,
            "stage4/val_loss":     val_loss,
            "stage4/val_acc":      val_acc,
            "stage4/val_auroc":    val_auroc,
        })

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), best_path)
            log.info(f"  ↑ New best AUROC {best_auroc:.4f} → saved to {best_path}")

    log.info(f"Training complete. Best AUROC: {best_auroc:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4 — Forgery Detection Training")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    init_wandb(cfg, stage="stage4", run_name="forgery-head", tags=["forgery", "detection"])
    run_training(cfg, smoke_test=args.smoke_test)
    finish_wandb()
    log.info("Stage 4 training complete ✓")


if __name__ == "__main__":
    main()
