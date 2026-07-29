"""
stage2_ocr_benchmark/finetune_trocr.py

Fine-tune microsoft/trocr-large-printed on synthetic ID card images for 3 epochs.
Uses HuggingFace Seq2SeqTrainer. Measures delta-CER on held-out test split.

Usage:
    python -m stage2_ocr_benchmark.finetune_trocr
    python -m stage2_ocr_benchmark.finetune_trocr --smoke-test
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_config, load_json, ensure_dir
from utils.logger import get_logger, init_wandb, finish_wandb
from utils.seed import set_global_seed

log = get_logger("stage2.finetune_trocr")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class IDCardOCRDataset:
    """
    torch.utils.data.Dataset wrapper for TrOCR fine-tuning.
    Each sample: image → concatenated field text string.
    """

    def __init__(self, records: List[Dict], processor, data_root: str,
                 max_target_length: int = 128):
        self.records = records
        self.processor = processor
        self.data_root = data_root
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.records)

    def _fields_to_text(self, fields: Dict) -> str:
        return " | ".join(f"{k}: {v}" for k, v in fields.items() if v)

    def __getitem__(self, idx):
        import torch
        rec = self.records[idx]
        img_path = Path(self.data_root) / rec["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384), (240, 240, 240))

        text = self._fields_to_text(rec.get("fields", {}))

        encoding = self.processor(
            images=img,
            text=text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = encoding["labels"].squeeze()
        # Replace padding token id with -100 so loss ignores it
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "labels":       labels,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Metrics callback
# ──────────────────────────────────────────────────────────────────────────────

def _make_compute_metrics(processor):
    from stage2_ocr_benchmark.metrics import mean_cer, mean_wer

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        pred_ids = np.argmax(logits, axis=-1) if logits.ndim == 3 else logits

        # Replace -100 in labels
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "cer": mean_cer(preds, labels),
            "wer": mean_wer(preds, labels),
        }

    return compute_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_finetune(cfg: Dict, smoke_test: bool = False) -> None:
    try:
        import torch
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            default_data_collator,
        )
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        sys.exit(1)

    s2 = cfg["stage2"]
    model_id    = s2["trocr_model"]
    ft_cfg      = s2["trocr_finetune"]
    output_dir  = ft_cfg["output_dir"]
    epochs      = 1 if smoke_test else ft_cfg["epochs"]
    batch_size  = 2 if smoke_test else ft_cfg["batch_size"]
    lr          = ft_cfg["learning_rate"]

    # Load splits
    splits_dir = Path(cfg["paths"]["splits"])
    if not (splits_dir / "train.csv").exists():
        log.error("Splits not found. Run stage1 split.py first.")
        sys.exit(1)

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")

    if smoke_test:
        train_df = train_df.head(20)
        val_df   = val_df.head(10)

    train_recs = train_df.to_dict(orient="records")
    val_recs   = val_df.to_dict(orient="records")

    # Load processor and model
    log.info(f"Loading TrOCR processor and model: {model_id}")
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Configure decoder
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    train_dataset = IDCardOCRDataset(train_recs, processor, data_root=".")
    val_dataset   = IDCardOCRDataset(val_recs,   processor, data_root=".")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Training on {device}")

    ensure_dir(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=ft_cfg.get("warmup_steps", 100),
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to=["wandb"] if cfg.get("wandb", {}).get("project") else [],
        run_name="trocr-finetune",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_make_compute_metrics(processor),
    )

    log.info("Starting TrOCR fine-tuning …")
    trainer.train()

    # Save final model + processor
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    log.info(f"Fine-tuned model saved → {output_dir}")

    # Delta-CER report
    log.info("Evaluating base vs fine-tuned CER on test split …")
    _evaluate_delta(cfg, processor, model, device)


def _evaluate_delta(cfg, processor, ft_model, device):
    """Compare base TrOCR vs fine-tuned on the test split."""
    from stage2_ocr_benchmark.engines import TrOCREngine
    from stage2_ocr_benchmark.metrics import cer

    splits_dir = Path(cfg["paths"]["splits"])
    if not (splits_dir / "test.csv").exists():
        log.warning("test.csv not found, skipping delta evaluation.")
        return

    test_df = pd.read_csv(splits_dir / "test.csv").head(100)
    test_recs = test_df.to_dict(orient="records")

    base_engine = TrOCREngine(cfg["stage2"]["trocr_model"], device=device)
    ft_cer_list, base_cer_list = [], []

    import torch
    ft_model.eval()

    for rec in test_recs:
        img_path = Path(".") / rec["image_path"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        gt  = " ".join(str(v) for v in rec.get("fields", {}).values() if v)

        # Base
        base_text = base_engine.run(img)

        # Fine-tuned
        pv = processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = ft_model.generate(pv)
        ft_text = processor.batch_decode(ids, skip_special_tokens=True)[0]

        base_cer_list.append(cer(base_text, gt))
        ft_cer_list.append(cer(ft_text, gt))

    if base_cer_list:
        base_mean = sum(base_cer_list) / len(base_cer_list)
        ft_mean   = sum(ft_cer_list)   / len(ft_cer_list)
        delta     = base_mean - ft_mean
        log.info(f"Delta-CER evaluation:")
        log.info(f"  Base TrOCR    CER: {base_mean:.4f}")
        log.info(f"  Fine-tuned    CER: {ft_mean:.4f}")
        log.info(f"  Improvement Δ:    {delta:+.4f}")

        from utils.logger import log_metrics
        log_metrics({
            "trocr_base_cer":     base_mean,
            "trocr_finetuned_cer": ft_mean,
            "trocr_delta_cer":    delta,
        })


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2 — TrOCR Fine-tune")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    run = init_wandb(cfg, stage="stage2", run_name="trocr-finetune", tags=["trocr"])
    run_finetune(cfg, smoke_test=args.smoke_test)
    finish_wandb()
    log.info("TrOCR fine-tune complete ✓")


if __name__ == "__main__":
    main()
