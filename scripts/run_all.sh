#!/usr/bin/env bash
# =============================================================================
# scripts/run_all.sh
# Sequential full pipeline runner.
# Edit paths below to match your checkpoint locations.
# =============================================================================

set -euo pipefail

# Load environment variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Identity Document Intelligence System        ║"
echo "║  Full Pipeline Runner                         ║"
echo "╚══════════════════════════════════════════════╝"

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Synthetic Data Factory
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Stage 1: Synthetic Data Factory ──────────────"
python -m stage1_data_factory.generator
python -m stage1_data_factory.split

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: OCR Benchmark + TrOCR Fine-tune
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Stage 2: OCR Benchmark ───────────────────────"
python -m stage2_ocr_benchmark.benchmark
python -m stage2_ocr_benchmark.finetune_trocr

# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: VLM SFT
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Stage 3: VLM SFT ─────────────────────────────"
python -m stage3_vlm_sft.train

SFT_ADAPTER="checkpoints/vlm-sft/lora_adapters"

# Run inference on train split (needed for Stage 5 preference pairs)
python -m stage3_vlm_sft.inference \
  --adapter "$SFT_ADAPTER" \
  --split train \
  --n 500

# Run inference on test split
python -m stage3_vlm_sft.inference \
  --adapter "$SFT_ADAPTER" \
  --split test

# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: Forgery Detection Head
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Stage 4: Forgery Detection ───────────────────"
python -m stage4_forgery_detection.train

FORGERY_CKPT="checkpoints/forgery-head/best_forgery_head.pt"

python -m stage4_forgery_detection.evaluate \
  --checkpoint "$FORGERY_CKPT" \
  --split test \
  --gradcam

# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: DPO Alignment
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Stage 5: DPO Alignment ───────────────────────"
python -m stage5_dpo.preference_dataset \
  --sft-csv outputs/vlm_inference/inference_train.csv

python -m stage5_dpo.train --sft-adapter "$SFT_ADAPTER"

DPO_ADAPTER="checkpoints/vlm-dpo/dpo_lora_adapters"

python -m stage5_dpo.evaluate \
  --sft-adapter "$SFT_ADAPTER" \
  --dpo-adapter "$DPO_ADAPTER" \
  --split test

# ──────────────────────────────────────────────────────────────────────────────
# Stage 6: Adversarial Eval Harness
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Stage 6: Adversarial Eval Harness ────────────"
python -m stage6_eval_harness.harness \
  --sft-adapter  "$SFT_ADAPTER" \
  --dpo-adapter  "$DPO_ADAPTER" \
  --forgery-ckpt "$FORGERY_CKPT"

python -m stage6_eval_harness.report

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Pipeline complete! 🎉                        ║"
echo "║  Results in: outputs/                        ║"
echo "╚══════════════════════════════════════════════╝"
