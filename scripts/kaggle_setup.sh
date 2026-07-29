#!/usr/bin/env bash
# =============================================================================
# scripts/kaggle_setup.sh
# Bootstrap script for Kaggle / Colab environment.
# Installs all dependencies, downloads fonts, sets up directories.
# =============================================================================

set -euo pipefail

echo "Setting up Identity Document Intelligence System on Kaggle/Colab …"

# ── GPU check ──────────────────────────────────────────────────────────────────
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
           print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ── Install dependencies ───────────────────────────────────────────────────────
pip install -q -r requirements.txt

# ── Install Tesseract (Kaggle) ─────────────────────────────────────────────────
if ! command -v tesseract &> /dev/null; then
  apt-get install -y -q tesseract-ocr tesseract-ocr-eng
  echo "Tesseract installed ✓"
fi

# ── Download DejaVu fonts ──────────────────────────────────────────────────────
FONT_DIR="data/raw/fonts"
mkdir -p "$FONT_DIR"

if [ ! -f "$FONT_DIR/DejaVuSans.ttf" ]; then
  echo "Downloading DejaVu fonts …"
  # Available in most Ubuntu environments
  DEJAVU_SRC="/usr/share/fonts/truetype/dejavu"
  if [ -d "$DEJAVU_SRC" ]; then
    cp "$DEJAVU_SRC/DejaVuSans.ttf"       "$FONT_DIR/"
    cp "$DEJAVU_SRC/DejaVuSans-Bold.ttf"  "$FONT_DIR/"
    cp "$DEJAVU_SRC/DejaVuSansMono.ttf"   "$FONT_DIR/"
    echo "Fonts copied from system ✓"
  else
    # Download from GitHub
    BASE_URL="https://github.com/dejavu-fonts/dejavu-fonts/raw/main/ttf"
    wget -q "$BASE_URL/DejaVuSans.ttf"      -O "$FONT_DIR/DejaVuSans.ttf"
    wget -q "$BASE_URL/DejaVuSans-Bold.ttf" -O "$FONT_DIR/DejaVuSans-Bold.ttf"
    wget -q "$BASE_URL/DejaVuSansMono.ttf"  -O "$FONT_DIR/DejaVuSansMono.ttf"
    echo "Fonts downloaded ✓"
  fi
fi

# ── Create output directories ──────────────────────────────────────────────────
mkdir -p data/{synthetic,splits}
mkdir -p data/synthetic/{genuine,spliced,photocopied}
mkdir -p checkpoints/{vlm-sft,vlm-dpo,forgery-head,trocr-finetuned}
mkdir -p outputs/{ocr_benchmark,vlm_inference,forgery_eval,dpo_eval,eval_harness}
mkdir -p logs

# ── Install package in editable mode ──────────────────────────────────────────
pip install -q -e .

echo ""
echo "Setup complete ✓"
echo "Next: copy your .env.example to .env and fill in WANDB_API_KEY and HF_TOKEN"
echo "Then run: bash scripts/run_all.sh"
