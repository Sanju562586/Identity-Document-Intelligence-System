"""
stage6_eval_harness/report.py

Generates the W&B report and HF-standard model card from harness results.

Usage:
    python -m stage6_eval_harness.report \
        --results outputs/eval_harness/harness_results.csv \
        --sft-adapter checkpoints/vlm-sft/lora_adapters
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_config, ensure_dir
from utils.logger import get_logger

log = get_logger("stage6.report")


# ──────────────────────────────────────────────────────────────────────────────
# W&B comparison charts
# ──────────────────────────────────────────────────────────────────────────────

def log_harness_to_wandb(df: pd.DataFrame, cfg: Dict) -> None:
    """Log grouped bar charts comparing all models to W&B."""
    try:
        import wandb
        if wandb.run is None:
            return

        # Full results table
        table = wandb.Table(dataframe=df)
        wandb.log({"harness/full_results": table})

        # Per-model averages
        if "model" in df.columns:
            for metric in ["field_f1_mean", "forgery_auroc", "latency_median_ms"]:
                if metric not in df.columns:
                    continue
                sub = df.groupby("model")[metric].mean().reset_index()
                sub_table = wandb.Table(dataframe=sub)
                wandb.log({
                    f"harness/avg_{metric}": wandb.plot.bar(
                        sub_table, label="model", value=metric,
                        title=f"Average {metric} by Model"
                    )
                })

            # Field F1 vs severity (line chart)
            if "severity" in df.columns:
                sev_df = (
                    df.groupby(["model", "severity"])["field_f1_mean"]
                    .mean().reset_index()
                )
                sev_table = wandb.Table(dataframe=sev_df)
                wandb.log({
                    "harness/f1_vs_severity": wandb.plot.line(
                        sev_table, x="severity", y="field_f1_mean", stroke="model",
                        title="Field Extraction F1 vs Severity"
                    )
                })
    except Exception as e:
        log.warning(f"W&B chart logging failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Model card generator
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CARD_TEMPLATE = """\
---
language: en
license: apache-2.0
tags:
- identity-document-intelligence
- field-extraction
- forgery-detection
- vision-language-model
- dpo
- peft
- lora
metrics:
- field-f1
- auroc
- ece
pipeline_tag: image-text-to-text
---

# Identity Document Intelligence System

**{model_name}** — Vision-Language Model fine-tuned for identity document field extraction
with preference alignment (DPO) for calibrated confidence.

## Model Description

This model is the result of a 6-stage ML pipeline:

1. **Synthetic Data Factory** — 5,000 synthetic ID cards (driving licence, Aadhaar, bank statement)
   with 8 degradation types and 3 forgery classes
2. **OCR Benchmark** — Tesseract, EasyOCR, TrOCR benchmarked across severity levels
3. **VLM SFT** — PaliGemma-3B / Qwen2-VL-2B fine-tuned with QLoRA (4-bit, r=16)
4. **Forgery Detection** — Dual-stream head (RGB + ELA) with Grad-CAM explanations
5. **DPO Alignment** — Preference tuning to reject overconfident wrong predictions
6. **Adversarial Eval Harness** — 24-condition grid evaluation with W&B reporting

## Intended Use

- Identity document field extraction (name, DOB, ID number, address)
- Document authenticity verification (genuine vs forged)
- Research on calibrated VLM predictions

## ⚠️ Limitations & Responsible Use

- Trained on **synthetic data only** — real-world performance requires validation
- **Not a replacement** for human document verification in high-stakes decisions
- Model may fail on non-English documents or uncommon document formats
- ELA-based forgery detection may produce false positives under heavy JPEG recompression

## Training Details

| Parameter         | Value                      |
|-------------------|----------------------------|
| Base Model        | PaliGemma-3B / Qwen2-VL-2B |
| LoRA Rank         | 16                         |
| LoRA Alpha        | 32                         |
| Quantisation      | 4-bit NF4 (bitsandbytes)   |
| SFT Epochs        | 3                          |
| DPO Beta          | 0.1                        |
| Training Images   | ~3,500 (genuine)           |
| DPO Pairs         | ~3,500                     |

## Evaluation Results

{eval_table}

## Usage

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import torch
from PIL import Image

processor = AutoProcessor.from_pretrained("{hf_repo}")
base = AutoModelForCausalLM.from_pretrained("google/paligemma-3b-pt-224",
                                            torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "{hf_repo}")

image = Image.open("your_id_card.jpg").convert("RGB")
prompt = "<image>\\nExtract all fields from this identity document and return a JSON object.\\n"
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512)
response = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)[0]
import json
fields = json.loads(response.split("\\n\\n")[0].strip())
print(fields)
```

## Citation

```bibtex
@misc{{identity_doc_intelligence_2025,
  title     = {{Identity Document Intelligence System: SFT+DPO Pipeline}},
  year      = {{2025}},
  note      = {{Synthetic data pipeline with VLM fine-tuning and DPO alignment}}
}}
```

## License

Apache 2.0 — see LICENSE file.

---
*Generated {date} by the Identity Document Intelligence System pipeline.*
"""


def generate_model_card(
    cfg: Dict,
    results_df: pd.DataFrame,
    model_name: str = "identity-doc-vlm-dpo",
) -> str:
    hf_repo = cfg.get("stage6", {}).get("hf_hub_repo", "your-username/identity-doc-vlm")
    out_path = cfg.get("stage6", {}).get("model_card_out", "outputs/model_card.md")
    ensure_dir(Path(out_path).parent)

    # Build eval table from harness results
    if not results_df.empty and "model" in results_df.columns:
        summary = results_df.groupby("model").agg({
            col: "mean" for col in ["field_f1_mean", "forgery_auroc", "latency_median_ms"]
            if col in results_df.columns
        }).reset_index()

        header = "| Model | Field F1 | Forgery AUROC | Latency ms |\n"
        header += "|-------|----------|---------------|------------|\n"
        rows = ""
        for _, row in summary.iterrows():
            rows += (
                f"| {row.get('model', 'N/A')} "
                f"| {row.get('field_f1_mean', 0):.4f} "
                f"| {row.get('forgery_auroc', 0):.4f} "
                f"| {row.get('latency_median_ms', 0):.1f} |\n"
            )
        eval_table = header + rows
    else:
        eval_table = "*(Run eval harness to populate this section)*"

    card_content = MODEL_CARD_TEMPLATE.format(
        model_name=model_name,
        hf_repo=hf_repo,
        eval_table=eval_table,
        date=datetime.now().strftime("%Y-%m-%d"),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    log.info(f"Model card written → {out_path}")
    return card_content


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 6 — Generate Report & Model Card")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--results", default="outputs/eval_harness/harness_results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df  = pd.read_csv(args.results) if Path(args.results).exists() else pd.DataFrame()

    from utils.logger import init_wandb, finish_wandb
    init_wandb(cfg, stage="stage6", run_name="report", tags=["report"])
    log_harness_to_wandb(df, cfg)
    generate_model_card(cfg, df)
    finish_wandb()
    log.info("Report generation complete ✓")


if __name__ == "__main__":
    main()
