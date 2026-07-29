"""
stage5_dpo/train.py

DPO fine-tuning using TRL DPOTrainer.
Uses the SFT model as the reference model.
Only trains the LoRA adapters — very lightweight.

Usage:
    python -m stage5_dpo.train --sft-adapter checkpoints/vlm-sft/lora_adapters
    python -m stage5_dpo.train --sft-adapter ... --smoke-test
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_config, ensure_dir
from utils.logger import get_logger, init_wandb, finish_wandb
from utils.seed import set_global_seed
from stage5_dpo.preference_dataset import build_preference_dataset, load_dpo_dataset

log = get_logger("stage5.train")


# ──────────────────────────────────────────────────────────────────────────────
# Model loader (SFT checkpoint as ref model)
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_and_ref(sft_adapter_dir: str, cfg: dict):
    """
    policy_model  = SFT checkpoint (will be fine-tuned with DPO)
    ref_model     = frozen copy of the same SFT checkpoint
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    s3 = cfg["stage3"]
    use_fallback = s3.get("use_fallback", False)
    base_id = s3["model_id_fallback"] if use_fallback else s3["model_id"]

    log.info(f"Loading base model: {base_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(sft_adapter_dir, trust_remote_code=True)

    # Policy model: base + LoRA adapters (trainable)
    base_policy = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
    )
    policy_model = PeftModel.from_pretrained(base_policy, sft_adapter_dir)
    policy_model.enable_input_require_grads()
    policy_model.config.use_cache = False

    # Reference model: base + LoRA adapters (frozen, separate instance)
    base_ref = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
    )
    ref_model = PeftModel.from_pretrained(base_ref, sft_adapter_dir)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return policy_model, ref_model, processor


# ──────────────────────────────────────────────────────────────────────────────
# DPO training
# ──────────────────────────────────────────────────────────────────────────────

def run_dpo(cfg: dict, sft_adapter_dir: str, smoke_test: bool = False) -> None:
    from trl import DPOTrainer, DPOConfig

    s5 = cfg["stage5"]
    train_cfg  = s5["training"]
    output_dir = train_cfg["output_dir"]
    ensure_dir(output_dir)

    # Build preference dataset if it doesn't already exist
    dpo_path = Path(cfg["paths"]["data_root"]) / "dpo" / "preference_pairs.jsonl"
    if not dpo_path.exists():
        log.info("Preference dataset not found — building now …")
        sft_csv = str(
            Path(cfg["paths"]["outputs"]) / "vlm_inference" / "inference_train.csv"
        )
        build_preference_dataset(cfg, sft_inference_csv=sft_csv if Path(sft_csv).exists() else None)

    # Load preference dataset
    log.info("Loading preference dataset …")
    dpo_ds = load_dpo_dataset(cfg)
    if smoke_test:
        dpo_ds = dpo_ds.select(range(min(20, len(dpo_ds))))

    # Val split from preference dataset
    split = dpo_ds.train_test_split(test_size=0.1, seed=cfg["project"]["seed"])
    train_ds = split["train"]
    val_ds   = split["test"]

    log.info(f"DPO pairs — train: {len(train_ds)}  |  val: {len(val_ds)}")

    # Load models
    policy_model, ref_model, processor = _load_model_and_ref(sft_adapter_dir, cfg)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=s5["beta"],
        loss_type=s5["loss_type"],
        max_length=s5["max_length"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=1 if smoke_test else train_cfg["num_epochs"],
        max_steps=5 if smoke_test else -1,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50 if not smoke_test else 3,
        save_strategy="steps",
        save_steps=50 if not smoke_test else 3,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name="vlm-dpo",
        dataloader_num_workers=0,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
    )

    log.info("Starting DPO training …")
    trainer.train()

    # Save updated LoRA adapters
    adapter_path = Path(output_dir) / "dpo_lora_adapters"
    policy_model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    log.info(f"DPO LoRA adapters saved → {adapter_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 5 — DPO Alignment")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--sft-adapter", required=True,
                        help="Path to SFT LoRA adapter directory")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    init_wandb(cfg, stage="stage5", run_name="vlm-dpo", tags=["dpo", "alignment"])
    run_dpo(cfg, sft_adapter_dir=args.sft_adapter, smoke_test=args.smoke_test)
    finish_wandb()
    log.info("Stage 5 DPO complete ✓")


if __name__ == "__main__":
    main()
