"""
stage3_vlm_sft/train.py

QLoRA SFT of PaliGemma-3B (or Qwen2-VL-2B) for identity document field extraction.

Training stack:
  - bitsandbytes 4-bit quantisation
  - PEFT LoRA (r=16, alpha=32)
  - TRL SFTTrainer
  - Mixed precision fp16, gradient checkpointing

Usage:
    python -m stage3_vlm_sft.train
    python -m stage3_vlm_sft.train --smoke-test
    python -m stage3_vlm_sft.train --use-fallback      # Qwen2-VL instead of PaliGemma
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_config, ensure_dir
from utils.logger import get_logger, init_wandb, finish_wandb
from utils.seed import set_global_seed
from stage3_vlm_sft.dataset import build_hf_dataset
from stage3_vlm_sft.collator import VLMSFTCollator

log = get_logger("stage3.train")


# ──────────────────────────────────────────────────────────────────────────────
# Model + processor loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_and_processor(cfg: dict, use_fallback: bool = False):
    """
    Load quantised base model + processor.
    Returns (model, processor, model_family).
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    s3 = cfg["stage3"]
    model_id = s3["model_id_fallback"] if use_fallback else s3["model_id"]
    quant    = s3["quantization"]

    log.info(f"Loading model: {model_id}  (quant={quant})")

    bnb_config = None
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # PaliGemma-specific: ensure image token is in tokenizer
    if "paligemma" in model_id.lower():
        model_family = "paligemma"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<image>"
    else:
        model_family = "qwen"

    model.config.use_cache = False          # disable KV cache for training
    model.enable_input_require_grads()      # needed for gradient checkpointing + PEFT
    log.info(f"Model loaded ({model_family})")
    return model, processor, model_family


# ──────────────────────────────────────────────────────────────────────────────
# LoRA setup
# ──────────────────────────────────────────────────────────────────────────────

def _apply_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora_cfg = cfg["stage3"]["lora"]
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def run_sft(cfg: dict, use_fallback: bool = False, smoke_test: bool = False) -> None:
    from transformers import TrainingArguments
    from trl import SFTTrainer

    s3 = cfg["stage3"]
    train_cfg = s3["training"]
    output_dir = train_cfg["output_dir"]
    ensure_dir(output_dir)

    # Dataset
    max_samples = 20 if smoke_test else None
    model_id_key = "model_id_fallback" if use_fallback else "model_id"
    model_family = "qwen" if use_fallback else "paligemma"

    log.info("Building training dataset …")
    train_ds = build_hf_dataset(cfg, split="train", model_family=model_family,
                                max_samples=max_samples)
    val_ds   = build_hf_dataset(cfg, split="val",   model_family=model_family,
                                max_samples=max_samples // 2 if max_samples else None)

    log.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Model + LoRA
    model, processor, model_family = _load_model_and_processor(cfg, use_fallback)
    model = _apply_lora(model, cfg)

    # Collator
    collator = VLMSFTCollator(
        processor=processor,
        max_seq_length=train_cfg["max_seq_length"],
        model_family=model_family,
    )

    # Training arguments
    epochs     = 1 if smoke_test else train_cfg["num_epochs"]
    steps      = 10 if smoke_test else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=steps or -1,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_steps=train_cfg["warmup_steps"],
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=train_cfg["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=f"vlm-sft-{model_family}",
        dataloader_num_workers=0,    # set to 0 for Windows compatibility
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        # dataset_text_field not used — collator handles everything
        peft_config=None,            # already applied
    )

    log.info("Starting SFT training …")
    trainer.train()

    # Save LoRA adapters
    adapter_path = Path(output_dir) / "lora_adapters"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    log.info(f"LoRA adapters saved → {adapter_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3 — VLM SFT")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--use-fallback", action="store_true",
                        help="Use Qwen2-VL-2B instead of PaliGemma")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Train for 10 steps on 20 samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    # Also respect config fallback flag
    use_fallback = args.use_fallback or cfg["stage3"].get("use_fallback", False)

    init_wandb(cfg, stage="stage3", run_name="vlm-sft", tags=["sft", "vlm"])
    run_sft(cfg, use_fallback=use_fallback, smoke_test=args.smoke_test)
    finish_wandb()
    log.info("Stage 3 SFT complete ✓")


if __name__ == "__main__":
    main()
