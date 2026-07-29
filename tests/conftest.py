"""
tests/conftest.py
Shared pytest fixtures and configuration.
"""
import sys
from pathlib import Path

import pytest

# Ensure the project root is always on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mock_cfg():
    """Minimal pipeline config for unit tests (no file I/O)."""
    return {
        "project": {"seed": 42},
        "paths": {
            "data_root": "data",
            "synthetic": "data/synthetic",
            "splits":    "data/splits",
            "outputs":   "outputs",
            "logs":      "logs",
        },
        "stage1": {
            "image_size": [320, 200],
            "fonts": {
                "regular": "data/raw/fonts/DejaVuSans.ttf",
                "bold":    "data/raw/fonts/DejaVuSans-Bold.ttf",
                "mono":    "data/raw/fonts/DejaVuSansMono.ttf",
            },
            "n_total": 10,
            "class_distribution": {
                "genuine": 0.70, "spliced": 0.15, "photocopied": 0.15
            },
            "degradation_severity_levels": [1, 2, 3],
            "max_degradations_per_image": 2,
        },
        "stage2": {
            "engines": ["tesseract"],
            "trocr_model": "microsoft/trocr-large-printed",
            "trocr_finetune": {
                "epochs": 1, "batch_size": 2, "learning_rate": 5e-5,
                "warmup_steps": 10, "output_dir": "checkpoints/trocr-finetuned",
            },
            "benchmark_sample_size": 10,
        },
        "stage3": {
            "model_id": "google/paligemma-3b-pt-224",
            "model_id_fallback": "Qwen/Qwen2-VL-2B-Instruct",
            "use_fallback": False,
            "quantization": "4bit",
            "lora": {
                "r": 16, "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05, "bias": "none",
            },
            "training": {
                "num_epochs": 1, "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4, "learning_rate": 2e-4,
                "lr_scheduler": "cosine", "warmup_steps": 10, "fp16": True,
                "gradient_checkpointing": True, "save_steps": 100,
                "logging_steps": 10, "max_seq_length": 256,
                "output_dir": "checkpoints/vlm-sft",
            },
            "inference": {"max_new_tokens": 256, "num_beams": 1, "temperature": 0.0},
        },
        "stage4": {
            "vision_encoder_dim": 768,
            "ela_quality": 90,
            "ela_amplify": 10,
            "ela_cnn_out_dim": 64,
            "mlp_hidden_dims": [128, 64],
            "training": {
                "num_epochs": 2, "batch_size": 4, "learning_rate": 1e-3,
                "weight_decay": 1e-4, "class_weights": [1.0, 2.0],
                "label_smoothing": 0.1, "output_dir": "checkpoints/forgery-head",
            },
            "gradcam_samples": 5,
        },
        "stage5": {
            "beta": 0.1, "loss_type": "sigmoid", "max_length": 256,
            "training": {
                "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4,
                "learning_rate": 5e-5, "num_epochs": 1,
                "output_dir": "checkpoints/vlm-dpo",
            },
            "preference": {"cer_rejection_threshold": 0.3},
        },
        "stage6": {
            "degradation_types": ["blur", "jpeg_compression"],
            "severity_levels": [1, 2],
            "n_per_condition": 5,
            "latency_warmup_runs": 2,
            "latency_benchmark_runs": 10,
            "output_dir": "outputs/eval_harness",
            "model_card_out": "outputs/model_card.md",
            "hf_hub_repo": "",
        },
        "wandb": {"project": "test-project", "entity": "", "log_images": False, "log_tables": False},
    }
