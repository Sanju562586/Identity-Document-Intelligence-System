"""
stage3_vlm_sft/collator.py

Custom data collator for PaliGemma VLM SFT.
Handles image loading, processor call, padding, and label masking
so that loss is only computed on the completion (not the prompt).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


@dataclass
class VLMSFTCollator:
    """
    Data collator for causal VLM SFT with image inputs.

    Pads sequences to the longest in the batch.
    Masks prompt tokens in labels with -100 (no loss on prompt).

    Args:
        processor:       HuggingFace processor (e.g. PaliGemmaProcessor).
        max_seq_length:  Maximum token length (truncation applied).
        image_size:      Target image size as (H, W) tuple.
        model_family:    "paligemma" | "qwen"
    """
    processor:      Any
    max_seq_length: int = 512
    image_size:     tuple = (224, 224)
    model_family:   str = "paligemma"

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        images    = []
        prompts   = []
        completions = []

        for f in features:
            # Load image
            img_path = f.get("image_path", "")
            try:
                img = Image.open(img_path).convert("RGB").resize(
                    (self.image_size[1], self.image_size[0]), Image.LANCZOS
                )
            except Exception:
                img = Image.new("RGB", (self.image_size[1], self.image_size[0]), (200, 200, 200))
            images.append(img)
            prompts.append(f.get("prompt", ""))
            completions.append(f.get("completion", ""))

        if self.model_family == "paligemma":
            return self._collate_paligemma(images, prompts, completions)
        else:
            return self._collate_qwen(images, prompts, completions)

    def _collate_paligemma(
        self,
        images: List[Image.Image],
        prompts: List[str],
        completions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        PaliGemma: processor takes images + text (prompt+completion).
        We encode prompt alone to find prompt length, then mask labels.
        """
        # Encode full sequences (prompt + completion)
        full_texts = [p + c for p, c in zip(prompts, completions)]
        full_enc = self.processor(
            images=images,
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        # Encode prompts only to find the boundary
        prompt_enc = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        input_ids  = full_enc["input_ids"]
        labels     = input_ids.clone()

        # Mask prompt tokens: set to -100
        for i, prompt_ids in enumerate(prompt_enc["input_ids"]):
            # Find actual length of prompt (before padding)
            prompt_len = (prompt_ids != self.processor.tokenizer.pad_token_id).sum().item()
            labels[i, :prompt_len] = -100

        # Also mask padding tokens
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        batch = {
            "input_ids":      input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels":         labels,
        }
        if "pixel_values" in full_enc:
            batch["pixel_values"] = full_enc["pixel_values"]
        if "token_type_ids" in full_enc:
            batch["token_type_ids"] = full_enc["token_type_ids"]

        return batch

    def _collate_qwen(
        self,
        images: List[Image.Image],
        prompts: List[str],
        completions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Qwen2-VL uses apply_chat_template — kept minimal for portability."""
        texts = [p + c for p, c in zip(prompts, completions)]
        enc = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        labels = enc["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return dict(enc)
