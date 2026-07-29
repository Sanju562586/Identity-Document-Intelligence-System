"""
stage3_vlm_sft/prompts.py

System prompt templates and conversation formatters for PaliGemma / Qwen2-VL SFT.
Keeps all prompt engineering in one place for easy iteration.
"""
import json
from typing import Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Task descriptions
# ──────────────────────────────────────────────────────────────────────────────

FIELD_EXTRACTION_SYSTEM = (
    "You are a precise identity document analysis assistant. "
    "Extract all visible fields from the identity document image and return "
    "a single valid JSON object. Do not add commentary. "
    "If a field is not clearly visible, use null for its value. "
    "Never hallucinate field values."
)

FORGERY_ASSESSMENT_SYSTEM = (
    "You are a forensic document analyst. Examine the identity document image "
    "for signs of tampering, forgery, or manipulation. "
    "Return a JSON object with your assessment."
)

# ──────────────────────────────────────────────────────────────────────────────
# Field extraction prompt builder
# ──────────────────────────────────────────────────────────────────────────────

CARD_TYPE_HINTS = {
    "driving_licence": (
        "This is a driving licence. Expected fields: name, dob, id_number, "
        "address, issue_date, expiry_date, blood_group, vehicle_class, state."
    ),
    "aadhaar": (
        "This is an Aadhaar identity card. Expected fields: name, dob, gender, "
        "id_number (12-digit UID), address."
    ),
    "bank_statement": (
        "This is a bank statement. Expected fields: name, account_number, ifsc, "
        "branch, bank_name, statement_from, statement_to, opening_balance, closing_balance."
    ),
}


def build_extraction_prompt(card_type: Optional[str] = None) -> str:
    """
    Build the user instruction for field extraction.
    Optionally includes card-type-specific hints.
    """
    hint = ""
    if card_type and card_type in CARD_TYPE_HINTS:
        hint = f"\n\nDocument type hint: {CARD_TYPE_HINTS[card_type]}"
    return (
        "Extract all fields from this identity document and return a JSON object.{hint}"
        "\n\nReturn ONLY valid JSON. Example format:\n"
        '{{"name": "...", "dob": "...", "id_number": "..."}}'
    ).format(hint=hint)


def build_extraction_response(fields: Dict, card_type: str) -> str:
    """Format the ground-truth response as a JSON string."""
    response = {"card_type": card_type, **fields}
    return json.dumps(response, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# PaliGemma conversation format
# ──────────────────────────────────────────────────────────────────────────────
# PaliGemma uses a simple prefix format: the image token is implicit.
# Input:  <image>\n{user_text}
# Target: {response}
# We follow the Seq2Seq / causal-LM format used by TRL SFTTrainer.

def format_paligemma_sample(
    user_text: str,
    response: str,
    image_token: str = "<image>",
) -> Dict[str, str]:
    """
    Returns a dict with 'prompt' and 'completion' keys
    compatible with TRL SFTTrainer chat template.
    """
    return {
        "prompt":     f"{image_token}\n{user_text}\n",
        "completion": response,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Qwen2-VL conversation format  (fallback model)
# ──────────────────────────────────────────────────────────────────────────────

def format_qwen_sample(
    user_text: str,
    response: str,
    system_text: str = FIELD_EXTRACTION_SYSTEM,
) -> list:
    """
    Returns a messages list compatible with Qwen2-VL chat template.
    The image placeholder is expected to be pre-inserted by the dataset.
    """
    return [
        {"role": "system",    "content": system_text},
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": response},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# DPO preference prompt builder
# ──────────────────────────────────────────────────────────────────────────────

CONFIDENCE_LEVELS = {
    "high":   "Confidence: HIGH — all fields extracted with high certainty.",
    "medium": "Confidence: MEDIUM — some fields may have low visibility.",
    "low":    "Confidence: LOW — image quality is degraded; fields may be inaccurate.",
}


def build_chosen_response(
    fields: Dict,
    card_type: str,
    confidence: str = "high",
) -> str:
    """Chosen = correct extraction with calibrated confidence."""
    payload = {"card_type": card_type, **fields}
    conf_note = CONFIDENCE_LEVELS.get(confidence, CONFIDENCE_LEVELS["high"])
    return json.dumps(payload, ensure_ascii=False) + f"\n\n{conf_note}"


def build_rejected_response(
    fields: Dict,
    card_type: str,
    error_type: str = "overconfident_wrong",
) -> str:
    """
    Rejected = wrong or overconfident response.
    error_type options:
      - overconfident_wrong   : corrupted fields + HIGH confidence (should be refused)
      - hallucinated_fields   : extra fabricated fields
    """
    import random, copy
    corrupted = copy.deepcopy(fields)

    if error_type == "overconfident_wrong":
        # Corrupt 1-2 fields
        keys = [k for k, v in corrupted.items() if v and isinstance(v, str)]
        for key in random.sample(keys, k=min(2, len(keys))):
            val = corrupted[key]
            # Introduce a plausible-looking error
            if any(c.isdigit() for c in val):
                val_chars = list(val)
                for i, c in enumerate(val_chars):
                    if c.isdigit():
                        val_chars[i] = str((int(c) + random.randint(1, 9)) % 10)
                        break
                corrupted[key] = "".join(val_chars)
            else:
                # Swap two characters
                val_l = list(val)
                if len(val_l) > 3:
                    i = random.randint(0, len(val_l) - 2)
                    val_l[i], val_l[i + 1] = val_l[i + 1], val_l[i]
                    corrupted[key] = "".join(val_l)

        payload = {"card_type": card_type, **corrupted}
        return (
            json.dumps(payload, ensure_ascii=False)
            + "\n\nConfidence: HIGH — all fields extracted with high certainty."
        )

    elif error_type == "hallucinated_fields":
        # Add fabricated fields
        corrupted["passport_number"] = "X" + "".join([str(random.randint(0, 9)) for _ in range(7)])
        corrupted["visa_status"] = "VALID"
        payload = {"card_type": card_type, **corrupted}
        return (
            json.dumps(payload, ensure_ascii=False)
            + "\n\nConfidence: HIGH — complete document analysis done."
        )

    return json.dumps({"card_type": card_type, **corrupted}, ensure_ascii=False)
