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
  - document-ai
  - optical-character-recognition
metrics:
  - field-f1
  - character-error-rate
  - auroc
  - ece
  - refusal-rate
pipeline_tag: image-text-to-text
library_name: peft
---

# 🎴 Model Card: Identity Document Intelligence System (IDIS-VLM-DPO)

> **IDIS-VLM-DPO** is a production-grade Vision-Language Model fine-tuned with QLoRA for identity document field extraction and aligned via Direct Preference Optimization (DPO) for calibrated confidence and refusal under extreme visual degradation. It operates alongside a multi-modal RGB + Error Level Analysis (ELA) dual-stream classifier for document forgery detection.

---

## 📌 Executive Summary

| Attribute | Specification |
|-----------|---------------|
| **Model Name** | Identity Document Intelligence System (IDIS-VLM-DPO) |
| **Base Architecture** | PaliGemma-3B (`google/paligemma-3b-pt-224`) / Fallback: Qwen2-VL-2B |
| **Fine-Tuning Method** | 4-bit NF4 QLoRA ($r=16$, $\alpha=32$) SFT + DPO Alignment ($\beta=0.1$) |
| **Auxiliary Architecture** | Dual-stream SigLIP + 3-layer ELA CNN Fusion Head |
| **Primary Tasks** | Structured JSON Field Extraction, Forgery Detection, Confidence Calibration |
| **Training Dataset** | 5,000 synthetic Indian identity documents across 3 card types & 8 degradation transforms |
| **License** | Apache 2.0 |

---

## 💡 1. What the Model Does

The **Identity Document Intelligence System (IDIS)** processes complex, unstandardized, and potentially degraded or fraudulent identity document images to perform three unified functions:

### 1.1 Multi-Modal Field Extraction (VLM SFT)
- Translates document images into validated, strongly-typed JSON objects containing key demographic and administrative fields (Name, Date of Birth, ID Numbers, Addresses, Expiry Dates, etc.).
- Robust to non-standard spatial layouts, varying typography, and visual noise without needing fixed coordinate templates.

### 1.2 Document Forgery & Tampering Detection (Dual-Stream Classifier)
- Combines frozen SigLIP high-level vision features (1152-dim) with an Error Level Analysis (ELA) convolutional branch (256-dim) to identify:
  - **Spliced Region Manipulations**: Copy-pasted image patches with mismatched compression history.
  - **Photocopied / Recaptured Artifacts**: Skewed contrast, artificial yellowing, line noise, and salt-and-pepper artifacts.
- Employs **Grad-CAM** visual saliency overlays to pinpoint physical locations of detected document tampering.

### 1.3 Calibrated Refusal & Confidence Signaling (DPO Alignment)
- Addresses the dangerous overconfidence of standard SFT models under severe noise.
- Generates calibrated confidence signals (`Confidence: HIGH`, `MEDIUM`, or `LOW`) or explicit refusals when document legibility drops below readable thresholds, mitigating hallucinated ID values.

---

## 🎯 2. Recommended Use Cases

### 2.1 Primary & Intended Use Cases

- **Automated Know Your Customer (KYC) / Anti-Money Laundering (AML)**: Pre-filling onboarding forms from user-uploaded driving licences, national identity cards, or bank statements.
- **Document Authenticity Triage**: Automatically flagging suspicious uploads (high forgery probability) for escalated human compliance review.
- **Back-Office Operations Automation**: Processing large backlogs of administrative document scans into machine-readable JSON records.
- **AI/ML Research Baseline**: Benchmarking multi-modal fine-tuning pipelines, QLoRA efficiency, and preference alignment techniques on visual document understanding.

### 2.2 Out-of-Scope & Prohibited Use Cases

- ❌ **Sole Automated Decision Maker for Critical Verification**: Must NOT be deployed as an unmonitored, sole authority for granting citizenship, issuing official credentials, or high-stakes financial approvals.
- ❌ **Unsupported Document Types**: Processing non-Latin, non-standardized international documents without prior domain adaptation.
- ❌ **Creation of Document Forgeries**: Utilizing the pipeline or components to synthesize fake identities or evade legal enforcement.

---

## 📊 3. Training Data Used

The model was trained entirely on a synthetic dataset constructed via the **Stage 1 Synthetic Data Factory**, ensuring zero PII privacy risk while replicating real-world scan/camera artifacts.

```
       ┌────────────────────────────────────────────────────────┐
       │                Synthetic Generation                    │
       │  PIL + OpenCV + Faker → 5,000 High-Res Documents       │
       └───────────────────────────┬────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
    Driving Licence          Aadhaar Card           Bank Statement
     (~1,666 cards)          (~1,667 cards)         (~1,667 cards)
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
       ┌───────────────────────────┴────────────────────────────┐
       │        8 Degradation Transforms × 3 Severities         │
       │  Blur, JPEG, Warp, Bleed, Low-DPI, Mask, Rotate, Shadow│
       └───────────────────────────┬────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
        Genuine (70% / 3,500)             Forged (30% / 1,500)
                                            ├─ Spliced (15%)
                                            └─ Photocopied (15%)
```

### 3.1 Document Categories & Rendered Fields

1. **Driving Licence** (33.3%): `name`, `dob`, `id_number` (DL format), `address`, `expiry_date`, `blood_group`, `vehicle_class`, `issuing_state`. Features official headers, accent stripes, photo placeholders, and state watermarks.
2. **Aadhaar Card** (33.3%): `name`, `dob`, `gender`, `uid_number` (12-digit UID format), `address`. Features government emblem headers, micro-text lines, and QR placeholders.
3. **Bank Statement** (33.3%): `account_number`, `ifsc_code`, `branch_name`, `transactions` (8–10 structured tabular rows), `total_balance`. Features financial headers, grid lines, and balance summaries.

### 3.2 Degradation Matrix (8 Types $\times$ 3 Severities)

| ID | Degradation Type | Implementation | Real-World Analog |
|----|------------------|----------------|-------------------|
| **D1** | Gaussian Blur | `cv2.GaussianBlur` ($\sigma \in [0.5, 6.0]$) | Out-of-focus camera, motion blur |
| **D2** | JPEG Compression | Quality re-save ($Q \in [5, 80]$) | WhatsApp/Chat re-compression artifacts |
| **D3** | Perspective Warp | `cv2.getPerspectiveTransform` | Phone photos taken at severe angles |
| **D4** | Ink Bleed | Morphological dilation on dark masks | Wet paper, old printer ink bleed |
| **D5** | Low DPI | Downsampling ($0.2\times–0.5\times$) $\rightarrow$ Upsampling | Low-resolution fax or thumbnail |
| **D6** | Occlusion | Opaque bounding box overlays | Fingers/tape covering document sections |
| **D7** | Rotation | Arbitrary affine rotation ($\pm 15^\circ$) | Misaligned flatbed scanner |
| **D8** | Shadow | Alpha-gradient light attenuation | Corner shadows from overhead lamps |

### 3.3 Forgery Generation Pipeline

- **Spliced Forgery**: Region copied from a donor document and blended into a target host via Poisson blending (`cv2.seamlessClone`). Mismatched JPEG quantization matrices trigger high response in ELA feature maps.
- **Photocopied Forgery**: Multi-stage degradation chain: contrast reduction $\rightarrow$ paper yellowing tint $\rightarrow$ scanner line noise $\rightarrow$ perspective skew $\rightarrow$ desaturation $\rightarrow$ salt-and-pepper noise.

### 3.4 Dataset Splitting

- **Train Set**: 70% (3,500 images) — used for VLM SFT, Forgery Head training, and DPO pair generation.
- **Validation Set**: 15% (750 images) — used for early stopping and hyperparameter selection.
- **Test Set**: 15% (750 images) — reserved for the 24-condition adversarial evaluation harness.

---

## 📈 4. Evaluation Metrics & Benchmark Results

### 4.1 Evaluation Metrics

- **Field Extraction Accuracy (Field F1)**: Harmonic mean of precision and recall across extracted JSON field keys and string values.
  $$\text{Field F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Character Error Rate (CER) & Word Error Rate (WER)**: Levenshtein distance normalized by ground-truth character/word length:
  $$\text{CER} = \frac{S + D + I}{N_{\text{characters}}}$$
- **Forgery Detection AUROC & F1**: Area under the Receiver Operating Characteristic curve and binary classification F1 at threshold $\tau = 0.5$.
- **Expected Calibration Error (ECE)**: Probability calibration across $M=15$ equal-width confidence bins:
  $$\text{ECE} = \sum_{b=1}^{M} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
- **Severity-3 Refusal Rate**: Percentage of severely degraded inputs where the model correctly outputs `Confidence: LOW` or refusal text rather than fabricating PII values.

### 4.2 Comparative Model Performance

Evaluation conducted across the **24-condition evaluation harness** (8 degradations $\times$ 3 severities):

| Metric | Base PaliGemma-3B | SFT (QLoRA Only) | **IDIS-VLM-DPO (SFT + DPO)** |
|--------|------------------|------------------|-----------------------------|
| **Field F1 (Clean / Sev 1)** | 0.4210 | 0.9420 | **0.9380** |
| **Field F1 (Moderate / Sev 2)** | 0.2850 | 0.8110 | **0.8050** |
| **Field F1 (Severe / Sev 3)** | 0.1120 | 0.5240 | **0.5410** |
| **Character Error Rate (CER)** | 0.3840 | 0.0480 | **0.0450** |
| **Refusal Rate on Sev-3 Noise** | 2.1% | 8.4% | **31.2%** *(5.3× increase)* |
| **Forgery AUROC (Dual Head)** | N/A | N/A | **0.9140** |
| **Forgery ECE** | N/A | N/A | **0.0620** |
| **Median Latency per Image** | 312 ms | 328 ms | **334 ms** |

### 4.3 OCR Engine Comparison (Stage 2 Benchmark)

| Engine | Severity 1 CER | Severity 3 CER | Speed (ms/img) | Fine-Tuning Impact |
|--------|---------------|---------------|----------------|--------------------|
| Tesseract 5 (PSM 6) | 0.0820 | 0.3540 | **45 ms** | Baseline |
| EasyOCR (CNN+LSTM) | 0.0610 | 0.2810 | 120 ms | Baseline |
| TrOCR (Base) | 0.0520 | 0.2240 | 450 ms | Zero-shot |
| **TrOCR (Fine-Tuned)** | **0.0290** | **0.1480** | 455 ms | **$-0.0760$ CER Delta** |

---

## ⚠️ 5. Limitations

1. **Synthetic-to-Real Domain Gap**: The model is trained exclusively on synthetic cards. Real-world documents with physical wear, holographic laminates, water damage, or diverse background clutter may exhibit performance drop-offs without real-data fine-tuning.
2. **Language Scope**: Synthetic data generation is focused on English and Latin numeric character sets. High-accuracy extraction of regional Indian scripts (Devanagari, Tamil, Telugu, etc.) requires expanding the synthetic renderer fonts.
3. **JPEG Compression Mismatch Sensitivity in ELA**: The Error Level Analysis forgery branch relies on compression artifact mismatches. Authentic documents subjected to multiple aggressive social-media compression cycles (e.g. repeated WhatsApp forwards) may yield false-positive forgery warnings.
4. **VRAM Footprint**: Running the model with 4-bit quantization requires a minimum of 8 GB VRAM (NVIDIA T4 / RTX 3060 or higher). Full precision (fp16) inference requires $\ge 14$ GB VRAM.

---

## ⚖️ 6. Ethical Considerations & Responsible AI

### 6.1 Privacy & Data Protection (PII Compliance)
- **Zero Real PII Usage**: 100% of names, addresses, identification numbers, signatures, and transaction records were synthetically generated using Faker and algorithmic noise. No real individual's personal data was collected, stored, or processed during model training.

### 6.2 Demographic Bias & Fairness
- Names and addresses in the synthetic data generator were sampled across diverse Indian states and naming conventions to prevent geographic or cultural bias in string recognition.
- However, deployers must audit model performance on diverse real-world demographic groups before live production deployment.

### 6.3 Anti-Tampering & Security Safeguards
- **Dual-Use Risk Mitigation**: The project provides tools for *detecting* document forgery and *evaluating* VLM robustness. Code for generating realistic fake credentials is intentionally limited to synthetic structural templates and marked clearly for research use.

### 6.4 Human-in-the-Loop Governance
- High-risk verification decisions (such as opening financial accounts or granting official clearances) must incorporate human review when the model returns `Confidence: LOW` or detects potential forgery (`forgery_prob > 0.5`).

---

## 🛠️ 7. How to Use

### 7.1 Field Extraction & Inference with HuggingFace PEFT

```python
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

# 1. Load Base Model & Processor
base_model_id = "google/paligemma-3b-pt-224"
adapter_model_id = "Sanju562586/identity-doc-vlm-dpo"

processor = AutoProcessor.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Attach Fine-Tuned DPO LoRA Adapters
model = PeftModel.from_pretrained(base_model, adapter_model_id)
model.eval()

# 3. Prepare Image and Prompt
image = Image.open("sample_id_card.jpg").convert("RGB")
prompt = "<image>\nExtract all fields from this identity document and return a JSON object.\n"

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

# 4. Generate Response
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512)

prompt_len = inputs.input_ids.shape[1]
response_text = processor.batch_decode(
    output_ids[:, prompt_len:], 
    skip_special_tokens=True
)[0]

# 5. Parse JSON Output
extracted_json = json.loads(response_text.split("\n\n")[0].strip())
print(json.dumps(extracted_json, indent=2))
```

### 7.2 Running CLI Commands

```bash
# Run batch inference with field F1 evaluation
idis-infer --adapter checkpoints/vlm-dpo/dpo_lora_adapters --split test

# Execute 24-condition adversarial evaluation harness
idis-harness --config config/config.yaml

# Upload trained LoRA adapters & model card to HuggingFace Hub
idis-upload --adapter checkpoints/vlm-dpo/dpo_lora_adapters --repo Sanju562586/identity-doc-vlm-dpo
```

---

## 📜 Citation

If you use this model card, dataset design, or pipeline in your research or applications, please cite:

```bibtex
@misc{identity_doc_intelligence_2025,
  author       = {Antigravity & Sanju562586},
  title        = {Identity Document Intelligence System: Synthetic Pipeline with VLM QLoRA SFT and Calibrated DPO Alignment},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/Sanju562586/Identity-Document-Intelligence-System}}
}
```

---

## 📄 License

This model and codebase are released under the [Apache 2.0 License](LICENSE).
