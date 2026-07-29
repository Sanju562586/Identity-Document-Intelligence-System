"""
stage2_ocr_benchmark/metrics.py

Character Error Rate (CER) and Word Error Rate (WER) computation.
Also provides field-level exact-match and normalised edit distance.

All functions operate on plain strings; no external OCR dependency required.
"""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Text normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """
    Normalise OCR output for fair comparison:
      - Unicode NFC normalisation
      - Strip leading/trailing whitespace
      - Collapse internal runs of whitespace
      - Lowercase
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Edit distance (Levenshtein) — character level
# ──────────────────────────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    # Rolling two-row DP
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


# ──────────────────────────────────────────────────────────────────────────────
# CER — Character Error Rate
# ──────────────────────────────────────────────────────────────────────────────

def cer(hypothesis: str, reference: str, do_normalise: bool = True) -> float:
    """
    CER = edit_distance(hyp, ref) / len(ref)
    Returns 0.0 if reference is empty.
    """
    if do_normalise:
        hypothesis = normalise(hypothesis)
        reference  = normalise(reference)
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    dist = _edit_distance(hypothesis, reference)
    return dist / len(reference)


def mean_cer(hypotheses: List[str], references: List[str]) -> float:
    """Average CER over a list of (hyp, ref) pairs."""
    if not hypotheses:
        return 0.0
    scores = [cer(h, r) for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores)


# ──────────────────────────────────────────────────────────────────────────────
# WER — Word Error Rate
# ──────────────────────────────────────────────────────────────────────────────

def _word_edit_distance(hyp_words: List[str], ref_words: List[str]) -> int:
    m, n = len(hyp_words), len(ref_words)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if hyp_words[i - 1] == ref_words[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def wer(hypothesis: str, reference: str, do_normalise: bool = True) -> float:
    """WER = word_edit_distance / len(ref_words). Returns 0.0 for empty ref."""
    if do_normalise:
        hypothesis = normalise(hypothesis)
        reference  = normalise(reference)
    hyp_w = hypothesis.split()
    ref_w = reference.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    dist = _word_edit_distance(hyp_w, ref_w)
    return dist / len(ref_w)


def mean_wer(hypotheses: List[str], references: List[str]) -> float:
    if not hypotheses:
        return 0.0
    scores = [wer(h, r) for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores)


# ──────────────────────────────────────────────────────────────────────────────
# Field-level exact match
# ──────────────────────────────────────────────────────────────────────────────

def field_exact_match(pred: Dict[str, str], gold: Dict[str, str]) -> Dict[str, bool]:
    """
    Returns per-field boolean match dict.
    Only evaluates fields present in gold.
    """
    results = {}
    for key, gold_val in gold.items():
        pred_val = pred.get(key, "")
        results[key] = normalise(str(pred_val)) == normalise(str(gold_val))
    return results


def field_f1(pred: Dict[str, str], gold: Dict[str, str]) -> float:
    """Micro-averaged field-level F1 (precision × recall)."""
    matches = field_exact_match(pred, gold)
    n_gold = len(gold)
    n_pred = len(pred)
    n_match = sum(matches.values())
    if n_gold == 0 and n_pred == 0:
        return 1.0
    precision = n_match / n_pred if n_pred else 0.0
    recall    = n_match / n_gold if n_gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate report
# ──────────────────────────────────────────────────────────────────────────────

def compute_aggregate_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute CER, WER, and exact-match rate over a batch."""
    n = len(references)
    if n == 0:
        return {"cer": 0.0, "wer": 0.0, "exact_match": 0.0}

    cer_scores = [cer(h, r) for h, r in zip(hypotheses, references)]
    wer_scores = [wer(h, r) for h, r in zip(hypotheses, references)]
    exact = [normalise(h) == normalise(r) for h, r in zip(hypotheses, references)]

    return {
        "cer":         sum(cer_scores) / n,
        "wer":         sum(wer_scores) / n,
        "exact_match": sum(exact) / n,
        "cer_p25":     sorted(cer_scores)[n // 4],
        "cer_p75":     sorted(cer_scores)[3 * n // 4],
    }
