"""
stage2_ocr_benchmark/engines.py

Unified OCR engine wrappers.  Each engine exposes a single method:

    engine.run(image: PIL.Image) -> str

All engines are lazily initialised on first use and gracefully degrade
(return empty string + log warning) if the library is not installed.

Supported engines:
  - TesseractEngine   (pytesseract)
  - EasyOCREngine     (easyocr)
  - TrOCREngine       (transformers, microsoft/trocr-large-printed)
  - PaddleOCREngine   (paddleocr — optional)
"""
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

log = logging.getLogger("stage2.engines")
if not log.handlers:
    import sys
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseOCREngine(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, image: Image.Image) -> str:
        """Extract all text from the image as a single string."""
        ...

    def run_timed(self, image: Image.Image) -> Dict:
        """Return {'text': str, 'latency_ms': float}."""
        t0 = time.perf_counter()
        try:
            text = self.run(image)
        except Exception as e:
            log.warning(f"[{self.name}] inference error: {e}")
            text = ""
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {"text": text, "latency_ms": round(elapsed, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# Tesseract
# ──────────────────────────────────────────────────────────────────────────────

class TesseractEngine(BaseOCREngine):
    name = "tesseract"

    def __init__(self, lang: str = "eng", config: str = "--oem 3 --psm 6"):
        self._lang   = lang
        self._config = config
        self._pt     = None   # lazy import

    def _get_pt(self):
        if self._pt is None:
            try:
                import pytesseract
                self._pt = pytesseract
            except ImportError:
                raise RuntimeError(
                    "pytesseract is not installed. Run: pip install pytesseract\n"
                    "Also install the Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki"
                )
        return self._pt

    def run(self, image: Image.Image) -> str:
        pt = self._get_pt()
        text = pt.image_to_string(image.convert("RGB"),
                                  lang=self._lang, config=self._config)
        return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# EasyOCR
# ──────────────────────────────────────────────────────────────────────────────

class EasyOCREngine(BaseOCREngine):
    name = "easyocr"

    def __init__(self, languages=None, gpu: bool = True):
        self._languages = languages or ["en"]
        self._gpu       = gpu
        self._reader    = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(self._languages, gpu=self._gpu,
                                              verbose=False)
            except ImportError:
                raise RuntimeError("easyocr not installed. Run: pip install easyocr")
        return self._reader

    def run(self, image: Image.Image) -> str:
        reader = self._get_reader()
        arr = np.array(image.convert("RGB"))
        results = reader.readtext(arr, detail=0, paragraph=True)
        return " ".join(results).strip()


# ──────────────────────────────────────────────────────────────────────────────
# TrOCR
# ──────────────────────────────────────────────────────────────────────────────

class TrOCREngine(BaseOCREngine):
    name = "trocr"

    def __init__(
        self,
        model_id: str = "microsoft/trocr-large-printed",
        device: Optional[str] = None,
    ):
        self._model_id = model_id
        self._device   = device
        self._processor = None
        self._model     = None

    def _load(self):
        if self._model is not None:
            return
        try:
            # pyrefly: ignore [missing-import]
            import torch
            # pyrefly: ignore [missing-import]
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError:
            raise RuntimeError("transformers / torch not installed.")

        device = self._device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device
        log.info(f"Loading TrOCR model '{self._model_id}' on {device} …")
        self._processor = TrOCRProcessor.from_pretrained(self._model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_id)
        self._model.to(device)
        self._model.eval()
        log.info("TrOCR loaded ✓")

    def run(self, image: Image.Image) -> str:
        import torch
        self._load()
        pixel_values = self._processor(
            images=image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self._device)
        with torch.no_grad():
            ids = self._model.generate(pixel_values)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def load_finetuned(self, checkpoint_dir: str) -> None:
        """Hot-swap to a locally fine-tuned checkpoint."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
        log.info(f"Loading fine-tuned TrOCR from {checkpoint_dir} …")
        self._processor = TrOCRProcessor.from_pretrained(checkpoint_dir)
        self._model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(device)
        self._model.eval()
        log.info("Fine-tuned TrOCR loaded ✓")


# ──────────────────────────────────────────────────────────────────────────────
# PaddleOCR (optional)
# ──────────────────────────────────────────────────────────────────────────────

class PaddleOCREngine(BaseOCREngine):
    name = "paddleocr"

    def __init__(self, lang: str = "en", use_gpu: bool = True):
        self._lang    = lang
        self._use_gpu = use_gpu
        self._ocr     = None

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self._lang,
                                      use_gpu=self._use_gpu, show_log=False)
            except ImportError:
                raise RuntimeError(
                    "paddleocr not installed. Run: pip install paddleocr paddlepaddle"
                )
        return self._ocr

    def run(self, image: Image.Image) -> str:
        ocr = self._get_ocr()
        arr = np.array(image.convert("RGB"))
        results = ocr.ocr(arr, cls=True)
        lines = []
        if results and results[0]:
            for line in results[0]:
                if line and len(line) >= 2:
                    lines.append(line[1][0])
        return " ".join(lines).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Engine registry
# ──────────────────────────────────────────────────────────────────────────────

_ENGINE_MAP = {
    "tesseract": TesseractEngine,
    "easyocr":   EasyOCREngine,
    "trocr":     TrOCREngine,
    "paddleocr": PaddleOCREngine,
}


def build_engine(name: str, **kwargs) -> BaseOCREngine:
    """Factory: instantiate an engine by name."""
    cls = _ENGINE_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown engine '{name}'. Choose from: {list(_ENGINE_MAP)}")
    return cls(**kwargs)


def build_engines(names) -> Dict[str, BaseOCREngine]:
    """Build multiple engines and return a name→engine dict."""
    engines = {}
    for name in names:
        try:
            engines[name] = build_engine(name)
            log.info(f"Engine '{name}' ready")
        except Exception as e:
            log.warning(f"Engine '{name}' could not be initialised: {e}")
    return engines
