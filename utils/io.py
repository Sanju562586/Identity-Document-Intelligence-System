"""
utils/io.py
JSON, image, and config I/O helpers used across all pipeline stages.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """Load and return the YAML config as a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# JSON
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)


def load_jsonl(path: Union[str, Path]):
    """Generator that yields one JSON object per line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Images
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: Union[str, Path], mode: str = "RGB") -> Image.Image:
    return Image.open(path).convert(mode)


def save_image(img: Union[Image.Image, np.ndarray], path: Union[str, Path], quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        img.save(path, format="JPEG", quality=quality)
    elif ext == ".png":
        img.save(path, format="PNG")
    else:
        img.save(path)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
