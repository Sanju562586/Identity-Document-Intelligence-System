"""
utils/seed.py
Global reproducibility seeding across Python, NumPy, PyTorch, and CUDA.
"""
import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed all RNG sources for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
