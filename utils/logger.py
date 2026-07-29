"""
utils/logger.py
Structured logging: Python logging + optional W&B integration.
All pipeline stages import get_logger() for consistent output.
"""
import logging
import os
import sys
from typing import Any, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Python Logger
# ──────────────────────────────────────────────────────────────────────────────

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:           # already configured
        return logger
    logger.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# W&B Helpers
# ──────────────────────────────────────────────────────────────────────────────

_wandb_run = None


def init_wandb(
    cfg: Dict[str, Any],
    stage: str,
    run_name: Optional[str] = None,
    tags: Optional[list] = None,
) -> Optional[Any]:
    """
    Initialise a W&B run if WANDB_API_KEY is set.
    Returns the run object or None if W&B is unavailable / disabled.
    """
    global _wandb_run
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        log = get_logger("wandb")
        log.warning("WANDB_API_KEY not set — W&B logging disabled for this run.")
        return None

    try:
        import wandb
        wb_cfg = cfg.get("wandb", {})
        _wandb_run = wandb.init(
            project=wb_cfg.get("project", "identity-doc-intelligence"),
            entity=wb_cfg.get("entity") or None,
            name=run_name or stage,
            tags=(tags or []) + [stage],
            config={stage: cfg.get(stage, {}), "project": cfg.get("project", {})},
            reinit=True,
        )
        return _wandb_run
    except Exception as exc:
        get_logger("wandb").warning(f"W&B init failed: {exc}")
        return None


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a dict of metrics to W&B if a run is active."""
    global _wandb_run
    if _wandb_run is None:
        return
    try:
        if step is not None:
            _wandb_run.log(metrics, step=step)
        else:
            _wandb_run.log(metrics)
    except Exception:
        pass


def finish_wandb() -> None:
    global _wandb_run
    if _wandb_run is not None:
        try:
            _wandb_run.finish()
        except Exception:
            pass
        _wandb_run = None
