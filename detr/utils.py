"""Seeding, logging, learning-rate schedule and checkpoint helpers."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def warmup_step_schedule(warmup_iters: int, drop_iter: int, gamma: float = 0.1):
    """Linear warmup over ``warmup_iters`` steps, then ``x gamma`` from ``drop_iter`` on."""
    def factor(it: int) -> float:
        warm = min(1.0, (it + 1) / max(warmup_iters, 1))
        return warm * (gamma if it >= drop_iter else 1.0)
    return factor


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path) -> dict:
    """Load ``checkpoint["model"]`` into ``model``; returns the full checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt
