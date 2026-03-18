"""Helper utilities for DGE-DTI experiments."""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


def setup_logging(level: str = "INFO", log_dir: Optional[str] = None) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level string (e.g., ``"INFO"``, ``"DEBUG"``).
        log_dir: Optional directory to write a log file. If None, logs only to stdout.
    """
    handlers: list = [logging.StreamHandler()]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, "run.log")))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def get_device(prefer_gpu: bool = True) -> str:
    """Return the best available torch device string.

    Args:
        prefer_gpu: If True and CUDA is available, returns ``"cuda"``.

    Returns:
        Device string (``"cuda"`` or ``"cpu"``).
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"
