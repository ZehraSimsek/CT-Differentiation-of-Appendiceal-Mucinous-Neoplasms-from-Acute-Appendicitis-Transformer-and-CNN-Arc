"""
utils/seed.py — Reproducibility Utilities
==========================================
Seeds all relevant RNG backends to ensure deterministic behaviour
across runs (essential for clinical model auditability).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Fix random seeds for Python stdlib, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The global seed value.

    Notes
    -----
    * ``torch.backends.cudnn.deterministic = True`` may slightly reduce
      GPU throughput but guarantees bit-exact reproducibility.
    * ``PYTHONHASHSEED`` is set for hash-based data structures.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic cuDNN algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"  ✓ All random seeds set to {seed}")
