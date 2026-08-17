"""
engine — Training & Evaluation Engine
======================================
Re-exports the core training and evaluation entry-points.
"""

from engine.evaluator import evaluate
from engine.trainer import train_one_epoch

__all__ = ["train_one_epoch", "evaluate"]
