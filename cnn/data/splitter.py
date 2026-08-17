"""
data/splitter.py — Patient-Level Stratified Splitting & 5-Fold CV
==================================================================
Orchestrates all data-splitting logic at the **Patient ID** level to
prevent data leakage in clinical machine learning.

Workflow
--------
    1. Hold-out test split — 15 % of unique Patient IDs, stratified by class.
    2. 5-Fold Stratified Group K-Fold — remaining 85 % of patients split
       into ~85 % train / ~15 % val per fold, with strict patient grouping.

All images belonging to the same patient are guaranteed to reside in
exactly one partition (train, val, or test) — never split across sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from config import PipelineConfig


# =========================================================================== #
#  DATA STRUCTURES                                                              #
# =========================================================================== #

@dataclass
class FoldSplit:
    """Holds patient-level split information for a single CV fold."""

    fold: int
    train_patient_ids: list[str]
    val_patient_ids: list[str]


@dataclass
class DataSplits:
    """Complete splitting result: hold-out test + per-fold train/val."""

    test_patient_ids: list[str]
    folds: list[FoldSplit] = field(default_factory=list)


# =========================================================================== #
#  MAIN SPLITTER                                                                #
# =========================================================================== #

def create_data_splits(
    config: PipelineConfig,
    patient_labels: dict[str, int],
) -> DataSplits:
    """
    Create patient-level data splits: hold-out test + stratified K-fold CV.

    Parameters
    ----------
    config : PipelineConfig
        Supplies ``test_size``, ``n_folds``, and ``seed``.
    patient_labels : dict[str, int]
        Mapping from patient ID → class label (0 or 1).

    Returns
    -------
    DataSplits
        Contains ``test_patient_ids`` and a list of ``FoldSplit`` objects.

    Raises
    ------
    ValueError
        If there are too few patients for the requested splits.
    """
    # Sorted for reproducibility
    all_pids = sorted(patient_labels.keys())
    all_labels = np.array([patient_labels[pid] for pid in all_pids])

    n_patients = len(all_pids)
    n_positive = int(all_labels.sum())
    n_negative = n_patients - n_positive

    print(f"\n  ── Patient Registry ──")
    print(f"  Total patients    : {n_patients}")
    print(f"  Positive (Musinoz): {n_positive}")
    print(f"  Negative (Appendisit): {n_negative}")

    # ====================================================================
    # Step 1: Hold-out Test Split (stratified, patient-level)
    # ====================================================================
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.seed,
    )
    dev_indices, test_indices = next(sss.split(all_pids, all_labels))

    test_pids = [all_pids[i] for i in test_indices]
    dev_pids = [all_pids[i] for i in dev_indices]
    dev_labels = all_labels[dev_indices]

    print(f"\n  ── Hold-out Test Split ({config.test_size:.0%}) ──")
    print(f"  Test patients     : {len(test_pids)}")
    print(
        f"    Positive        : "
        f"{sum(1 for pid in test_pids if patient_labels[pid] == 1)}"
    )
    print(
        f"    Negative        : "
        f"{sum(1 for pid in test_pids if patient_labels[pid] == 0)}"
    )
    print(f"  Dev patients      : {len(dev_pids)} (for {config.n_folds}-fold CV)")

    # ====================================================================
    # Step 2: Stratified Group K-Fold on dev patients
    # ====================================================================
    # For StratifiedGroupKFold: X can be anything array-like of len(dev),
    # y = labels, groups = patient IDs (each image grouped by patient)
    # Since we're splitting at patient level, groups == indices.
    sgkf = StratifiedGroupKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed,
    )

    # We use dev_pids (the .h5 filenames) as the group tokens for StratifiedGroupKFold
    dev_groups = np.array(dev_pids)

    folds: list[FoldSplit] = []

    print(f"\n  ── {config.n_folds}-Fold Cross-Validation Splits ──")

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(dev_pids, dev_labels, groups=dev_groups), start=1
    ):
        train_pids = [dev_pids[i] for i in train_idx]
        val_pids = [dev_pids[i] for i in val_idx]

        # --- Leakage assertion ---
        train_set = set(train_pids)
        val_set = set(val_pids)
        test_set = set(test_pids)

        assert train_set.isdisjoint(val_set), (
            f"Fold {fold_idx}: patient leakage between train and val!"
        )
        assert train_set.isdisjoint(test_set), (
            f"Fold {fold_idx}: patient leakage between train and test!"
        )
        assert val_set.isdisjoint(test_set), (
            f"Fold {fold_idx}: patient leakage between val and test!"
        )

        n_train_pos = sum(1 for p in train_pids if patient_labels[p] == 1)
        n_val_pos = sum(1 for p in val_pids if patient_labels[p] == 1)

        print(
            f"  Fold {fold_idx}  │  "
            f"Train: {len(train_pids)} patients ({n_train_pos} pos)  │  "
            f"Val: {len(val_pids)} patients ({n_val_pos} pos)  │  "
            f"Leakage: ✗ None"
        )

        folds.append(FoldSplit(
            fold=fold_idx,
            train_patient_ids=train_pids,
            val_patient_ids=val_pids,
        ))

    return DataSplits(test_patient_ids=test_pids, folds=folds)
