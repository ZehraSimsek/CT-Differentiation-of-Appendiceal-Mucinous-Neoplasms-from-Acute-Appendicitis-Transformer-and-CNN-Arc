"""
data — Data Sub-package
========================
Exposes the dataset classes, patient-level splitting, dummy generators,
and transform builders.
"""

from data.dataset import CustomCTDataset, generate_dummy_data, generate_dummy_patient_data
from data.patient_dataset import PatientCTDataset
from data.splitter import DataSplits, FoldSplit, create_data_splits
from data.transforms import get_train_transforms, get_val_transforms

__all__ = [
    # Original (backward-compatible)
    "CustomCTDataset",
    "generate_dummy_data",
    # Patient-level pipeline
    "PatientCTDataset",
    "generate_dummy_patient_data",
    "create_data_splits",
    "DataSplits",
    "FoldSplit",
    # Transforms
    "get_train_transforms",
    "get_val_transforms",
]
