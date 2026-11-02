"""
ML Engine Constants & Hyperparameters
-------------------------------------
Single source of truth for model architectures, input dimensions,
and dataset configurations within the ML training engine.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "ml_engine" / "data" / "skin_disease_dataset" / "base_dir"
DEFAULT_RAW_DIR = REPO_ROOT / "ml_engine" / "data" / "raw"
SPLITS = ["train_dir", "val_dir", "test_dir"]

TARGET_IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*TARGET_IMAGE_SIZE, 3)
DEFAULT_BATCH_SIZE = 64
NUM_CLASSES = 7

# Diagnostic Classes (HAM10000)
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_INDICES = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}
