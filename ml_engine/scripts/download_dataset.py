"""
HAM10000 Dataset Verifier & Downloader
---------------------------------------
Verifies dataset integrity in ml_engine/data/skin_disease_dataset/base_dir/
and provides automated retrieval guidance and validation for the 7 classes:
akiec, bcc, bkl, df, mel, nv, vasc.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "ml_engine" / "data" / "skin_disease_dataset" / "base_dir"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SPLITS = ["train_dir", "val_dir", "test_dir"]

def verify_dataset(base_dir=DEFAULT_DATASET_DIR):
    """
    Checks if train, val, and test directories exist with all 7 diagnostic classes
    and contains valid image files.

    Returns:
        dict: Summary of image counts per split and class.
        bool: True if dataset is complete and ready for training.
    """
    base_path = Path(base_dir)
    logger.info(f"Checking dataset integrity at: {base_path}")

    if not base_path.is_dir():
        logger.warning(f"Dataset base directory does not exist: {base_path}")
        return {}, False

    summary = {}
    is_valid = True
    total_images_all = 0

    for split in SPLITS:
        split_path = base_path / split
        summary[split] = {}
        split_total = 0

        if not split_path.is_dir():
            logger.warning(f"Missing split directory: {split_path}")
            is_valid = False
            continue

        for cls in CLASSES:
            cls_path = split_path / cls
            if not cls_path.is_dir():
                logger.warning(f"Missing class folder: {cls_path}")
                summary[split][cls] = 0
                is_valid = False
                continue

            images = [f for f in cls_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            count = len(images)
            summary[split][cls] = count
            split_total += count

        summary[split]["_total"] = split_total
        total_images_all += split_total

    logger.info("--- HAM10000 Dataset Status Summary ---")
    for split, counts in summary.items():
        total = counts.get("_total", 0)
        logger.info(f"📁 {split}: {total} images")
        for cls in CLASSES:
            logger.info(f"   - {cls:<8}: {counts.get(cls, 0)}")

    logger.info(f"Total dataset images found: {total_images_all}")

    if total_images_all == 0:
        is_valid = False

    return summary, is_valid

def print_kaggle_setup_instructions(target_dir=DEFAULT_DATASET_DIR):
    """
    Prints step-by-step instructions for Kaggle CLI 1-click dataset retrieval.
    """
    logger.info("\n" + "=" * 60)
    logger.info("📥 HAM10000 Dataset Download Options:")
    logger.info("=" * 60)
    logger.info("Option 1: Using Kaggle CLI (Recommended for fast local download)")
    logger.info("   1. Ensure kaggle is installed:  pip install kaggle")
    logger.info("   2. Download the dataset:        kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ml_engine/data/raw --unzip")
    logger.info("   3. Run automated preparation:   python ml_engine/pipeline/prepare_dataset.py --raw-dir ml_engine/data/raw")
    logger.info("-" * 60)
    logger.info("Option 2: Direct Kaggle Web Download")
    logger.info("   1. Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    logger.info("   2. Download and unzip files into: ml_engine/data/raw/")
    logger.info("   3. Run: python ml_engine/pipeline/prepare_dataset.py --raw-dir ml_engine/data/raw")
    logger.info("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify or guide download for HAM10000 dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to structured skin_disease_dataset/base_dir"
    )
    args = parser.parse_args()

    summary, ready = verify_dataset(args.dataset_dir)
    if not ready:
        print_kaggle_setup_instructions(args.dataset_dir)
        sys.exit(1)
    else:
        logger.info("✅ Dataset is verified, complete, and ready for model training!")
        sys.exit(0)

