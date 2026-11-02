"""
HAM10000 Dataset Verifier & Downloader
---------------------------------------
Verifies dataset integrity in ml_engine/data/skin_disease_dataset/base_dir/
and provides automated retrieval guidance, Kaggle API integration,
and automatic data preparation for the 7 diagnostic classes:
akiec, bcc, bkl, df, mel, nv, vasc.
"""

import os
import sys
import shutil
import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "ml_engine" / "data" / "skin_disease_dataset" / "base_dir"
DEFAULT_RAW_DIR = REPO_ROOT / "ml_engine" / "data" / "raw"
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


def check_kaggle_credentials():
    """Checks whether Kaggle credentials exist in environment or ~/.kaggle/kaggle.json."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    home = Path.home()
    kaggle_file = home / ".kaggle" / "kaggle.json"
    if kaggle_file.is_file():
        return True

    return False


def print_kaggle_setup_instructions(target_dir=DEFAULT_DATASET_DIR):
    """Prints step-by-step instructions for Kaggle dataset retrieval."""
    logger.info("\n" + "=" * 68)
    logger.info("📥 HAM10000 Dataset Download & Preparation Guide")
    logger.info("=" * 68)
    logger.info("The HAM10000 dataset is hosted on Kaggle (~2.6 GB, 10,015 images).")
    logger.info("Choose one of the two options below to configure the dataset:\n")
    logger.info("🔷 Option 1: Automated Download via Kaggle API / CLI (Recommended)")
    logger.info("   1. Log in to Kaggle: https://www.kaggle.com")
    logger.info("   2. Go to Account Settings -> API -> click 'Create New Token'")
    logger.info("      (This downloads 'kaggle.json')")
    logger.info("   3. Place 'kaggle.json' in: ~/.kaggle/kaggle.json")
    logger.info("      On Windows: %USERPROFILE%\\.kaggle\\kaggle.json")
    logger.info("   4. Run the automated downloader & preprocessor:")
    logger.info("      python ml_engine/scripts/download_dataset.py --download --prepare")
    logger.info("      (or inside Docker: docker compose run --rm ml-engine python ml_engine/scripts/download_dataset.py --download --prepare)\n")
    logger.info("--------------------------------------------------------------------")
    logger.info("🔷 Option 2: Direct Browser Download (No CLI setup needed)")
    logger.info("   1. Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    logger.info("   2. Click 'Download (2.6 GB)' and extract the archive into:")
    logger.info("      ml_engine/data/raw/")
    logger.info("   3. Run the automated split & stratify script:")
    logger.info("      python ml_engine/pipeline/prepare_dataset.py --raw-dir ml_engine/data/raw")
    logger.info("=" * 68 + "\n")


def download_from_kaggle(raw_dir=DEFAULT_RAW_DIR):
    """Downloads HAM10000 dataset via Kaggle CLI / Python API."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if not check_kaggle_credentials():
        logger.error("❌ Kaggle credentials not found!")
        print_kaggle_setup_instructions()
        return False

    logger.info(f"⏳ Initiating automated Kaggle download to {raw_path}...")
    try:
        # Try Kaggle Python API first
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "kmader/skin-cancer-mnist-ham10000",
            path=str(raw_path),
            unzip=True,
            quiet=False
        )
        logger.info("✅ Kaggle dataset downloaded and extracted successfully.")
        return True
    except ImportError:
        # Fallback to kaggle CLI subprocess
        try:
            logger.info("Kaggle Python module not found, invoking 'kaggle' CLI command...")
            cmd = ["kaggle", "datasets", "download", "-d", "kmader/skin-cancer-mnist-ham10000", "-p", str(raw_path), "--unzip"]
            subprocess.run(cmd, check=True)
            logger.info("✅ Kaggle CLI download complete.")
            return True
        except Exception as e:
            logger.error(f"Failed to execute Kaggle CLI download: {e}")
            return False
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify, download, and prepare HAM10000 dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to structured skin_disease_dataset/base_dir"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Path to raw download directory"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Trigger automated download from Kaggle if credentials are configured"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Automatically trigger dataset stratification into train/val/test after download"
    )
    args = parser.parse_args()

    summary, ready = verify_dataset(args.dataset_dir)

    if ready:
        logger.info("✅ Dataset is already verified, complete, and ready for model training!")
        sys.exit(0)

    if args.download:
        success = download_from_kaggle(args.raw_dir)
        if success and args.prepare:
            from ml_engine.pipeline.prepare_dataset import prepare_dataset
            logger.info("🔄 Running automated stratified data preparation...")
            prepare_dataset(args.raw_dir, args.dataset_dir)
            _, ready_after = verify_dataset(args.dataset_dir)
            if ready_after:
                logger.info("🎉 Dataset successfully downloaded, prepared, and verified!")
                sys.exit(0)
        sys.exit(0 if success else 1)

    print_kaggle_setup_instructions(args.dataset_dir)
    sys.exit(1)
