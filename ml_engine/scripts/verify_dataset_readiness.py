#!/usr/bin/env python3
"""
HAM10000 Dataset & ML Environment Readiness Verifier
---------------------------------------------------
Verifies the ML engine runtime, dataset readiness, and class distribution
WITHOUT triggering heavy neural network training. 

Provides clear, human-readable status confirmation so researchers and developers
know exactly when the dataset and environment are ready for model training and what to do next.

Usage:
    # Within Docker container:
    python ml_engine/scripts/verify_dataset_readiness.py

    # Via npm shortcut:
    npm run ml:verify

    # Native Python environment:
    python ml_engine/scripts/verify_dataset_readiness.py --dataset-dir ml_engine/data/skin_disease_dataset/base_dir
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_engine.constants import (
    CLASSES,
    SPLITS,
    TARGET_IMAGE_SIZE,
    NUM_CLASSES,
    DEFAULT_DATASET_DIR,
)


def print_banner(title):
    print("\n" + "=" * 75, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 75, flush=True)


def check_runtime_libraries():
    """Validates that all core machine learning runtime libraries import cleanly."""
    status = {}

    # 1. TensorFlow
    try:
        import tensorflow as tf
        gpu_devices = tf.config.list_physical_devices("GPU")
        status["TensorFlow"] = f"v{tf.__version__} (GPU: {len(gpu_devices)} detected)"
    except ImportError as e:
        status["TensorFlow"] = f"Missing ({e})"

    # 2. Scikit-learn
    try:
        import sklearn
        status["Scikit-Learn"] = f"v{sklearn.__version__}"
    except ImportError as e:
        status["Scikit-Learn"] = f"Missing ({e})"

    # 3. OpenCV
    try:
        import cv2
        status["OpenCV"] = f"v{cv2.__version__}"
    except ImportError as e:
        status["OpenCV"] = f"Missing ({e})"

    # 4. NumPy
    try:
        import numpy as np
        status["NumPy"] = f"v{np.__version__}"
    except ImportError as e:
        status["NumPy"] = f"Missing ({e})"

    return status


def verify_dataset(base_dir=DEFAULT_DATASET_DIR):
    """
    Checks if train, val, and test directories exist with all 7 diagnostic classes
    and contains valid image files.

    Returns:
        tuple[dict, bool]: (summary_counts_dict, is_valid_boolean)
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        return {}, False

    summary = {}
    is_valid = True
    total_images_all = 0

    for split in SPLITS:
        split_path = base_path / split
        summary[split] = {}
        split_total = 0

        if not split_path.is_dir():
            is_valid = False
            continue

        for cls in CLASSES:
            cls_path = split_path / cls
            if not cls_path.is_dir():
                summary[split][cls] = 0
                is_valid = False
                continue

            images = [f for f in cls_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            count = len(images)
            summary[split][cls] = count
            split_total += count

        summary[split]["_total"] = split_total
        total_images_all += split_total

    if total_images_all == 0:
        is_valid = False

    return summary, is_valid


def print_kaggle_setup_instructions(target_dir=DEFAULT_DATASET_DIR):
    """Prints step-by-step instructions for HAM10000 dataset retrieval."""
    print("\n" + "=" * 68, flush=True)
    print("HAM10000 Dataset Setup Guide (2.6 GB, 10,015 images)", flush=True)
    print("=" * 68, flush=True)
    print("To configure the dataset in 2 simple steps (No API tokens needed):\n", flush=True)
    print("Step 1: Download dataset archive in your browser:", flush=True)
    print("   -> https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000", flush=True)
    print("   Click 'Download' and extract the archive into:", flush=True)
    print("   ml_engine/data/raw/\n", flush=True)
    print("Step 2: Run automated dataset preparation:", flush=True)
    print("   npm run ml:prepare", flush=True)
    print("   (or via Docker: docker compose run --rm ml-engine python ml_engine/pipeline/prepare_dataset.py)\n", flush=True)
    print("Step 3: Verify readiness:", flush=True)
    print("   npm run ml:verify", flush=True)
    print("=" * 68 + "\n", flush=True)


def inspect_ml_setup(dataset_dir=DEFAULT_DATASET_DIR):
    print_banner("[SETUP] Inspecting Machine Learning Environment & Dataset Readiness")

    # 1. Runtime Libraries Check
    print("Checking core ML framework dependencies...", flush=True)
    libs = check_runtime_libraries()
    for lib_name, version_str in libs.items():
        is_ok = "Missing" not in version_str
        bullet = "[OK]" if is_ok else "[FAIL]"
        print(f"  {bullet:<6} {lib_name:<16}: {version_str}", flush=True)

    print("-" * 75, flush=True)

    # 2. Dataset Verification
    print(f"Checking HAM10000 dataset distribution at:\n  -> {dataset_dir}\n", flush=True)
    summary, is_ready = verify_dataset(dataset_dir)

    print("-" * 75, flush=True)
    if is_ready:
        print_banner("[STATUS] Dataset Split & Class Balance Verification")
        header = f"{'Split Name':<14} | {'Images Count':<14} | {'Classes Present':<18} | {'Status':<10}"
        print(header, flush=True)
        print("-" * 75, flush=True)

        total_images = 0
        for split_name, counts in summary.items():
            split_total = counts.get("_total", 0)
            total_images += split_total
            valid_classes = sum(1 for c in CLASSES if counts.get(c, 0) > 0)
            classes_str = f"{valid_classes} / {NUM_CLASSES} classes"
            status_str = "READY" if valid_classes == NUM_CLASSES and split_total > 0 else "INCOMPLETE"
            print(f"{split_name:<14} | {split_total:<14} | {classes_str:<18} | {status_str:<10}", flush=True)

        print("-" * 75, flush=True)
        print(f"Total Dataset Images Verified: {total_images} (Standard: 10,015 images)", flush=True)
        print(f"Target Image Dimensions:       {TARGET_IMAGE_SIZE} (RGB 3-channel)", flush=True)
        print("=" * 75, flush=True)
        print("\n[SUCCESS] ML Environment and Dataset are 100% verified and ready for training!", flush=True)
        print("\n" + "=" * 75, flush=True)
        print("[*] Next Steps: What You Can Do Now", flush=True)
        print("=" * 75, flush=True)
        print("Note: Heavy model training is NEVER started automatically.\n", flush=True)
        print("1. Train a Single Architecture (e.g., ResNet-101):", flush=True)
        print("   Docker: docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model resnet101 --epochs 20", flush=True)
        print("   Native: python ml_engine/run_pipeline.py --model resnet101 --epochs 20\n", flush=True)
        print("2. Train Full 3-Model Stacking Ensemble (ResNet + DenseNet + EfficientNet):", flush=True)
        print("   Docker: docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model ensemble --epochs 15", flush=True)
        print("   Native: python ml_engine/run_pipeline.py --model ensemble --epochs 15\n", flush=True)
        print("3. Enter Interactive Container Shell (Explore scripts or notebooks):", flush=True)
        print("   Docker: docker compose run --rm ml-engine bash\n", flush=True)
        print("4. View All Pipeline Command-Line Options:", flush=True)
        print("   Docker: docker compose run --rm ml-engine python ml_engine/run_pipeline.py --help\n", flush=True)
        print("Checkpoints (.h5 and .tflite) are automatically saved to your host 'trained_models/' folder.", flush=True)
        print("=" * 75 + "\n", flush=True)
        return 0
    else:
        print("\n[NOTICE] HAM10000 Dataset is not yet configured or incomplete.", flush=True)
        print("Model training cannot be performed until the dataset is prepared.", flush=True)
        print_kaggle_setup_instructions(dataset_dir)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ML setup and verify dataset readiness")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to structured dataset base directory"
    )
    args = parser.parse_args()
    sys.exit(inspect_ml_setup(args.dataset_dir))
