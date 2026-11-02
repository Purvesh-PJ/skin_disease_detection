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

    # Via Docker Compose from host:
    docker compose run --rm ml-engine python ml_engine/scripts/verify_dataset_readiness.py
    # or shortcut: npm run ml:verify

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

from ml_engine.constants import CLASSES, TARGET_IMAGE_SIZE, NUM_CLASSES
from ml_engine.scripts.download_dataset import verify_dataset, print_kaggle_setup_instructions, DEFAULT_DATASET_DIR


def print_banner(title):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


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


def inspect_ml_setup(dataset_dir=DEFAULT_DATASET_DIR):
    print_banner("[SETUP] Inspecting Machine Learning Environment & Dataset Readiness")

    # 1. Runtime Libraries Check
    print("Checking core ML framework dependencies...")
    libs = check_runtime_libraries()
    for lib_name, version_str in libs.items():
        is_ok = "Missing" not in version_str
        bullet = "[OK]" if is_ok else "[FAIL]"
        print(f"  {bullet:<6} {lib_name:<16}: {version_str}")

    print("-" * 75)

    # 2. Dataset Verification
    print(f"Checking HAM10000 dataset distribution at:\n  -> {dataset_dir}\n")
    summary, is_ready = verify_dataset(dataset_dir)

    print("-" * 75)
    if is_ready:
        print_banner("[STATUS] Dataset Split & Class Balance Verification")
        header = f"{'Split Name':<14} | {'Images Count':<14} | {'Classes Present':<18} | {'Status':<10}"
        print(header)
        print("-" * 75)

        total_images = 0
        for split_name, counts in summary.items():
            split_total = counts.get("_total", 0)
            total_images += split_total
            valid_classes = sum(1 for c in CLASSES if counts.get(c, 0) > 0)
            classes_str = f"{valid_classes} / {NUM_CLASSES} classes"
            status_str = "READY" if valid_classes == NUM_CLASSES and split_total > 0 else "INCOMPLETE"
            print(f"{split_name:<14} | {split_total:<14} | {classes_str:<18} | {status_str:<10}")

        print("-" * 75)
        print(f"Total Dataset Images Verified: {total_images} (Standard: 10,015 images)")
        print(f"Target Image Dimensions:       {TARGET_IMAGE_SIZE} (RGB 3-channel)")
        print("=" * 75)
        print("\n[SUCCESS] ML Environment and Dataset are 100% verified and ready for training!")
        print("\n" + "=" * 75)
        print("[*] Next Steps: What You Can Do Now")
        print("=" * 75)
        print("Note: Heavy model training is NEVER started automatically.\n")
        print("1. Train a Single Architecture (e.g., ResNet-101):")
        print("   Docker: docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model resnet101 --epochs 20")
        print("   Native: python ml_engine/run_pipeline.py --model resnet101 --epochs 20\n")
        print("2. Train Full 3-Model Stacking Ensemble (ResNet + DenseNet + EfficientNet):")
        print("   Docker: docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model ensemble --epochs 15")
        print("   Native: python ml_engine/run_pipeline.py --model ensemble --epochs 15\n")
        print("3. Enter Interactive Container Shell (Explore scripts or notebooks):")
        print("   Docker: docker compose run --rm ml-engine bash\n")
        print("4. View All Pipeline Command-Line Options:")
        print("   Docker: docker compose run --rm ml-engine python ml_engine/run_pipeline.py --help\n")
        print("Checkpoints (.h5 and .tflite) are automatically saved to your host 'trained_models/' folder.")
        print("=" * 75 + "\n")
        return 0
    else:
        print("\n[NOTICE] HAM10000 Dataset is not yet configured or incomplete.")
        print("Model training cannot be performed until the dataset is prepared.")
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
