"""
ML Engine Master CLI Entrypoint
-------------------------------
Single-command CLI to train individual models or the full ensemble locally,
evaluate performance against the test dataset, and automatically export
quantized .tflite flatbuffers.

Usage:
    # Train and export ResNet-101:
    python ml_engine/run_pipeline.py --model resnet101 --epochs 10

    # Train and export DenseNet-121:
    python ml_engine/run_pipeline.py --model densenet121 --epochs 10

    # Train and export EfficientNet-B3:
    python ml_engine/run_pipeline.py --model efficientnetb3 --epochs 10

    # Run full 3-model ensemble training & stacking:
    python ml_engine/run_pipeline.py --model ensemble --epochs 15
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_engine.constants import DEFAULT_BATCH_SIZE
from ml_engine.scripts.download_dataset import DEFAULT_DATASET_DIR
from ml_engine.pipeline.orchestrator import run_training_pipeline, SUPPORTED_MODELS


def main():
    parser = argparse.ArgumentParser(description="Master ML Training & TFLite Export Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet101",
        choices=SUPPORTED_MODELS,
        help="Model architecture to train (resnet101, densenet121, efficientnetb3, ensemble)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to structured skin_disease_dataset/base_dir"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "trained_models"),
        help="Directory where .h5 and .tflite checkpoints will be saved"
    )

    args = parser.parse_args()
    run_training_pipeline(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
