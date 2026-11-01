"""
TFLite Model Quantization & Exporter
------------------------------------
Converts trained Keras (.h5 / SavedModel) neural network checkpoints
into lightweight, quantized TensorFlow Lite flatbuffers (.tflite).
Enforces Dynamic Range Quantization to reduce model sizes by ~70%
and ensure runtime inference strictly abides by Render's 512 MB memory limit.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "trained_models"

def convert_h5_to_tflite(h5_model_path, output_tflite_path=None, quantize=True):
    """
    Converts a Keras .h5 model file to a quantized .tflite flatbuffer.

    Args:
        h5_model_path (str or Path): Path to the trained Keras .h5 checkpoint.
        output_tflite_path (str or Path, optional): Path where .tflite will be written.
            Defaults to replacing .h5 extension with .tflite in the same directory.
        quantize (bool): Whether to enable Dynamic Range Quantization. Defaults to True.

    Returns:
        Path: Absolute path to the generated .tflite model file.
    """
    h5_path = Path(h5_model_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {h5_path}")

    if output_tflite_path is None:
        output_tflite_path = h5_path.with_suffix(".tflite")
    else:
        output_tflite_path = Path(output_tflite_path)

    os.makedirs(output_tflite_path.parent, exist_ok=True)

    logger.info(f"Loading Keras model from: {h5_path}")
    import tensorflow as tf

    # Load Keras model
    model = tf.keras.models.load_model(str(h5_path), compile=False)

    # Initialize TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        logger.info("Applying Dynamic Range Quantization (Optimization: DEFAULT)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert model to flatbuffer
    logger.info("Converting model graph to TFLite flatbuffer...")
    tflite_model_bytes = converter.convert()

    # Save to disk
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model_bytes)

    h5_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
    tflite_size_mb = os.path.getsize(output_tflite_path) / (1024 * 1024)
    reduction = ((h5_size_mb - tflite_size_mb) / h5_size_mb) * 100 if h5_size_mb > 0 else 0

    logger.info(f"✅ Conversion Successful!")
    logger.info(f"   Original (.h5):    {h5_size_mb:.2f} MB")
    logger.info(f"   Quantized (.tflite): {tflite_size_mb:.2f} MB ({reduction:.1f}% size reduction)")
    logger.info(f"   Saved at:          {output_tflite_path}")

    return output_tflite_path

def convert_all_models(models_dir=DEFAULT_MODEL_DIR, quantize=True):
    """
    Scans a directory for all .h5 checkpoints and converts each to .tflite.
    """
    dir_path = Path(models_dir)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Model directory not found: {dir_path}")

    h5_files = list(dir_path.glob("*.h5"))
    if not h5_files:
        logger.warning(f"No .h5 checkpoints found in: {dir_path}")
        return []

    logger.info(f"Found {len(h5_files)} .h5 checkpoint(s) in {dir_path}. Starting batch conversion...")
    converted_paths = []
    for h5_file in h5_files:
        try:
            out_path = convert_h5_to_tflite(h5_file, quantize=quantize)
            converted_paths.append(out_path)
        except Exception as e:
            logger.error(f"Failed to convert {h5_file.name}: {e}")

    return converted_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras .h5 models to Quantized TFLite flatbuffers.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to specific .h5 model to convert. If omitted, converts all models in --models-dir."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to scan for .h5 models when --model-path is not specified."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Custom destination path for .tflite flatbuffer."
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable Dynamic Range Quantization."
    )

    args = parser.parse_args()

    quantize_flag = not args.no_quantize

    if args.model_path:
        convert_h5_to_tflite(args.model_path, args.output_path, quantize=quantize_flag)
    else:
        convert_all_models(args.models_dir, quantize=quantize_flag)
