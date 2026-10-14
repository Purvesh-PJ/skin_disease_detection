"""
TFLite Converter Script
-----------------------
Converts heavy Keras .h5 ensemble models into lightweight, highly optimized .tflite models.
Enables full 3-model ensemble inference on resource-constrained cloud containers (e.g. Render 512MB RAM).
"""

import os
import gc
import sys
import tensorflow as tf

MODELS = {
    "densenet": ("densenet121.h5", "densenet121.tflite"),
    "efficientnet": ("efficientnetb3.h5", "efficientnetb3.tflite"),
    "resnet": ("resnet101.h5", "resnet101.tflite")
}

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained_models"))

def convert_h5_to_tflite(h5_filename, tflite_filename):
    h5_path = os.path.join(MODELS_DIR, h5_filename)
    tflite_path = os.path.join(MODELS_DIR, tflite_filename)

    if not os.path.exists(h5_path):
        print(f"[ERROR] Source model not found: {h5_path}")
        return False

    print(f"\n==========================================")
    print(f"Converting {h5_filename} -> {tflite_filename}...")
    original_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
    print(f"Original .h5 size: {original_size_mb:.2f} MB")

    try:
        # Load keras model
        model = tf.keras.models.load_model(h5_path, compile=False)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Apply standard default optimizations (dynamic range quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()

        # Save tflite file
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        compression = (1 - (tflite_size_mb / original_size_mb)) * 100
        print(f"[SUCCESS] Saved {tflite_filename}: {tflite_size_mb:.2f} MB ({compression:.1f}% reduction!)")

        del model
        del tflite_model
        del converter
        tf.keras.backend.clear_session()
        gc.collect()
        return True

    except Exception as e:
        print(f"[ERROR] Failed to convert {h5_filename}: {e}", file=sys.stderr)
        tf.keras.backend.clear_session()
        gc.collect()
        return False

def main():
    print(f"Scanning models in: {MODELS_DIR}")
    for key, (h5_file, tflite_file) in MODELS.items():
        convert_h5_to_tflite(h5_file, tflite_file)
    print("\nAll models conversion finished.")

if __name__ == "__main__":
    main()
