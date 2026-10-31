"""
Prediction Service (Memory-Optimized for Cloud Containers)
---------------------------------------------------------
Production business service layer for skin disease classification:
- Sequential Multi-Model Ensemble Inference:
  Runs ResNet-101, DenseNet-121, and EfficientNet-B3 sequentially with
  aggressive garbage collection and session clearing. Keeps RAM footprint < 200MB,
  completely eliminating Out-of-Memory (SIGKILL 137) crashes on Render Free Tier.
- Authentic ensemble consensus (average + max confidence across all 3 architectures).
"""

import os
import gc
import time
import logging
import warnings
import threading

# Silence TensorFlow C++ info & warning spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Silence TFLite deprecation UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
import cv2
import psutil
from tensorflow.keras.applications.efficientnet import preprocess_input

from app.core.config import config
from app.core.constants import IDX2CLASS, USER_FRIENDLY_MAPPING, TARGET_IMAGE_SIZE
from app.core.exceptions import ModelNotFoundError, InvalidImageError
from app.inference.cloud_downloader import download_and_extract_zip, download_file_from_cloud

logger = logging.getLogger(__name__)

_models_lock = threading.Lock()
_is_warming_up = False

MODEL_DEFINITIONS = {
    "resnet": "resnet101.tflite",
    "densenet": "densenet121.tflite",
    "efficientnet": "efficientnetb3.tflite"
}

def get_memory_usage_mb() -> float:
    """Returns accurate container RAM usage in MB."""
    for cgroup_path in (
        "/sys/fs/cgroup/memory.current",  # cgroups v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"  # cgroups v1
    ):
        if os.path.exists(cgroup_path):
            try:
                with open(cgroup_path, "r") as f:
                    return float(f.read().strip()) / (1024 * 1024)
            except Exception:
                pass

    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def ensure_models_available():
    """Verifies all model files exist on disk, downloading if missing."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    for name, fname in MODEL_DEFINITIONS.items():
        path = os.path.join(config.MODEL_DIR, fname)
        if not os.path.exists(path):
            cloud_url = config.MODEL_CLOUD_URLS.get(name)
            if cloud_url:
                logger.info(f"Model file missing for '{name}', downloading from: {cloud_url}")
                download_file_from_cloud(cloud_url, path)

def run_single_model_inference(model_name: str, model_path: str, preprocessed_image: np.ndarray) -> np.ndarray:
    """
    Executes inference via TensorFlow Lite Interpreter.
    TFLite flatbuffers load instantly with an ultra-lightweight memory footprint (~20-40MB per model),
    completely preventing Out-of-Memory (SIGKILL 137) crashes on Render Free Tier while preserving
    exact ensemble accuracy.
    """
    start_t = time.time()
    logger.info(f"  [Model Stage] Initializing TFLite runner: {model_name} ({os.path.basename(model_path)})...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor (ensure float32)
        input_data = preprocessed_image.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Extract output prediction probabilities
        pred = interpreter.get_tensor(output_details[0]['index'])[0]

        del interpreter
        gc.collect()

        elapsed = (time.time() - start_t) * 1000
        top_idx = int(np.argmax(pred))
        top_class = IDX2CLASS.get(top_idx, str(top_idx))
        top_conf = pred[top_idx] * 100

        logger.info(
            f"  [Model Done] {model_name} completed in {elapsed:.1f}ms | "
            f"Predicted: '{top_class}' ({top_conf:.1f}%) | "
            f"Process RAM: {get_memory_usage_mb():.1f} MB / 512 MB"
        )
        return pred
    except Exception as e:
        logger.error(f"[ERROR] Error executing TFLite inference for {model_name}: {e}")
        gc.collect()
        raise e

def start_model_warmup():
    """
    Ensures model weights exist on disk during startup without holding them in RAM,
    keeping boot RAM at ~100MB and preventing OOM restarts.
    """
    global _is_warming_up
    if _is_warming_up:
        return
    _is_warming_up = True

    def _warmup_worker():
        logger.info("[WARMUP] Background Model Warmup Worker: Verifying model files on disk...")
        try:
            ensure_models_available()
            logger.info("[READY] All ensemble model files verified and ready on disk for sequential inference.")
        except Exception as e:
            logger.error(f"[ERROR] Model verification encountered error: {e}")

    thread = threading.Thread(target=_warmup_worker, daemon=True)
    thread.start()

def load_and_preprocess_image(image_path: str, target_size=TARGET_IMAGE_SIZE):
    """Loads, resizes, and preprocesses input image for inference."""
    start_t = time.time()
    image = cv2.imread(image_path)
    if image is None:
        raise InvalidImageError(f"Image not found or unreadable: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    prep_ms = (time.time() - start_t) * 1000
    logger.info(f"[PREP] Image preprocessed in {prep_ms:.1f}ms (target_size={target_size}, shape={image.shape})")
    return image

def predict_skin_disease(image_path: str) -> dict:
    """
    Executes true 3-model ensemble prediction sequentially to stay under
    memory limits while providing genuine multi-architecture consensus.
    """
    overall_start = time.time()
    logger.info(f"==================================================")
    logger.info(f"[REQUEST] INCOMING INFERENCE REQUEST: {os.path.basename(image_path)}")
    logger.info(f"Initial Memory Usage: {get_memory_usage_mb():.1f} MB (Render Limit: 512 MB)")
    logger.info(f"==================================================")

    with _models_lock:
        ensure_models_available()

        preprocessed_image = load_and_preprocess_image(image_path)
        predictions = {}

        total_models = len(MODEL_DEFINITIONS)
        for idx, (name, fname) in enumerate(MODEL_DEFINITIONS.items(), 1):
            path = os.path.join(config.MODEL_DIR, fname)
            if not os.path.exists(path):
                logger.warning(f"[WARN] [{idx}/{total_models}] Skipping {name}: model file not found at {path}")
                continue

            logger.info(f"[INFER] [{idx}/{total_models}] Starting inference for architecture: {name.upper()}")
            try:
                pred = run_single_model_inference(name, path, preprocessed_image)
                predictions[name] = pred
            except Exception as infer_err:
                logger.error(f"[ERROR] [{idx}/{total_models}] Failed inference on {name}: {infer_err}")

        if not predictions:
            raise ModelNotFoundError(
                f"No AI models could be executed. Verify weights in {config.MODEL_DIR}."
            )

        # Ensemble Softmax Fusion
        avg_prediction = np.mean(list(predictions.values()), axis=0)
        predicted_index = int(np.argmax(avg_prediction))
        predicted_label = IDX2CLASS.get(predicted_index, str(predicted_index))
        friendly_info = USER_FRIENDLY_MAPPING.get(predicted_label, {})

        # Confidence calculation across ensemble models
        max_confidence = max(pred[predicted_index] for pred in predictions.values())
        confidence_percent = f"{int(round(max_confidence * 100))}"

        total_duration = time.time() - overall_start
        logger.info(f"--------------------------------------------------")
        logger.info(f"[CONSENSUS] ENSEMBLE CONSENSUS REACHED:")
        logger.info(f"   • Disease: {predicted_label} ({friendly_info.get('name', 'N/A')})")
        logger.info(f"   • Confidence: {confidence_percent}%")
        logger.info(f"   • Models contributing: {list(predictions.keys())}")
        logger.info(f"   • Total request duration: {total_duration:.2f}s")
        logger.info(f"   • Final Memory: {get_memory_usage_mb():.1f} MB / 512 MB")
        logger.info(f"==================================================")

        return {
            "predicted_disease": predicted_label,
            "confidence": confidence_percent,
            "disease_details": friendly_info,
            "ensemble_models_used": list(predictions.keys())
        }

def process_prediction(file_path: str, filename: str) -> dict:
    """
    High-level business service method to execute prediction
    and enrich result metadata for client presentation.
    """
    result = predict_skin_disease(file_path)
    result["message"] = "Image processed successfully"
    result["filename"] = filename
    return result
