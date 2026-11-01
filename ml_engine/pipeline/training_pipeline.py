"""
ML Training Pipeline Orchestrator
---------------------------------
Orchestrates end-to-end dataset verification, model training across architectures
(ResNet-101, DenseNet-121, EfficientNet-B3, Stacking Ensemble), performance evaluation,
and automated quantized TFLite flatbuffer compilation.
"""

import os
import logging
from pathlib import Path

from ml_engine.constants import DEFAULT_BATCH_SIZE
from ml_engine.scripts.download_dataset import verify_dataset, DEFAULT_DATASET_DIR
from ml_engine.pipeline.data_augmentation import create_data_generators
from ml_engine.pipeline.class_weights import compute_balanced_class_weights
from ml_engine.pipeline.model_trainer import ModelTrainer
from ml_engine.export.tflite import convert_h5_to_tflite

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUPPORTED_MODELS = ["resnet101", "densenet121", "efficientnetb3", "ensemble"]


def run_training_pipeline(model_name, epochs=20, batch_size=DEFAULT_BATCH_SIZE, dataset_dir=DEFAULT_DATASET_DIR, output_dir=None):
    """
    Executes training of specified architecture and exports quantized TFLite.
    
    Args:
        model_name (str): One of ['resnet101', 'densenet121', 'efficientnetb3', 'ensemble']
        epochs (int): Number of training epochs.
        batch_size (int): Image generator batch size.
        dataset_dir (str or Path): Path to structured dataset base_dir.
        output_dir (str or Path, optional): Directory to save .h5 and .tflite models.
        
    Returns:
        dict: Summary of artifacts produced.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Must be one of {SUPPORTED_MODELS}")

    if output_dir is None:
        output_dir = REPO_ROOT / "trained_models"
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Verify Dataset
    _, ready = verify_dataset(dataset_dir)
    if not ready:
        raise RuntimeError("Dataset verification failed! Please run 'python ml_engine/scripts/download_dataset.py' to configure data.")

    # 2. Setup Generators & Class Weights (Distinct Single-Responsibility Modules)
    train_gen, val_gen, test_gen = create_data_generators(dataset_dir, batch_size)
    class_weights = compute_balanced_class_weights(train_gen)

    # 3. Model Routing & Execution
    trainer = ModelTrainer(patience=3)
    artifacts = {}

    if model_name == "resnet101":
        from ml_engine.architectures.resnet import build_resnet101
        logger.info("--- Building ResNet-101 Model ---")
        model = build_resnet101()
        save_path = output_dir / "resnet101.h5"
        trainer.train_and_save(model, train_gen, val_gen, class_weights=class_weights, save_path=str(save_path), epochs=epochs)
        tflite_path = convert_h5_to_tflite(save_path, output_dir / "resnet101.tflite")
        artifacts["h5"] = str(save_path)
        artifacts["tflite"] = str(tflite_path)
        logger.info(f"ResNet-101 ready: {tflite_path}")

    elif model_name == "densenet121":
        from ml_engine.architectures.densenet import build_densenet121
        logger.info("--- Building DenseNet-121 Model ---")
        model = build_densenet121()
        save_path = output_dir / "densenet121.h5"
        trainer.train_and_save(model, train_gen, val_gen, class_weights=class_weights, save_path=str(save_path), epochs=epochs)
        tflite_path = convert_h5_to_tflite(save_path, output_dir / "densenet121.tflite")
        artifacts["h5"] = str(save_path)
        artifacts["tflite"] = str(tflite_path)
        logger.info(f"DenseNet-121 ready: {tflite_path}")

    elif model_name == "efficientnetb3":
        from ml_engine.architectures.efficientnet import build_efficientnetb3
        logger.info("--- Building EfficientNet-B3 Model ---")
        model = build_efficientnetb3()
        save_path = output_dir / "efficientnetb3.h5"
        trainer.train_and_save(model, train_gen, val_gen, class_weights=class_weights, save_path=str(save_path), epochs=epochs)
        tflite_path = convert_h5_to_tflite(save_path, output_dir / "efficientnetb3.tflite")
        artifacts["h5"] = str(save_path)
        artifacts["tflite"] = str(tflite_path)
        logger.info(f"EfficientNet-B3 ready: {tflite_path}")

    elif model_name == "ensemble":
        from ml_engine.pipeline.ensemble_training import train_base_models, stack_and_train_ensemble
        from ml_engine.pipeline.model_evaluation import evaluate_ensemble_model

        logger.info("--- Running Full 3-Model Ensemble Training ---")
        base_models = train_base_models(train_gen, val_gen, class_weights=class_weights, epochs=epochs, save_dir=str(output_dir))
        meta_model = stack_and_train_ensemble(base_models, val_gen)
        evaluate_ensemble_model(base_models, meta_model, test_gen)

        # Convert each base model to quantized TFLite
        res_tflite = convert_h5_to_tflite(output_dir / "resnet101.h5", output_dir / "resnet101.tflite")
        dense_tflite = convert_h5_to_tflite(output_dir / "densenet121.h5", output_dir / "densenet121.tflite")
        eff_tflite = convert_h5_to_tflite(output_dir / "efficientnetb3.h5", output_dir / "efficientnetb3.tflite")
        artifacts["ensemble_tflite"] = [str(res_tflite), str(dense_tflite), str(eff_tflite)]
        logger.info("All 3 Ensemble TFLite flatbuffers successfully generated!")

    logger.info("🎉 Pipeline execution completed successfully!")
    return artifacts
