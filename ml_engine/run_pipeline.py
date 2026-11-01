"""
Master ML Training & TFLite Export Pipeline Runner
---------------------------------------------------
Single-command CLI to train individual models or the full ensemble locally,
evaluate performance against the test dataset, and automatically export
quantized .tflite flatbuffers to trained_models/.

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

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_engine.constants import TARGET_IMAGE_SIZE, DEFAULT_BATCH_SIZE, CLASSES
from ml_engine.data.download_dataset import verify_dataset, DEFAULT_DATASET_DIR
from ml_engine.pipeline.export_tflite import convert_h5_to_tflite

SUPPORTED_MODELS = ["resnet101", "densenet121", "efficientnetb3", "ensemble"]

def create_data_generators(dataset_dir=DEFAULT_DATASET_DIR, batch_size=DEFAULT_BATCH_SIZE):
    """
    Initializes train, validation, and test Keras ImageDataGenerators.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = Path(dataset_dir) / "train_dir"
    val_dir = Path(dataset_dir) / "val_dir"
    test_dir = Path(dataset_dir) / "test_dir"

    # Data augmentation on train
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    # Clean scaling for validation and test
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    logger.info("Initializing Data Generators...")
    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=TARGET_IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True
    )

    val_gen = eval_datagen.flow_from_directory(
        str(val_dir),
        target_size=TARGET_IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False
    )

    test_gen = eval_datagen.flow_from_directory(
        str(test_dir),
        target_size=TARGET_IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

def compute_balanced_class_weights(train_gen):
    """
    Computes balanced class weights to penalize dominant classes (like 'nv')
    and boost sensitivity on minority malignant conditions (like 'mel', 'bcc', 'akiec').
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = train_gen.classes
    unique_classes = np.unique(classes)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=classes)
    class_weights_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    logger.info(f"Computed Balanced Class Weights: {class_weights_dict}")
    return class_weights_dict

def run_training_pipeline(model_name, epochs=20, batch_size=DEFAULT_BATCH_SIZE, dataset_dir=DEFAULT_DATASET_DIR, output_dir=None):
    """
    Executes training of specified architecture and exports quantized TFLite.
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "trained_models"
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Verify Dataset
    _, ready = verify_dataset(dataset_dir)
    if not ready:
        raise RuntimeError("Dataset verification failed! Please run 'python ml_engine/data/download_dataset.py' to configure data.")

    # 2. Setup Generators & Class Weights
    train_gen, val_gen, test_gen = create_data_generators(dataset_dir, batch_size)
    class_weights = compute_balanced_class_weights(train_gen)

    # 3. Model Routing
    from ml_engine.pipeline.trainer import ModelTrainer

    trainer = ModelTrainer(patience=3)

    if model_name == "resnet101":
        from ml_engine.architectures.resnet import build_resnet101
        logger.info("--- Building ResNet-101 Model ---")
        model = build_resnet101()
        save_path = output_dir / "resnet101.h5"
        trainer.train_and_save(model, train_gen, val_gen, class_weights=class_weights, save_path=str(save_path), epochs=epochs)
        tflite_path = convert_h5_to_tflite(save_path, output_dir / "resnet101.tflite")
        logger.info(f"ResNet-101 ready: {tflite_path}")

    elif model_name == "densenet121":
        from ml_engine.architectures.densenet import build_densenet121
        logger.info("--- Building DenseNet-121 Model ---")
        model = build_densenet121()
        save_path = output_dir / "densenet121.h5"
        trainer.train_and_save(model, train_gen, val_gen, class_weights=class_weights, save_path=str(save_path), epochs=epochs)
        tflite_path = convert_h5_to_tflite(save_path, output_dir / "densenet121.tflite")
        logger.info(f"DenseNet-121 ready: {tflite_path}")

    elif model_name == "efficientnetb3":
        from ml_engine.architectures.efficientnet import build_efficientnetb3
        logger.info("--- Building EfficientNet-B3 Model ---")
        model = build_efficientnetb3()
        save_path = output_dir / "efficientnetb3.h5"
        trainer.train_and_save(model, train_gen, val_gen, class_weights=class_weights, save_path=str(save_path), epochs=epochs)
        tflite_path = convert_h5_to_tflite(save_path, output_dir / "efficientnetb3.tflite")
        logger.info(f"EfficientNet-B3 ready: {tflite_path}")

    elif model_name == "ensemble":
        from ml_engine.pipeline.ensemble_model import train_base_models, stack_and_train_ensemble, evaluate_ensemble_model
        logger.info("--- Running Full 3-Model Ensemble Training ---")
        base_models = train_base_models(train_gen, val_gen, class_weights=class_weights, epochs=epochs, save_dir=str(output_dir))
        meta_model = stack_and_train_ensemble(base_models, val_gen)
        evaluate_ensemble_model(base_models, meta_model, test_gen)

        # Convert each base model to quantized TFLite
        convert_h5_to_tflite(output_dir / "resnet101.h5", output_dir / "resnet101.tflite")
        convert_h5_to_tflite(output_dir / "densenet121.h5", output_dir / "densenet121.tflite")
        convert_h5_to_tflite(output_dir / "efficientnetb3.h5", output_dir / "efficientnetb3.tflite")
        logger.info("All 3 Ensemble TFLite flatbuffers successfully generated!")

    logger.info("🎉 Pipeline execution completed successfully!")

if __name__ == "__main__":
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
        help="Batch size (default: 64)"
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

