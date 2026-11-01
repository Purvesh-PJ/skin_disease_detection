"""
Dataset Generators & Augmentation Utilities
------------------------------------------
Handles loading, batch generation, and augmentation pipelines
for the HAM10000 7-class dermatological dataset.
"""

import logging
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ml_engine.constants import TARGET_IMAGE_SIZE, DEFAULT_BATCH_SIZE
from ml_engine.scripts.download_dataset import DEFAULT_DATASET_DIR

logger = logging.getLogger(__name__)


def create_data_generators(dataset_dir=DEFAULT_DATASET_DIR, batch_size=DEFAULT_BATCH_SIZE):
    """
    Initializes train, validation, and test Keras ImageDataGenerators.
    Applies data augmentation on training set and rescaling on val/test sets.

    Args:
        dataset_dir (str or Path): Path to structured dataset directory containing
                                   train_dir, val_dir, test_dir.
        batch_size (int): Batch size for generators.

    Returns:
        tuple: (train_gen, val_gen, test_gen) DirectoryIterators.
    """
    train_dir = Path(dataset_dir) / "train_dir"
    val_dir = Path(dataset_dir) / "val_dir"
    test_dir = Path(dataset_dir) / "test_dir"

    # Data augmentation on training dataset
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

    # Clean scaling for validation and test datasets
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
