"""
Ensemble Model Stacking & Meta-Learner
--------------------------------------
Coordinates base model inference collection and trains the stacking meta-model
(Logistic Regression) over validation probability vectors.
"""

import os
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression

from ml_engine.architectures.resnet import build_resnet101
from ml_engine.architectures.efficientnet import build_efficientnetb3
from ml_engine.architectures.densenet import build_densenet121
from ml_engine.pipeline.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def train_base_models(train_generator, val_generator, class_weights=None, epochs=30, save_dir=None):
    """
    Instantiates and trains ResNet101, EfficientNetB3, and DenseNet121 base models.
    
    Args:
        train_generator: DirectoryIterator for training data.
        val_generator: DirectoryIterator for validation data.
        class_weights (dict, optional): Class weights dictionary.
        epochs (int): Number of epochs.
        save_dir (str or Path, optional): Directory to save .h5 checkpoints.

    Returns:
        list: [resnet_model, efficientnet_model, densenet_model]
    """
    trainer = ModelTrainer()

    logger.info("--- Building & Training ResNet101 Base Model ---")
    resnet_model = build_resnet101()
    resnet_path = os.path.join(save_dir, "resnet101.h5") if save_dir else None
    resnet_model, _ = trainer.train_and_save(
        resnet_model, train_generator, val_generator, class_weights=class_weights, save_path=resnet_path, epochs=epochs
    )

    logger.info("--- Building & Training EfficientNetB3 Base Model ---")
    efficientnet_model = build_efficientnetb3()
    eff_path = os.path.join(save_dir, "efficientnetb3.h5") if save_dir else None
    efficientnet_model, _ = trainer.train_and_save(
        efficientnet_model, train_generator, val_generator, class_weights=class_weights, save_path=eff_path, epochs=epochs
    )

    logger.info("--- Building & Training DenseNet121 Base Model ---")
    densenet_model = build_densenet121()
    dense_path = os.path.join(save_dir, "densenet121.h5") if save_dir else None
    densenet_model, _ = trainer.train_and_save(
        densenet_model, train_generator, val_generator, class_weights=class_weights, save_path=dense_path, epochs=epochs
    )

    return [resnet_model, efficientnet_model, densenet_model]


def stack_and_train_ensemble(base_models, val_generator):
    """
    Collects validation set predictions from base models, stacks them,
    and trains a LogisticRegression meta-model.

    Args:
        base_models (list): List of trained Keras base models.
        val_generator: Validation data generator.

    Returns:
        LogisticRegression: Fitted meta-model.
    """
    logger.info("--- Collecting predictions from base models for stacking ---")
    val_preds = []
    for model in base_models:
        preds = model.predict(val_generator)
        val_preds.append(preds)

    stacked_val_preds = np.concatenate(val_preds, axis=1)
    val_labels = val_generator.classes if hasattr(val_generator, 'classes') else val_generator.labels

    logger.info("--- Fitting Meta-Model (Logistic Regression) ---")
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(stacked_val_preds, val_labels)

    return meta_model
