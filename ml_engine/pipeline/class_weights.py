"""
Dataset Class Weights Calculation
---------------------------------
Handles balanced class weights computation for penalizing dominant classes
and boosting sensitivity on minority malignant conditions.
"""

import logging
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


def compute_balanced_class_weights(train_gen):
    """
    Computes balanced class weights using scikit-learn to balance dominant classes
    (like 'nv') against minority malignant conditions (like 'mel', 'bcc', 'akiec').

    Args:
        train_gen: Keras DirectoryIterator yielding training samples.

    Returns:
        dict: Mapping of class index (int) to computed weight (float).
    """
    classes = train_gen.classes
    unique_classes = np.unique(classes)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=classes)
    class_weights_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    logger.info(f"Computed Balanced Class Weights: {class_weights_dict}")
    return class_weights_dict
