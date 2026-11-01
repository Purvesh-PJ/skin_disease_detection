"""
Model & Ensemble Performance Evaluator
--------------------------------------
Evaluates trained models against the test dataset and outputs
detailed classification reports and confusion matrices.
"""

import logging
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def evaluate_model_predictions(true_labels, predicted_labels, class_names=None):
    """
    Computes and logs classification report and confusion matrix.

    Args:
        true_labels (array-like): Ground truth labels.
        predicted_labels (array-like): Predicted class indices.
        class_names (list, optional): List of class string names.

    Returns:
        tuple: (report_str, confusion_matrix_ndarray)
    """
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    cm = confusion_matrix(true_labels, predicted_labels)

    logger.info(f"Classification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)
    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    return report, cm


def evaluate_ensemble_model(base_models, meta_model, test_generator, class_names=None):
    """
    Generates test predictions across base models, stacks them,
    runs inference through the meta-model, and logs evaluation metrics.

    Args:
        base_models (list): List of trained Keras base models.
        meta_model: Trained stacking meta-model (e.g., LogisticRegression).
        test_generator: Keras DirectoryIterator yielding test images.
        class_names (list, optional): Target class names.

    Returns:
        tuple: (final_preds, report, cm)
    """
    import numpy as np

    logger.info("--- Evaluating Ensemble Model on Test Data ---")
    test_preds = []
    for model in base_models:
        preds = model.predict(test_generator)
        test_preds.append(preds)

    stacked_test_preds = np.concatenate(test_preds, axis=1)
    final_preds = meta_model.predict(stacked_test_preds)

    true_labels = test_generator.classes if hasattr(test_generator, 'classes') else test_generator.labels
    report, cm = evaluate_model_predictions(true_labels, final_preds, class_names=class_names)

    return final_preds, report, cm
