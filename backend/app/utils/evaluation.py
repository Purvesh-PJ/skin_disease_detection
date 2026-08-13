"""
Model Evaluation Utilities
--------------------------
Shared evaluation functions for calculating classification metrics and confusion matrices.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_generator):
    """
    Evaluates a trained Keras model on test data generator and prints 
    classification report and confusion matrix.
    """
    print("\n🔄 Running Model Inference on Test Set...")
    test_preds = model.predict(test_generator)
    y_pred = np.argmax(test_preds, axis=1)
    y_true = test_generator.classes
    
    target_names = list(test_generator.class_indices.keys()) if hasattr(test_generator, 'class_indices') else None

    print("\n########## CLASSIFICATION REPORT ##########")
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("\n########## CONFUSION MATRIX ##########")
    print(confusion_matrix(y_true, y_pred))

    return y_true, y_pred
