"""
Centralized Model Trainer Utility
---------------------------------
Reusable training pipeline for fitting Keras models, configuring callbacks,
and saving weights.
"""

import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_and_save_model(model, train_generator, val_generator, class_weights=None, save_path=None, epochs=30):
    """
    Executes model training with EarlyStopping and ReduceLROnPlateau callbacks,
    and saves trained model weights to save_path.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    if save_path:
        model.save(save_path)
        print(f"✅ Model saved successfully at: {save_path}")

    return model, history
