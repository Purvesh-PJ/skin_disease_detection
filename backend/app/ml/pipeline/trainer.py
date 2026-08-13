"""
Configurable ML Model Trainer
-----------------------------
Flexible training runner with configurable callbacks, learning rates,
and automated checkpoint saving.
"""

import os
import logging
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Enterprise Model Trainer with configurable parameters and callbacks."""
    def __init__(self, patience=3, lr_factor=0.5, min_lr=1e-6):
        self.patience = patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr

    def train_and_save(self, model, train_generator, val_generator, class_weights=None, save_path=None, epochs=30):
        """Executes model fitting with EarlyStopping and ReduceLROnPlateau."""
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=self.lr_factor, patience=self.patience, min_lr=self.min_lr, verbose=1)
        ]

        logger.info(f"Starting model training for {epochs} epochs...")
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
            logger.info(f"Model checkpoint saved at: {save_path}")

        return model, history

def train_and_save_model(model, train_generator, val_generator, class_weights=None, save_path=None, epochs=30):
    """Functional wrapper for ModelTrainer."""
    trainer = ModelTrainer()
    return trainer.train_and_save(model, train_generator, val_generator, class_weights, save_path, epochs)
