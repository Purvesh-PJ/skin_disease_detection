"""
EfficientNetB3 Architecture Model Definition & Training Logic
--------------------------------------------------------------
Fine-tuned EfficientNetB3 architecture for 7-class skin lesion classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from app.ai_models.evaluation import evaluate_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "efficientnetb3.h5")

def create_efficientnetb3(input_shape=(224, 224, 3), num_classes=7):
    """
    Creates and compiles an EfficientNetB3 model for skin disease classification.
    """
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_efficientnetb3(train_generator, val_generator, class_weights=None, save_path=None, epochs=30):
    """
    Trains EfficientNetB3 model with EarlyStopping and ReduceLROnPlateau callbacks.
    """
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_classes = len(np.unique(train_generator.classes)) if hasattr(train_generator, 'classes') else 7
    model = create_efficientnetb3(num_classes=num_classes)
    
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
    
    model.save(save_path)
    print(f"✅ EfficientNetB3 Model saved at: {save_path}")
    return model, history
