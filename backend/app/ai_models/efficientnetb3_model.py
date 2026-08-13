"""
EfficientNetB3 Architecture Model Definition
---------------------------------------------
Fine-tuned EfficientNetB3 architecture for 7-class skin lesion classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from app.ai_models.model_trainer import train_and_save_model
from app.constants.disease_constants import TARGET_IMAGE_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "efficientnetb3.h5")

def create_efficientnetb3(input_shape=(*TARGET_IMAGE_SIZE, 3), num_classes=7):
    """Creates and compiles an EfficientNetB3 model."""
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
    """Trains EfficientNetB3 model using the centralized model trainer."""
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH
    num_classes = len(np.unique(train_generator.classes)) if hasattr(train_generator, 'classes') else 7
    model = create_efficientnetb3(num_classes=num_classes)
    return train_and_save_model(model, train_generator, val_generator, class_weights, save_path, epochs)
