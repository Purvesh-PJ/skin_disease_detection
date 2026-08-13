"""
ResNet101 Architecture Model Definition
---------------------------------------
Fine-tuned ResNet101 architecture for 7-class skin lesion classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from app.ai_models.model_trainer import train_and_save_model
from app.constants.disease_constants import TARGET_IMAGE_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "resnet101.h5")

initial_lr = 1e-4
lr_schedule = ExponentialDecay(initial_lr, decay_steps=10000, decay_rate=0.9, staircase=True)

def create_resnet101(input_shape=(*TARGET_IMAGE_SIZE, 3), num_classes=7):
    """Creates and compiles a fine-tuned ResNet101 model."""
    base_model = ResNet101(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = True

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, kernel_regularizer=l2(0.0001))(x)
    x = Activation("mish")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, kernel_regularizer=l2(0.0001))(x)
    x = Activation("mish")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=SGD(learning_rate=lr_schedule, momentum=0.9),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model

def train_resnet101(train_generator, val_generator, class_weights=None, save_path=None, epochs=25):
    """Trains ResNet101 model using the centralized model trainer."""
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH
    num_classes = len(np.unique(train_generator.classes)) if hasattr(train_generator, 'classes') else 7
    model = create_resnet101(num_classes=num_classes)
    return train_and_save_model(model, train_generator, val_generator, class_weights, save_path, epochs)
