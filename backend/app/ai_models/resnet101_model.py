"""
ResNet101 Architecture Model Definition & Training Logic
--------------------------------------------------------
Fine-tuned ResNet101 architecture for 7-class skin lesion classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from app.utils.evaluation import evaluate_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "resnet101.h5")

initial_lr = 1e-4
lr_schedule = ExponentialDecay(initial_lr, decay_steps=10000, decay_rate=0.9, staircase=True)

def create_resnet101(input_shape=(224, 224, 3), num_classes=7):
    """
    Creates and compiles a fine-tuned ResNet101 model.
    """
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
    """
    Trains ResNet101 model with EarlyStopping and ReduceLROnPlateau callbacks.
    """
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_classes = len(np.unique(train_generator.classes)) if hasattr(train_generator, 'classes') else 7
    model = create_resnet101(num_classes=num_classes)

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
    print(f"✅ ResNet101 Model saved at: {save_path}")
    return model, history

def evaluate_resnet101_model(model, test_generator):
    """Evaluates ResNet101 model performance."""
    return evaluate_model(model, test_generator)
