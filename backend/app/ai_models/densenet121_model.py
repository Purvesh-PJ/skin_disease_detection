"""
DenseNet121 Architecture Model Definition & Training Logic
----------------------------------------------------------
Fine-tuned DenseNet121 architecture for 7-class skin lesion classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from app.ai_models.evaluation import evaluate_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "densenet121.h5")

def create_densenet121_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Creates and compiles a fine-tuned DenseNet121 model for skin disease classification.
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  
    for layer in base_model.layers[:-70]:
        layer.trainable = False  

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_densenet121_model(train_generator, val_generator, class_weights=None, save_path=None, epochs=30):
    """
    Trains the DenseNet121 model with EarlyStopping and ReduceLROnPlateau callbacks.
    """
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_classes = len(class_weights) if class_weights else 7
    model = create_densenet121_model(input_shape=(224, 224, 3), num_classes=num_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save(save_path)
    print(f"✅ DenseNet121 Model saved at: {save_path}")
    return model, history

def evaluate_densenet121_model(model, test_generator):
    """Evaluates DenseNet121 model performance."""
    return evaluate_model(model, test_generator)
