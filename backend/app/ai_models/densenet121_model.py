"""
DenseNet121 Architecture Model Definition
-----------------------------------------
Fine-tuned DenseNet121 architecture for 7-class skin lesion classification.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from app.ai_models.model_trainer import train_and_save_model
from app.constants.disease_constants import TARGET_IMAGE_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_SAVE_PATH = os.path.join(BASE_DIR, "trained_models", "densenet121.h5")

def create_densenet121_model(input_shape=(*TARGET_IMAGE_SIZE, 3), num_classes=7):
    """Creates and compiles a fine-tuned DenseNet121 model."""
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
    """Trains DenseNet121 model using the centralized model trainer."""
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH
    num_classes = len(class_weights) if class_weights else 7
    model = create_densenet121_model(num_classes=num_classes)
    return train_and_save_model(model, train_generator, val_generator, class_weights, save_path, epochs)
