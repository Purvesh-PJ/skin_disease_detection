"""
EfficientNetB3 Neural Network Architecture Factory
--------------------------------------------------
Factory function for fine-tuned EfficientNetB3 architecture.
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from app.core.constants import TARGET_IMAGE_SIZE

def build_efficientnetb3(input_shape=(*TARGET_IMAGE_SIZE, 3), num_classes=7, learning_rate=1e-4):
    """
    Constructs and compiles an EfficientNetB3 model.
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
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
