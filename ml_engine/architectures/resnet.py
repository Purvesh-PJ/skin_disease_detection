"""
ResNet101 Neural Network Architecture Factory
----------------------------------------------
Factory function for fine-tuned ResNet101 architecture.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from app.core.constants import TARGET_IMAGE_SIZE

def build_resnet101(input_shape=(*TARGET_IMAGE_SIZE, 3), num_classes=7, initial_lr=1e-4):
    """
    Constructs and compiles a fine-tuned ResNet101 model.
    """
    lr_schedule = ExponentialDecay(initial_lr, decay_steps=10000, decay_rate=0.9, staircase=True)
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
