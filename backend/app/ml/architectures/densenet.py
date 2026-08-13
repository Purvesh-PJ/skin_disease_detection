"""
DenseNet121 Neural Network Architecture Factory
-----------------------------------------------
Factory function for fine-tuned DenseNet121 architecture.
"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from app.core.constants import TARGET_IMAGE_SIZE

def build_densenet121(input_shape=(*TARGET_IMAGE_SIZE, 3), num_classes=7, learning_rate=1e-4):
    """
    Constructs and compiles a fine-tuned DenseNet121 model.
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
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
