import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Function to create the Xception model
def create_xception_model(input_shape=(224, 224, 3), num_classes=7):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# Function to train the model
def train_xception_model(train_generator, val_generator, class_weights, save_path="../../trained_models/xception.h5", epochs=30, batch_size=64):
    model = create_xception_model(input_shape=(224, 224, 3), num_classes=len(class_weights))

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save(save_path)
    print(f"✅ Model saved at: {save_path}")
    return model, history

# Function to evaluate the model
def evaluate_xception_model(model_path, test_generator):
    model = load_model(model_path)
    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=1)

    print("\n########## CLASSIFICATION REPORT ##########")
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

    print("\n########## CONFUSION MATRIX ##########")
    print(confusion_matrix(y_true, y_pred))

# Example usage:
# train_generator, val_generator, test_generator, class_weights = get_data_generators()
# model, history = train_xception_model(train_generator, val_generator, class_weights)
# evaluate_xception_model('xception_model.h5', test_generator)
