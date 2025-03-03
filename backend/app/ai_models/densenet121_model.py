import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Function to create the DenseNet121 model
def create_densenet121_model(input_shape=(224, 224, 3), num_classes=7):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)  # Add BatchNorm
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)  # Add BatchNorm
    x = Dropout(0.3)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# Function to train the DenseNet121 model
def train_densenet121_model(train_generator, val_generator, class_weights, save_path="../../trained_models/densenet121.h5", epochs=30):
    
    model = create_densenet121_model(input_shape=(224, 224, 3), num_classes=len(class_weights))

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

# Function to evaluate the DenseNet121 model
def evaluate_densenet121_model(model_path, test_generator):
    model = load_model(model_path)
    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=1)

    print("\n########## CLASSIFICATION REPORT ##########")
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

    print("\n########## CONFUSION MATRIX ##########")
    print(confusion_matrix(y_true, y_pred))

# Example usage:
# train_generator, val_generator, test_generator, class_weights = get_data_generators()
# model, history = train_densenet121_model(train_generator, val_generator, class_weights)
# evaluate_densenet121_model('../../trained_models/densenet121.h5', test_generator)
