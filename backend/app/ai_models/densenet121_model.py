import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import os

# Function to create the DenseNet121 model
def create_densenet121_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Creates and compiles a DenseNet121 model for skin disease classification.
    - Base model is set to trainable, but the first 50 layers are frozen.
    """
    
    # Load DenseNet121 with pretrained ImageNet weights (excluding top layers)
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  
    for layer in base_model.layers[:-70]:  # Unfreeze more layers
        layer.trainable = False  

    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)  # Normalize before activation
    x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = LeakyReLU(alpha=0.1)(x)  # Use LeakyReLU
    x = Dropout(0.5)(x)  # Increase dropout

    x = BatchNormalization()(x)
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = LeakyReLU(alpha=0.1)(x)  # Use LeakyReLU
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),  # Reduce learning rate
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

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
        # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        # EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
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
    print(f"✅ Model saved at: {save_path}")
    return model, history

# Function to evaluate the DenseNet121 model
def evaluate_densenet121_model(model, test_generator):
    """
    Evaluates the model and prints classification report & confusion matrix.
    """
    test_preds = model.predict(test_generator)
    y_pred = np.argmax(test_preds, axis=1)
    y_true = test_generator.classes
    print("\n########## CLASSIFICATION REPORT ##########")
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))
    print("\n########## CONFUSION MATRIX ##########")
    print(confusion_matrix(y_true, y_pred))


