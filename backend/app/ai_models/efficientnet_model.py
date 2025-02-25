import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

def create_efficientnetb3(input_shape=(224, 224, 3), num_classes=7):
    """
    Creates and compiles an EfficientNetB3 model.
    """
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Unfreeze last few layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
        layer.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_efficientnetb3(train_generator, val_generator, class_weights, save_path="D:\skin_disease_detection\trained_models\single_efficientnetb3\efficientnetb3_model.h5", epochs=30):
    """
    Trains EfficientNetB3 with Early Stopping.
    """
    model = create_efficientnetb3(num_classes=len(train_generator.class_indices))
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping],
        verbose=1
    )
    
    model.save(save_path)
    print(f"Model saved at: {save_path}")
    return model, history

def evaluate_model(model, test_generator):
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

