import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Learning Rate Schedule
initial_lr = 1e-4
lr_schedule = ExponentialDecay(initial_lr, decay_steps=10000, decay_rate=0.9, staircase=True)

def create_resnet101(input_shape=(224, 224, 3), num_classes=7):
    """
    Creates and compiles a ResNet101 model.
    """
    base_model = ResNet101(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Unfreeze last few layers for fine-tuning

    # Freeze all layers except the last 20 for fine-tuning
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
    model.compile(optimizer=SGD(learning_rate=lr_schedule, momentum=0.9, weight_decay=1e-4), 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    return model

def train_resnet101(train_generator, val_generator, class_weights, 
                     save_path="../../trained_models/resnet101.h5", epochs=25):
    """
    Trains ResNet101 with Early Stopping.
    """
    num_classes = len(np.unique(train_generator.classes))
    model = create_resnet101(num_classes=num_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
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
