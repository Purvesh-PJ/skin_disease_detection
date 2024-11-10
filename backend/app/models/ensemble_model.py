# backend/app/models/ensemble_model.py
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def create_resnet50_model(input_shape=(224, 224, 3), num_classes=7):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=7):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_densenet_model(input_shape=(224, 224, 3), num_classes=7):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_base_models(train_data, train_labels, val_data, val_labels):
    # Initialize and train each base model
    resnet_model = create_resnet50_model()
    resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    resnet_model.fit(train_data, train_labels, epochs=30, validation_data=(val_data, val_labels))

    efficientnet_model = create_efficientnet_model()
    efficientnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    efficientnet_model.fit(train_data, train_labels, epochs=30, validation_data=(val_data, val_labels))

    densenet_model = create_densenet_model()
    densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    densenet_model.fit(train_data, train_labels, epochs=30, validation_data=(val_data, val_labels))

    return resnet_model, efficientnet_model, densenet_model

def stack_and_train_ensemble(models, val_data, val_labels):
    # Convert one-hot encoded labels back to integer labels
    val_labels_int = np.argmax(val_labels, axis=1)  # Convert one-hot labels to integer labels
    
    # Get predictions from base models
    resnet_preds = models[0].predict(val_data)
    efficientnet_preds = models[1].predict(val_data)
    densenet_preds = models[2].predict(val_data)

    # Stack the predictions from the base models
    stacked_preds = np.stack([resnet_preds, efficientnet_preds, densenet_preds], axis=1)

    # Train meta-model (Logistic Regression)
    meta_model = LogisticRegression(max_iter=1000)  # Set max_iter to avoid convergence issues
    meta_model.fit(stacked_preds.reshape(len(val_data), -1), val_labels_int)  # Use integer labels

    # Save the trained meta-model
    joblib.dump(meta_model, 'ensemble_model.pkl')

    return meta_model

def evaluate_ensemble_model(meta_model, test_data, test_labels):
    # Get predictions from base models
    resnet_preds = meta_model[0].predict(test_data)
    efficientnet_preds = meta_model[1].predict(test_data)
    densenet_preds = meta_model[2].predict(test_data)

    # Stack the test predictions
    stacked_test_preds = np.stack([resnet_preds, efficientnet_preds, densenet_preds], axis=1)

    # Predict using meta-model
    final_preds = meta_model.predict(stacked_test_preds.reshape(len(test_data), -1))

    # Evaluate ensemble model
    ensemble_accuracy = accuracy_score(test_labels, final_preds)
    print(f'Ensemble Model Test Accuracy: {ensemble_accuracy}')
