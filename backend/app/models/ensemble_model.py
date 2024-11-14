# backend/app/models/ensemble_model.py
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

def compute_class_weights(metadata):
    # Encode the labels to match the generator's class indices
    le = LabelEncoder()
    le.fit(metadata['dx'])
    encoded_labels = le.transform(metadata['dx'])
    
    # Compute class weights based on the encoded labels
    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(encoded_labels), 
        y=encoded_labels
    )
    
    # Map weights to class indices and ensure float32 type for compatibility
    class_weights_dict = {i: np.float32(weight) for i, weight in enumerate(class_weights)}
    
    print("Class weights:", class_weights_dict)
    return class_weights_dict

def train_base_models(train_generator, val_generator, metadata):
    # Early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    class_weights_dict = compute_class_weights(metadata)

    # Train ResNet50 model
    resnet_model = create_resnet50_model()
    resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    resnet_model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[early_stopping, lr_scheduler])

    # Train EfficientNetB0 model
    efficientnet_model = create_efficientnet_model()
    efficientnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    efficientnet_model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[early_stopping, lr_scheduler])

    # Train DenseNet121 model
    densenet_model = create_densenet_model()
    densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    densenet_model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[early_stopping, lr_scheduler])

    return resnet_model, efficientnet_model, densenet_model


def stack_and_train_ensemble(models, val_generator):
    # Get predictions from base models (predict on the generator batches)
    resnet_preds = models[0].predict(val_generator, steps=len(val_generator), verbose=1)
    efficientnet_preds = models[1].predict(val_generator, steps=len(val_generator), verbose=1)
    densenet_preds = models[2].predict(val_generator, steps=len(val_generator), verbose=1)

    # Stack the predictions from the base models
    stacked_preds = np.hstack([resnet_preds, efficientnet_preds, densenet_preds])

    # Convert one-hot encoded labels from val_generator to integer labels
    val_labels_int = np.argmax(val_generator.labels, axis=1)

    # Train meta-model (Logistic Regression)
    meta_model = LogisticRegression(max_iter=1000)  # Set max_iter to avoid convergence issues
    meta_model.fit(stacked_preds, val_labels_int)  # Use integer labels

    # Save the trained meta-model
    joblib.dump(meta_model, 'ensemble_model.pkl')

    return meta_model

def evaluate_ensemble_model(models, meta_model, test_generator):
    # Get predictions from base models (predict on the generator batches)
    resnet_preds = models[0].predict(test_generator, steps=len(test_generator), verbose=1)
    efficientnet_preds = models[1].predict(test_generator, steps=len(test_generator), verbose=1)
    densenet_preds = models[2].predict(test_generator, steps=len(test_generator), verbose=1)

    # Stack the test predictions
    stacked_test_preds = np.hstack([resnet_preds, efficientnet_preds, densenet_preds])

    # Convert one-hot encoded labels from test_generator to integer labels
    test_labels_int = np.argmax(test_generator.labels, axis=1)

    # Predict using meta-model
    final_preds = meta_model.predict(stacked_test_preds)

    # Evaluate ensemble model
    ensemble_accuracy = accuracy_score(test_labels_int, final_preds)
    print(f'Ensemble Model Test Accuracy: {ensemble_accuracy}')