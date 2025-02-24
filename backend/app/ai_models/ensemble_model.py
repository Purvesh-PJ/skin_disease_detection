import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


# Function to create a base model with transfer learning
def create_base_model(model_name='ResNet50', input_shape=(224, 224, 3), num_classes=7):
  if model_name == 'ResNet50':
      base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
  elif model_name == 'EfficientNetB0':
      base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
  elif model_name == 'DenseNet121':
      base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
  else:
      raise ValueError("Model name must be 'ResNet50', 'EfficientNetB0', or 'DenseNet121'.")

  # Freeze the base model layers
  base_model.trainable = False
  
  # Add custom top layers
  x = GlobalAveragePooling2D()(base_model.output)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)
  output = Dense(num_classes, activation='softmax')(x)
  
  # Create the final model
  model = Model(inputs=base_model.input, outputs=output)
  
  return model


# Function to train the individual models
def train_model(model, train_generator, validation_generator, class_weight, epochs=3, batch_size=32):
  model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
  
  # Use early stopping to avoid overfitting
  early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
  
  # Train the model
  history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping],
        verbose=1,
    )
  
  return model, history

# Function to create the ensemble model
def create_ensemble_model(train_generator, validation_generator, test_generator, class_weights, models=['ResNet50', 'EfficientNetB0', 'DenseNet121'], save_model_path='../../trained_models'):
    os.makedirs(save_model_path, exist_ok=True)

    # Step 1: Train each model and collect predictionss
    base_models = []
    base_model_preds = []  # List to store predictions from each model
    
    for model_name in models:
        print("\n")
        print(f"########## TRAINING : {model_name} ##########")
        # print(f" Class weights passing to model")
        # print(class_weights)
        # print("\n")
        model = create_base_model(model_name=model_name)
        trained_model, history = train_model(model, train_generator, validation_generator, class_weights)
        base_models.append(trained_model)
        
        # Save the trained model
        model_save_path = os.path.join(save_model_path, f"{model_name}_model.h5")
        trained_model.save(model_save_path)
        print(f"SAVED {model_name} model to {model_save_path}")
        
        # Collect predictions for stacking
        val_preds = trained_model.predict(validation_generator)
        base_model_preds.append(val_preds)  # Append predictions for each model
    
    # Step 2: Stack predictions (combine model outputs)
    print("\n")
    print("########## STACKING PREDICTIONS FROM BASE MODELS ##########")
    base_model_preds = np.concatenate(base_model_preds, axis=1)  # Stack predictions horizontally
    print("Shape of base_model_preds:", base_model_preds.shape)
    print("Length of validation_generator.classes:", len(validation_generator.classes))
    
    # Step 3: Train the meta-model (Logistic Regression)
    print("\n")
    print("########## TRAINING : Meta Model Logistic Regression ##########")
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(base_model_preds, validation_generator.classes)  # Ensure alignment
    
    # Step 4: Save the meta-model
    meta_model_save_path = os.path.join(save_model_path, "meta_model.pkl")
    import joblib
    joblib.dump(meta_model, meta_model_save_path)
    print(f"SAVED meta-model to {meta_model_save_path}")
    
    # Step 5: Evaluate the ensemble model
    print("\n")
    print("########## EVALUATING ##########")
    test_preds = []
    for model in base_models:
        test_preds.append(model.predict(test_generator))
    
    test_preds = np.concatenate(test_preds, axis=1)  # Combine predictions from all models
    final_preds = meta_model.predict(test_preds)
    
    return meta_model, final_preds, test_generator


# Function to evaluate the ensemble model
def evaluate_model(final_preds, test_generator):
    # Convert predictions to labels
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(test_generator.classes)
    
    print("\n")
    print("########## CLASSIFICATION REPORT ##########")
    print(classification_report(true_labels, final_preds))
    print("\n")
    
    print("########## CONFUSION MATRIX ##########")
    print(confusion_matrix(true_labels, final_preds))
    print("\n")


# Full pipeline for training and evaluating the ensemble model
def run_ensemble_model(train_generator, validation_generator, test_generator, class_weights):
    # Step 1: Create and train the ensemble model
    meta_model, final_preds, test_generator = create_ensemble_model(train_generator, validation_generator, test_generator, class_weights)

    # Step 2: Evaluate the model
    evaluate_model(final_preds, test_generator)
