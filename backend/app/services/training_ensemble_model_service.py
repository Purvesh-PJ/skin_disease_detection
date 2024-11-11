# backend/app/services/training_service.py
import pandas as pd
from models.ensemble_model import train_base_models, stack_and_train_ensemble, evaluate_ensemble_model

def train_and_evaluate_ensemble(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    metadata_path = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_metadata.csv'
    metadata = pd.read_csv(metadata_path)
    # Step 1: Train individual base models
    resnet_model, efficientnet_model, densenet_model = train_base_models(train_data, train_labels, val_data, val_labels, metadata)

    # Step 2: Stack base model predictions and train the ensemble meta-model
    meta_model = stack_and_train_ensemble([resnet_model, efficientnet_model, densenet_model], val_data, val_labels)

    # Step 3: Evaluate the ensemble model
    evaluate_ensemble_model(meta_model, test_data, test_labels)
