# backend/app/services/training_service.py

from models.ensemble_model import train_base_models, stack_and_train_ensemble, evaluate_ensemble_model

def train_and_evaluate_ensemble(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    # Step 1: Train individual base models
    resnet_model, efficientnet_model, densenet_model = train_base_models(train_data, train_labels, val_data, val_labels)

    # Step 2: Stack base model predictions and train the ensemble meta-model
    meta_model = stack_and_train_ensemble([resnet_model, efficientnet_model, densenet_model], val_data, val_labels)

    # Step 3: Evaluate the ensemble model
    evaluate_ensemble_model(meta_model, test_data, test_labels)
