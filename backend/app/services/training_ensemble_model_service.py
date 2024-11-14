from models.ensemble_model import train_base_models, stack_and_train_ensemble, evaluate_ensemble_model

# backend/app/models/ensemble_model.py

def train_and_evaluate_ensemble(train_generator, val_generator, test_generator):
    try:
        # Step 1: Train individual base models
        resnet_model, efficientnet_model, densenet_model = train_base_models(train_generator, val_generator)

        # Step 2: Stack base model predictions and train the ensemble meta-model
        meta_model = stack_and_train_ensemble([resnet_model, efficientnet_model, densenet_model], val_generator)

        # Step 3: Evaluate the ensemble model
        evaluate_ensemble_model(
            [resnet_model, efficientnet_model, densenet_model],  # Pass base models
            meta_model,  # Pass the trained meta-model
            test_generator
        )

    except Exception as e:
        print(f"An error occurred during training and evaluation: {e}")

