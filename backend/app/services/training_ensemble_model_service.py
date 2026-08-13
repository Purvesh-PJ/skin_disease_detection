from app.ml.ensemble_model import train_base_models, stack_and_train_ensemble, evaluate_ensemble_model

def train_and_evaluate_ensemble(train_generator, val_generator, test_generator, class_weights=None, save_dir=None):
    try:
        # Step 1: Train individual base models
        base_models = train_base_models(train_generator, val_generator, class_weights=class_weights, save_dir=save_dir)

        # Step 2: Stack base model predictions and train the ensemble meta-model
        meta_model = stack_and_train_ensemble(base_models, val_generator)

        # Step 3: Evaluate the ensemble model
        evaluate_ensemble_model(
            base_models,  # Pass base models
            meta_model,   # Pass the trained meta-model
            test_generator
        )

    except Exception as e:
        print(f"An error occurred during training and evaluation: {e}")


