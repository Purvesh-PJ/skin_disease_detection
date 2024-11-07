import numpy as np
from utils.test_generators import test_data_generators  # Adjust the import based on your actual structure

# Check if the data generators work
def testing_data_generators():
    # Call your data generator function
    train_generator, validation_generator, label_encoder = test_data_generators()

    # Check if the generators produce data
    print("Checking train generator...")
    for i in range(3):  # Get a few batches
        x_train_batch, y_train_batch = next(train_generator)
        print(f"Batch {i + 1} - x_train shape: {x_train_batch.shape}, y_train shape: {y_train_batch.shape}")

    print("\nChecking validation generator...")
    for i in range(3):  # Get a few batches
        x_val_batch, y_val_batch = next(validation_generator)
        print(f"Batch {i + 1} - x_val shape: {x_val_batch.shape}, y_val shape: {y_val_batch.shape}")

    # Return the generators for further use
    return train_generator, validation_generator

if __name__ == "__main__":
    testing_data_generators()
