import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def split_data_by_lesion(metadata):
    """
    Splits metadata into training, validation, and test sets ensuring no data leakage by lesion_id.
    """
    unique_lesions = metadata['lesion_id'].unique()
    train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=0.4, random_state=42)
    val_lesions, test_lesions = train_test_split(temp_lesions, test_size=0.5, random_state=42)

    train_metadata = metadata[metadata['lesion_id'].isin(train_lesions)].reset_index(drop=True)
    val_metadata = metadata[metadata['lesion_id'].isin(val_lesions)].reset_index(drop=True)
    test_metadata = metadata[metadata['lesion_id'].isin(test_lesions)].reset_index(drop=True)

    return train_metadata, val_metadata, test_metadata

def aggregate_predictions_by_lesion(predictions, metadata):
    """
    Aggregates predictions at the lesion level by averaging.
    :param predictions: Array of predictions (e.g., probabilities).
    :param metadata: DataFrame containing lesion_id and associated predictions.
    :return: Aggregated predictions DataFrame.
    """
    metadata['prediction'] = predictions
    aggregated = metadata.groupby('lesion_id')['prediction'].mean().reset_index()
    return aggregated

def get_data_generators(
    metadata_path=r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_metadata.csv',
    target_size=(224, 224),
    batch_size=32,
    sample_size=100,
    use_subset=True
):
    try:
        folder_1 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_1'
        folder_2 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_2'

        print("##### GETTING DATA GENERATORS #####\n")
        print("\n")
        # Load metadata
        print("Loading metadata...")
        metadata = pd.read_csv(metadata_path)
        
        print("\n")
        print("##### TOP FIVE SAMPLE DATA #####")
        print("\n")

        # Add image paths
        metadata['path'] = metadata['image_id'].apply(
            lambda x: os.path.join(folder_1, f"{x}.jpg")
            if os.path.exists(os.path.join(folder_1, f"{x}.jpg"))
            else os.path.join(folder_2, f"{x}.jpg")
        )

        # Optionally use a subset for testing
        if use_subset and sample_size:
            unique_classes = metadata['dx'].nunique()
            # Check if sample_size is divisible by the number of classes
            if sample_size % unique_classes != 0:
                print(f"Warning: sample_size {sample_size} is not evenly divisible by the number of classes ({unique_classes}). Adjusting sample_size.")
                sample_size = (sample_size // unique_classes) * unique_classes
            # Ensure that sample_size per class is reasonable
            min_samples_per_class = metadata['dx'].value_counts().min()
            if sample_size // unique_classes > min_samples_per_class:
                print(f"Warning: sample_size per class is higher than the smallest class size ({min_samples_per_class}). Reducing sample size.")
                sample_size = min_samples_per_class * unique_classes
            # Ensure balanced sampling across all classes
            metadata = metadata.groupby('dx').apply(lambda x: x.sample(n=sample_size // unique_classes, random_state=42)).reset_index(drop=True)
            print(f"Using a balanced subset of {sample_size} samples.")
        
        print(f"Metadata: {metadata.head()}")  # Check first few rows
        print("\n")

        # Encode labels to integers and convert them to strings
        print("##### LABELS #####")
        print("\n")
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        metadata['label'] = label_encoder.fit_transform(metadata['dx']).astype(str)
        print(f"Labels encoded. Unique classes: {label_encoder.classes_}")
        print("\n")

        # Split data by lesion_id
        print("Splitting data by lesion_id...")
        train_metadata, val_metadata, test_metadata = split_data_by_lesion(metadata)
        print("\n")

        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.3,
            shear_range=0.3,
            brightness_range=[0.7, 1.3],
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Validation Data Normalization Only
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
        
        # Testing Data Normalization Only
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Generators
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_metadata,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        x_batch, y_batch = next(train_generator)

        print("\n")
        print("##### BATCH SHAPES #####")
        print("\n")
        print(f"x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")
        print(f"x_batch dtype: {x_batch.dtype}")
        print(f"y_batch dtype: {y_batch.dtype}")

        # Example check
        x_batch, y_batch = next(train_generator)  # Get the first batch
        print(f"First batch input shape: {x_batch.shape}, labels shape: {y_batch.shape}")

        # Validation generator
        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=val_metadata,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Testing generator (No subset since it's for testing)
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_metadata,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Check if generators are properly created
        print("\n")
        print("##### FINAL GENERATORS #####")
        print("\n")
        print(f"Train Generator: {train_generator}")
        print(f"Validation Generator: {validation_generator}")
        print(f"Test Generator: {test_generator}")
        print("\n")

        print("Data generators created successfully.")

        return train_generator, validation_generator, test_generator, label_encoder

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

# Status: Original structure and key functionalities retained, new additions made seamlessly.
