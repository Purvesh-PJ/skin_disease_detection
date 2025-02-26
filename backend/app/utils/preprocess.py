import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

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

    # Debugging: Check for data leakage
    print("\n")
    print("########## CHECKING DATA LEAKAGE ##########")
    print("Overlap between train and validation lesion_ids:", set(train_metadata['lesion_id']).intersection(set(val_metadata['lesion_id'])))
    print("Overlap between validation and test lesion_ids:", set(val_metadata['lesion_id']).intersection(set(test_metadata['lesion_id'])))

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


def calculate_class_weights(metadata, label_column='label'):
    try:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(metadata[label_column])

        class_weights_raw = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(encoded_labels),
            y=encoded_labels
        )

        class_weights = {int(label): float(weight) for label, weight in zip(np.unique(encoded_labels), class_weights_raw)}

        print("########## CLASS WEIGHTS ##########")
        print("Class Weights:", class_weights)
        return class_weights, label_encoder

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def get_data_generators(metadata_path=r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_metadata.csv', target_size=(224, 224), batch_size=32, sample_size=None, use_subset=False):
    try:
        folder_1 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_1'
        folder_2 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_2'

        print("########## GETTING DATA GENERATORS ##########\n")
        # Load metadata
        print("Loading metadata...")
        metadata = pd.read_csv(metadata_path)

        print("\n")
        print("########## TOP FIVE SAMPLE DATA ##########")

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
                print(f"WARNING : sample_size {sample_size} is not evenly divisible by the number of classes ({unique_classes}). Adjusting sample_size.")
                sample_size = (sample_size // unique_classes) * unique_classes
            # Ensure that sample_size per class is reasonable
            min_samples_per_class = metadata['dx'].value_counts().min()
            if sample_size // unique_classes > min_samples_per_class:
                print(f"WARNING : sample_size per class is higher than the smallest class size ({min_samples_per_class}). Reducing sample size.")
                sample_size = min_samples_per_class * unique_classes
            # Ensure balanced sampling across all classes
            metadata = metadata.groupby('dx').apply(lambda x: x.sample(n=sample_size // unique_classes, random_state=42)).reset_index(drop=True)
            print(f"Using a balanced subset of {sample_size} samples.")

        print(f"Metadata: {metadata.head()}")  # Check first few rows
        print("\n")

        # Encode labels to integers and convert them to strings
        print("########## LABELS ##########")
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        metadata['label'] = label_encoder.fit_transform(metadata['dx']).astype(str)
        print(f"Labels encoded. Unique classes: {label_encoder.classes_}")
        print("\n")

        # Split data by lesion_id
        print("########## SPLITTING DATA BY LENSION ##########")
        print("Splitting data by lesion_id...")
        train_metadata, val_metadata, test_metadata = split_data_by_lesion(metadata)
        print("\n")
        
        # Debugging: Check class distribution
        print("########## CLASS DISTRIBUTION ##########")
        print("Train class distribution:", train_metadata['label'].value_counts())
        print("Validation class distribution:", val_metadata['label'].value_counts())
        print("\n")

        # Calculate class weights for training data
        print("########## CLASS WEIGHTS ##########")
        print("Calculating class weights...")
        class_weights, label_encoder = calculate_class_weights(train_metadata, label_column='label')
        print(f"Class Weights: {class_weights}")
        print("\n")

        print("########## ENCODING LABELS ##########")
        print(f"label encoder: {label_encoder}")
        print("\n")

        # Data augmentation
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
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
        validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        # Testing Data Normalization Only
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

        print("########## BATCH SHAPES ##########")
        print(f"x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")
        print("\n")

        print("########## BATCH TYPES ##########")
        print(f"x_batch dtype: {x_batch.dtype}, y_batch dtype: {y_batch.dtype}\n")
        print("\n")

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
        print("########## FINAL GENERATORS ##########")
        print(f"Train Generator: {train_generator}")
        print(f"Validation Generator: {validation_generator}")
        print(f"Test Generator: {test_generator}")
        print("\n")

        print("Data generators created successfully.")

        return train_generator, validation_generator, test_generator, label_encoder, class_weights

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None
