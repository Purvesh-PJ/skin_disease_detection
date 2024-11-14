import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

def get_data_generators(metadata_path=r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_metadata.csv', target_size=(224, 224), batch_size=32, sample_size=None, use_subset=False):
    try:
        folder_1 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_1'
        folder_2 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_2'

        print("##### GETTING DATA GENERATORS #####")
        print("\n")
        # Load and preprocess metadata
        print("Loading metadata...")
        metadata = pd.read_csv(metadata_path)
        
        print("\n")
        print("##### TOP FIVE SAMPLE DATA #####")
        metadata['path'] = metadata['image_id'].apply( lambda x: os.path.join(folder_1, f"{x}.jpg") if os.path.exists(os.path.join(folder_1, f"{x}.jpg")) else os.path.join(folder_2, f"{x}.jpg"))

        if use_subset and sample_size:
            metadata = metadata.sample(n=sample_size, random_state=42)
            print(f"Using a subset of {sample_size} samples for testing.")
        print(f"Metadata: {metadata.head()}")  # Check first few rows
        print("\n")

        # Encode labels to integers and convert them to strings
        print("##### LABELS #####")
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        metadata['label'] = label_encoder.fit_transform(metadata['dx']).astype(str)  # Convert to string
        print(f"Labels encoded. Unique classes: {label_encoder.classes_}")
        print("\n")

        # Training Data Augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.3,
            shear_range=0.3,
            brightness_range=[0.7, 1.3],
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # Split for validation
        )

        # Validation Data Normalization Only
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

        # Testing Data Normalization Only
        test_datagen = ImageDataGenerator(
            rescale=1./255
        )

        # Training generator
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=metadata,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'  # Use training subset
        )

        x_batch, y_batch = next(train_generator)
        print("\n")
        print("##### BATCH SHAPES #####")
        print(f"x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")
        print(f"x_batch dtype: {x_batch.dtype}")
        print(f"y_batch dtype: {y_batch.dtype}")

        # Example check
        x_batch, y_batch = next(train_generator)  # Get the first batch
        print(f"First batch input shape: {x_batch.shape}, labels shape: {y_batch.shape}")

        # Validation generator
        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=metadata,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'  # Use validation subset
        )

        # Testing generator (No subset since it's for testing)
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=metadata,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode=None,
        )

        # Check if generators are properly created
        print("\n")
        print("##### FINAL GENERATORS #####")
        print(f"Train Generator: {train_generator}")
        print(f"Validation Generator: {validation_generator}")
        print(f"Test Generator: {test_generator}")
        print("\n")

        print("Data generators created successfully.")
        print("\n")

        return train_generator, validation_generator, test_generator, label_encoder

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None


