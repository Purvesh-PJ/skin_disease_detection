import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

def get_data_generators(metadata_path=r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_metadata.csv', target_size=(600, 450), batch_size=32, sample_size=None):
    try:
        # Paths to image folders
        folder_1 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_1'
        folder_2 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_2'
        
        # Load metadata
        print("Loading metadata...")
        metadata = pd.read_csv(metadata_path)
        metadata['path'] = metadata['image_id'].apply(
            lambda x: os.path.join(folder_1, f"{x}.jpg") if os.path.exists(os.path.join(folder_1, f"{x}.jpg")) else os.path.join(folder_2, f"{x}.jpg")
        )
        print("Metadata loaded and paths created successfully.")
        
        # Limit the metadata to a subset if sample_size is provided
        if sample_size is not None:
            metadata = metadata.sample(n=sample_size, random_state=42)
            print(f"Using a subset of {sample_size} samples for testing purposes.")

        # Encode labels to integers and convert them to strings
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        metadata['label'] = label_encoder.fit_transform(metadata['dx']).astype(str)  # Convert to string
        print(f"Labels encoded. Unique classes: {label_encoder.classes_}")

        # Define data augmentation and normalization
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],  # Adjust brightness
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # Reserve 20% for validation
        )

        # Data augmentation for training data
        train_generator = datagen.flow_from_dataframe(
            dataframe=metadata,
            directory=None,  # Paths are specified in the 'path' column
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse',
            subset='training'
        )

        # Data generator for validation data
        validation_generator = datagen.flow_from_dataframe(
            dataframe=metadata,
            directory=None,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation'
        )
        
        print("Data generators created successfully.")
        return train_generator, validation_generator, label_encoder

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
