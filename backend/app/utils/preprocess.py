import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def get_data_generators(metadata_path=r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_metadata.csv', 
                        target_size=(224, 224), batch_size=32, sample_size=None):
    try:
        folder_1 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_1'
        folder_2 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_2'

        # Load and preprocess metadata
        print("Loading metadata...")
        metadata = pd.read_csv(metadata_path)
        metadata['path'] = metadata['image_id'].apply(
            lambda x: os.path.join(folder_1, f"{x}.jpg") if os.path.exists(os.path.join(folder_1, f"{x}.jpg")) 
            else os.path.join(folder_2, f"{x}.jpg")
        )
        if sample_size:
            metadata = metadata.sample(n=sample_size, random_state=42)
            print(f"Using a subset of {sample_size} samples for testing.")

        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        metadata['label'] = label_encoder.fit_transform(metadata['dx']).astype(str)
        print(f"Labels encoded. Unique classes: {label_encoder.classes_}")

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
            validation_split=0.2
        )

        # Validation Data Normalization Only
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

        # Training generator
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=metadata,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse',
            subset='training'
        )

        # Validation generator
        validation_generator = validation_datagen.flow_from_dataframe(
            dataframe=metadata,
            x_col='path',
            y_col='label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation'
        )

        print("Data generators created successfully.")

        # Check a sample of images and labels from the training generator
        train_images, train_labels = next(train_generator)
        plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(train_images[i])
            plt.title(f"Label: {train_labels[i]}")
            plt.axis('off')
        plt.show()

        # Check label distribution in training and validation sets
        # train_labels_all = [label for _, label in train_generator]
        # train_label_counts = Counter(train_labels_all)
        # print("Training Label Distribution:", train_label_counts)

        # val_labels_all = [label for _, label in validation_generator]
        # val_label_counts = Counter(val_labels_all)
        # print("Validation Label Distribution:", val_label_counts)

        # # Confirm class labels
        # print("Encoded Classes:", label_encoder.classes_)

        # # Confirm batch and image shapes
        # print("Train batch shape:", train_images.shape)
        # print("Validation batch shape:", next(validation_generator)[0].shape)

        return train_generator, validation_generator, label_encoder

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None



# import os
# import pandas as pd
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.preprocessing import LabelEncoder

# def get_data_generators(metadata_path=r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_metadata.csv', target_size=(600, 450), batch_size=32, sample_size=None):
#     try:
#         # Paths to image folders
#         folder_1 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_1'
#         folder_2 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_2'
        
#         # Load metadata
#         print("Loading metadata...")
#         metadata = pd.read_csv(metadata_path)
#         metadata['path'] = metadata['image_id'].apply(
#             lambda x: os.path.join(folder_1, f"{x}.jpg") if os.path.exists(os.path.join(folder_1, f"{x}.jpg")) else os.path.join(folder_2, f"{x}.jpg")
#         )
#         print("Metadata loaded and paths created successfully.")
        
#         # Limit the metadata to a subset if sample_size is provided
#         if sample_size is not None:
#             metadata = metadata.sample(n=sample_size, random_state=42)
#             print(f"Using a subset of {sample_size} samples for testing purposes.")

#         # Encode labels to integers and convert them to strings
#         print("Encoding labels...")
#         label_encoder = LabelEncoder()
#         metadata['label'] = label_encoder.fit_transform(metadata['dx']).astype(str)  # Convert to string
#         print(f"Labels encoded. Unique classes: {label_encoder.classes_}")

#         # Define data augmentation and normalization
#         datagen = ImageDataGenerator(
#             rescale=1./255,
#             rotation_range=20,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             zoom_range=0.2,
#             shear_range=0.2,
#             brightness_range=[0.8, 1.2],  # Adjust brightness
#             horizontal_flip=True,
#             fill_mode='nearest',
#             validation_split=0.2  # Reserve 20% for validation
#         )

#         # Data augmentation for training data
#         train_generator = datagen.flow_from_dataframe(
#             dataframe=metadata,
#             directory=None,  # Paths are specified in the 'path' column
#             x_col='path',
#             y_col='label',
#             target_size=target_size,
#             batch_size=batch_size,
#             class_mode='sparse',
#             subset='training'
#         )

#         # Data generator for validation data
#         validation_generator = datagen.flow_from_dataframe(
#             dataframe=metadata,
#             directory=None,
#             x_col='path',
#             y_col='label',
#             target_size=target_size,
#             batch_size=batch_size,
#             class_mode='sparse',
#             subset='validation'
#         )
        
#         print("Data generators created successfully.")
#         return train_generator, validation_generator, label_encoder

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None, None, None
