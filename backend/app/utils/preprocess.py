import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_data_generators(metadata_path=r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_metadata.csv', target_size=(600, 450), batch_size=32):
    
    # Paths to image folders
    folder_1 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_1'
    folder_2 = r'D:\skin_disease_detection\backend\data\Ham10000\HAM10000_images_part_2'
    
    # Load metadata to map images to labels
    metadata = pd.read_csv(metadata_path)
    metadata.set_index('image_id', inplace=True)

    # Define data augmentation and normalization
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    def load_images_and_labels(folder):
        images = []
        labels = []
        for filename in os.listdir(folder):
            img_id = os.path.splitext(filename)[0]
            if img_id in metadata.index:
                label = metadata.loc[img_id, 'dx']  # Retrieve the label from metadata
                img_path = os.path.join(folder, filename)
                img = load_img(img_path, target_size=target_size)
                images.append(img_to_array(img))
                labels.append(label)
        return np.array(images), np.array(labels)

    # Load images and labels from both folders
    images_1, labels_1 = load_images_and_labels(folder_1)
    images_2, labels_2 = load_images_and_labels(folder_2)

    # Combine both sets
    images = np.concatenate((images_1, images_2), axis=0)
    labels = np.concatenate((labels_1, labels_2), axis=0)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Shuffle the data before splitting
    indices = np.arange(len(images))
    np.random.shuffle(indices)  # Shuffle indices

    images = images[indices]
    labels = labels[indices]

    # Split data into training and validation
    num_val_samples = int(0.2 * len(images))
    x_train, x_val = images[:-num_val_samples], images[-num_val_samples:]
    y_train, y_val = labels[:-num_val_samples], labels[-num_val_samples:]

    # Data augmentation for training data
    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset="training")
    validation_generator = datagen.flow(x_val, y_val, batch_size=batch_size, subset="validation")

    return train_generator, validation_generator, label_encoder
