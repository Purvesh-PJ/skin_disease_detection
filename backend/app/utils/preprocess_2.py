import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def get_data_generators(base_dir=r'D:\skin_disease_detection\backend\data2\base_dir', target_size=(224, 224), batch_size=64, sample_size=None, use_subset=False, test_split=0.2):
    """
    Creates train, validation, and test data generators from a structured directory.
    """
    train_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')
    
    print("Loading training and validation data...")
    
    # Data augmentation for training (reduced complexity)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation and testing
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    print("Creating validation generator...")
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Splitting validation data into validation and test sets...")
    filepaths = val_generator.filepaths
    labels = val_generator.classes
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        filepaths, labels, test_size=test_split, stratify=labels, random_state=42
    )
    
    print(f"Total images: {len(filepaths)} | Training: {len(train_val_files)} | Testing: {len(test_files)}")
    
    print("Creating new validation generator...")
    validation_df = pd.DataFrame({'filename': train_val_files, 'class': train_val_labels.astype(str)})
    validation_generator = val_test_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    print("Creating test generator...")
    test_df = pd.DataFrame({'filename': test_files, 'class': test_labels.astype(str)})
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Creating train generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    print("Computing class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    
    print("Class weights computed:", class_weights)
    print("Data generators created successfully.")
    
    return train_generator, validation_generator, test_generator, class_weights
