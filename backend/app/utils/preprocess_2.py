import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def check_class_distribution(generator, dataset_name):
    """Print and visualize class distribution."""
    labels = generator.classes
    class_counts = dict(Counter(labels))
    print(f"\n📊 Class distribution in {dataset_name}:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count} images")

    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xticks(list(class_counts.keys()))
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Images")
    plt.title(f"Class Distribution in {dataset_name}")
    plt.show()

def get_data_generators(base_dir=r'D:\skin_disease_detection\backend\data2\base_dir', target_size=(224, 224), batch_size=64, test_split=0.2):
    """
    Creates train, validation, and test data generators and checks for proper splits.
    """
    train_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')

    print("\n🔄 Loading training and validation data...")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print("\n📂 Creating validation generator...")
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,  # ✅ Fixed: Resizing to (224, 224)
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Get file paths and labels
    filepaths = val_generator.filepaths
    labels = val_generator.classes

    # Check initial validation class distribution
    check_class_distribution(val_generator, "Original Validation Set")

    print("\n🔀 Splitting validation data with balanced classes...")
    # Convert filepaths and labels into a DataFrame
    df = pd.DataFrame({'filename': filepaths, 'label': labels})
    # Set minimum per class
    min_per_class = 100  # Adjust as needed
    # Ensure at least `min_per_class` in both validation & test
    test_df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), min_per_class), random_state=42))
    remaining_df = df.drop(test_df.index)
    # Further split remaining data into validation & train
    val_df = remaining_df.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), min_per_class * 2), random_state=42))
    train_val_df = remaining_df.drop(val_df.index)
    # Extract filenames and labels
    train_val_files, train_val_labels = train_val_df['filename'].values, train_val_df['label'].values
    val_files, val_labels = val_df['filename'].values, val_df['label'].values
    test_files, test_labels = test_df['filename'].values, test_df['label'].values
    print(f"\n✅ After rebalancing:")
    print(f"   Training + Validation: {len(train_val_files)}")
    print(f"   Validation: {len(val_files)}")
    print(f"   Testing: {len(test_files)}")

    # Convert split data into DataFrames for generators
    validation_df = pd.DataFrame({'filename': train_val_files, 'class': train_val_labels.astype(str)})
    test_df = pd.DataFrame({'filename': test_files, 'class': test_labels.astype(str)})

    print("\n📂 Creating new validation generator...")
    validation_generator = val_test_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col='filename',
        y_col='class',
        target_size=target_size,  # ✅ Fixed: Resizing to (224, 224)
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    print("\n📂 Creating test generator...")
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=target_size,  # ✅ Fixed: Resizing to (224, 224)
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("\n📂 Creating train generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,  # ✅ Fixed: Resizing to (224, 224)
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Compute class weights
    print("\n⚖️ Computing class weights for training data...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    
    print("Class weights computed:", class_weights)

    # Check class distributions
    check_class_distribution(train_generator, "Train Set")
    check_class_distribution(validation_generator, "New Validation Set")
    check_class_distribution(test_generator, "Test Set")

    print("\n✅ Data generators created successfully!")
    
    return train_generator, validation_generator, test_generator, class_weights
