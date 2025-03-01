import os
import math
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.core.composition import OneOf
import cv2
from collections import Counter
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

def check_class_distribution(generator, dataset_name):
    """
    Print and visualize class distribution using the generator's classes attribute.
    """
    labels = generator.classes  # 'classes' attribute contains the labels
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

def create_train_transform(target_size=(224, 224)):
    """
    Creates an Albumentations augmentation pipeline using the given target size.
    Augmentations include horizontal flip, rotation, brightness/contrast adjustments,
    a random resized crop, and a final resize to guarantee uniform output.
    """
    height, width = target_size
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2)
        ], p=0.3),
        A.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
        # Always resize to ensure all outputs have the same dimensions
        A.Resize(height=height, width=width)
    ])

def create_val_transform(target_size=(224, 224)):
    """
    Creates a simple Albumentations transform that resizes images to the target size.
    Used for validation and test generators.
    """
    height, width = target_size
    return A.Resize(height=height, width=width)

class CustomDataGenerator(Sequence):
    """
    A custom data generator that:
    - Loads images from disk in batches.
    - Applies on-the-fly augmentation if a transform is provided.
    - Preprocesses images for EfficientNet input.
    - Shuffles data after each epoch.
    """
    def __init__(self, file_paths, labels, batch_size, transform=None, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform  # Augmentation pipeline (e.g., Albumentations)
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_paths))
        self.classes = labels  # Expose labels as an attribute for evaluation
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        # Use math.ceil to ensure all images are processed even if not a perfect batch size
        return math.ceil(len(self.file_paths) / self.batch_size)
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.file_paths[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]
        images = np.array([self.__load_image(f) for f in batch_files])
        return images, np.array(batch_labels)
    
    def __load_image(self, image_path):
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or cannot be read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply augmentation if provided (e.g., for training)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        # Preprocess image as required by EfficientNet
        return preprocess_input(image)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def get_data_generators(base_dir=r'D:\skin_disease_detection\backend\own_data\base_dir', 
                        target_size=(224, 224), batch_size=64):
    """
    Creates training, validation, and test data generators using Albumentations.
    The training generator applies augmentation; validation and test generators apply only resizing.
    """
    train_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')
    test_dir = os.path.join(base_dir, 'test_dir')

    print("\n🔄 Loading training, validation, and test data...")
    # Use training set to determine consistent class ordering.
    classes = sorted(os.listdir(train_dir))
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    def load_images_labels(directory):
        file_paths = []
        labels = []
        for class_name in classes:
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                for file in sorted(os.listdir(class_path)):
                    file_paths.append(os.path.join(class_path, file))
                    labels.append(class_indices[class_name])
        return np.array(file_paths), np.array(labels)
    
    train_files, train_labels = load_images_labels(train_dir)
    val_files, val_labels = load_images_labels(val_dir)
    test_files, test_labels = load_images_labels(test_dir)
    
    # Create augmentation pipelines
    train_transform = create_train_transform(target_size)
    val_transform = create_val_transform(target_size)
    
    # Create generators with appropriate transforms
    train_generator = CustomDataGenerator(train_files, train_labels, batch_size, transform=train_transform, shuffle=True)
    validation_generator = CustomDataGenerator(val_files, val_labels, batch_size, transform=val_transform, shuffle=False)
    test_generator = CustomDataGenerator(test_files, test_labels, batch_size, transform=val_transform, shuffle=False)
    
    # Attach class_indices mapping to each generator for evaluation purposes.
    train_generator.class_indices = class_indices
    validation_generator.class_indices = class_indices
    test_generator.class_indices = class_indices
    
    print("\n⚖️ Computing class weights for training data...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    print("Class weights computed:", class_weights)
    
    print("\n✅ Data generators created successfully!")
    return train_generator, validation_generator, test_generator, class_weights
