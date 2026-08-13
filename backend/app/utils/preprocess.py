"""
Image Preprocessing & Data Generators Utility
---------------------------------------------
Preprocessing functions, Albumentations image augmentations,
and Keras CustomDataGenerator for model training & evaluation.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
try:
    import albumentations as A
except ImportError:
    A = None
import cv2
from collections import Counter
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASE_DIR = os.path.join(BASE_DIR, "data", "skin_disease_dataset", "base_dir")

def check_class_distribution(generator, dataset_name, plot=False):
    """Calculates and optionally plots class distribution for a dataset generator."""
    labels = generator.classes
    class_counts = dict(Counter(labels))
    print(f"\n📊 Class distribution in {dataset_name}:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count} images")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
        plt.xticks(list(class_counts.keys()))
        plt.xlabel("Class Labels")
        plt.ylabel("Number of Images")
        plt.title(f"Class Distribution in {dataset_name}")
        plt.show()

def create_train_transform(target_size=(224, 224)):
    """Creates training data augmentation pipeline using Albumentations."""
    height, width = target_size
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR),
        A.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
        A.Resize(height=height, width=width)
    ])

def create_val_transform(target_size=(224, 224)):
    """Creates validation/test data resizing transform."""
    height, width = target_size
    return A.Resize(height=height, width=width)

class CustomDataGenerator(Sequence):
    """Custom Keras Data Generator with Albumentations augmentation and ResNet preprocessing."""
    def __init__(self, file_paths, labels, batch_size, transform=None, shuffle=True, display_samples=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.display_samples = display_samples
        self.indexes = np.arange(len(self.file_paths))
        self.classes = labels
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return math.ceil(len(self.file_paths) / self.batch_size)
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.file_paths[i] for i in indexes]
        batch_labels = [self.labels[i] for i in indexes]
        images = np.array([self.__load_image(f) for f in batch_files])
        if self.display_samples:
            self.display_sample_images(images, batch_labels)
        return images, np.array(batch_labels)
    
    def __load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or cannot be read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return preprocess_input(image)

    def display_sample_images(self, images, labels, images_per_row=3):
        """Displays sample images in multiple rows."""
        num_samples = min(5, len(images))
        num_cols = min(images_per_row, num_samples)
        num_rows = math.ceil(num_samples / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        imagenet_mean = np.array([103.939, 116.779, 123.68])

        for i in range(num_samples):
            row, col = divmod(i, num_cols)
            ax = axes[row, col] if num_rows > 1 else axes[col]

            img = images[i].copy()
            img = img + imagenet_mean
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = img[..., ::-1]

            ax.imshow(img)
            ax.set_title(f"Class {labels[i]}")
            ax.axis("off")

        for i in range(num_samples, num_rows * num_cols):
            row, col = divmod(i, num_cols)
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.show()
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def get_data_generators(base_dir=None, target_size=(224, 224), batch_size=64, display_samples=False):
    """
    Constructs train, validation, and test Keras data generators with computed class weights.
    """
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    train_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')
    test_dir = os.path.join(base_dir, 'test_dir')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    print("\n🔄 Loading training, validation, and test data...")
    classes = sorted(os.listdir(train_dir))
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    def load_images_labels(directory):
        file_paths, labels = [], []
        if os.path.exists(directory):
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
    
    train_transform = create_train_transform(target_size)
    val_transform = create_val_transform(target_size)
    
    train_generator = CustomDataGenerator(train_files, train_labels, batch_size, transform=train_transform, shuffle=True, display_samples=display_samples)
    validation_generator = CustomDataGenerator(val_files, val_labels, batch_size, transform=val_transform, shuffle=False, display_samples=display_samples)
    test_generator = CustomDataGenerator(test_files, test_labels, batch_size, transform=val_transform, shuffle=False, display_samples=display_samples)
    
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
    
    check_class_distribution(train_generator, "Training Set", plot=False)
    check_class_distribution(validation_generator, "Validation Set", plot=False)
    check_class_distribution(test_generator, "Test Set", plot=False)
    
    print("\n✅ Data generators created successfully!")
    
    return train_generator, validation_generator, test_generator, class_weights
