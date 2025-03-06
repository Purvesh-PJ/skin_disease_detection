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
    labels = generator.classes
    class_counts = dict(Counter(labels))
    print(f"\n📊 Class distribution in {dataset_name}:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count} images")
    plt.figure(figsize=(6, 4))  # Reduced figure size
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xticks(list(class_counts.keys()))
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Images")
    plt.title(f"Class Distribution in {dataset_name}")
    plt.show()

def create_train_transform(target_size=(224, 224)):
    """
    Creates an Albumentations augmentation pipeline using the given target size.
    """
    height, width = target_size
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR),
        A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),  # 🔥 Sharpens images
        A.Emboss(alpha=(0.1, 0.4), strength=(0.2, 0.5), p=0.3),    # 🔥 Enhances texture
        # A.UnsharpMask(blur_limit=(3, 5), sigma_limit=0.5, p=0.3),  # 🔥 Extra sharpening
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),     # 🔥 Improves local contrast
        A.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5), 
        A.Resize(height=height, width=width)
    ])

def create_val_transform(target_size=(224, 224)):
    """
    Creates a simple Albumentations transform that resizes images to the target size.
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
    - (Optional) Displays a few sample images from each batch if display_samples=True.
    """
    def __init__(
        self, 
        file_paths, 
        labels, 
        batch_size, 
        transform=None, 
        shuffle=True, 
        display_samples=False
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.display_samples = display_samples  # Toggle for displaying images
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
        
        # Display sample images if enabled
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
    
    def display_sample_images(self, images, labels):
        """Displays 4-5 sample images from the batch with class labels."""
        num_samples = min(5, len(images))
        plt.figure(figsize=(8, 3))  # Smaller figure for displayed images
        
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            
            img = images[i]  
            
            # **Step 1: Handle Different Normalization Methods**
            if img.min() < 0:  # Assuming `[-1, 1]` normalization
                img = (img + 1) / 2  # Rescale to [0, 1]
            elif img.max() > 1:  # Assuming `[0, 255]` range
                img = img / 255.0  # Normalize to [0, 1]

            # **Step 2: Ensure Correct Color Format**
            if img.shape[-1] == 3:  # Check if RGB
                img = np.clip(img, 0, 1)  # Ensure valid range

            plt.imshow(img)
            plt.title(f"Class {labels[i]}")
            plt.axis("off")
        plt.show()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def get_data_generators(
    base_dir=r'D:\\skin_disease_detection\\backend\\own_data\\base_dir', 
    target_size=(224, 224), 
    batch_size=64,
    display_samples=False
):
    """
    Creates training, validation, and test data generators using Albumentations.
    display_samples: set to True if you want to see sample images from each batch.
    """
    train_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')
    test_dir = os.path.join(base_dir, 'test_dir')

    print("\n🔄 Loading training, validation, and test data...")
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
    
    train_transform = create_train_transform(target_size)
    val_transform = create_val_transform(target_size)
    
    # Pass display_samples to the generator
    train_generator = CustomDataGenerator(
        train_files, 
        train_labels, 
        batch_size, 
        transform=train_transform, 
        shuffle=True, 
        display_samples=display_samples
    )
    validation_generator = CustomDataGenerator(
        val_files, 
        val_labels, 
        batch_size, 
        transform=val_transform, 
        shuffle=False, 
        display_samples=display_samples
    )
    test_generator = CustomDataGenerator(
        test_files, 
        test_labels, 
        batch_size, 
        transform=val_transform, 
        shuffle=False, 
        display_samples=display_samples
    )
    
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
    
    check_class_distribution(train_generator, "Training Set")
    check_class_distribution(validation_generator, "Validation Set")
    check_class_distribution(test_generator, "Test Set")
    
    print("\n✅ Data generators created successfully!")
    
    return train_generator, validation_generator, test_generator, class_weights