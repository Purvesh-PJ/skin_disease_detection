import os
import math
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from collections import Counter
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

def check_class_distribution(generator, dataset_name):
    labels = generator.classes
    class_counts = dict(Counter(labels))
    print(f"\n📊 Class distribution in {dataset_name}:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count} images")
    plt.figure(figsize=(6, 4))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xticks(list(class_counts.keys()))
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Images")
    plt.title(f"Class Distribution in {dataset_name}")
    plt.show()

def create_train_transform(target_size=(224, 224)):
    height, width = target_size
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR),
        # A.RandomBrightnessContrast(p=0.1, brightness_limit=0.05, contrast_limit=0.05),
        A.RandomResizedCrop(size=(height,width), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
        # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        # A.Emboss(alpha=(0.1, 0.4), strength=(0.2, 0.5), p=0.3),
        # A.UnsharpMask(blur_limit=(3, 5), sigma_limit=0.5, p=0.3),
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.Resize(height=height, width=width)
    ])

def create_val_transform(target_size=(224, 224)):
    height, width = target_size
    return A.Resize(height=height, width=width)

class CustomDataGenerator(Sequence):
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
        """Displays sample images in multiple rows while keeping original functionality."""
        num_samples = min(5, len(images))

        # Calculate grid size for rows & columns
        num_cols = min(images_per_row, num_samples)  # Maximum `images_per_row` per row
        num_rows = math.ceil(num_samples / num_cols)  # Determine required rows

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

        # ResNet ImageNet mean values (BGR order, as ResNet expects OpenCV format)
        imagenet_mean = np.array([103.939, 116.779, 123.68])  # Shape: (3,)

        for i in range(num_samples):
            row, col = divmod(i, num_cols)  # Get row & col index
            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle 1-row case

            img = images[i].copy()  # Avoid modifying original data
            
            # **Step 1: Undo ResNet Preprocessing**
            img = img + imagenet_mean  # Add back ImageNet mean
            img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure valid range

            # **Step 2: Convert BGR → RGB (since ResNet expects OpenCV format)**
            img = img[..., ::-1]  # Swap channels from BGR to RGB

            ax.imshow(img)
            ax.set_title(f"Class {labels[i]}")
            ax.axis("off")

        # Remove empty subplots if any
        for i in range(num_samples, num_rows * num_cols):
            row, col = divmod(i, num_cols)
            fig.delaxes(axes[row, col])  # Remove unused subplot

        plt.tight_layout()
        plt.show()
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def get_data_generators(base_dir=r"D:\skin_disease_detection\backend\own_data_2\base_dir", 
    target_size=(224, 224), 
    batch_size=64, 
    display_samples=False
):
    train_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')
    test_dir = os.path.join(base_dir, 'test_dir')

    print("\n🔄 Loading training, validation, and test data...")
    classes = sorted(os.listdir(train_dir))
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    def load_images_labels(directory):
        file_paths, labels = [], []
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
    
    check_class_distribution(train_generator, "Training Set")
    check_class_distribution(validation_generator, "Validation Set")
    check_class_distribution(test_generator, "Test Set")
    
    print("\n✅ Data generators created successfully!")
    
    return train_generator, validation_generator, test_generator, class_weights
