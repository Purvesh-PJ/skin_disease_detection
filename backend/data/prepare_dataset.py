"""
HAM10000 Automated Dataset Preparation Script
----------------------------------------------
This script processes the raw HAM10000 dataset downloaded from Kaggle
(HAM10000_metadata.csv and image folders) and automatically splits and organizes
the images into structured train, validation, and test directories with class subfolders
ready for model training.

Classes (7): akiec, bcc, bkl, df, mel, nv, vasc

Output Directory Structure:
backend/data/skin_disease_dataset/base_dir/
├── train_dir/ (80%)
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
├── val_dir/ (10%)
│   └── ...
└── test_dir/ (10%)
    └── ...
"""

import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def prepare_dataset(raw_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Reads metadata from raw_dir, splits images into train/val/test using stratified split,
    and copies images into organized subfolders under output_dir.
    """
    print(f"📦 Starting HAM10000 Dataset Preparation...")
    print(f"   Raw Input Directory: {raw_dir}")
    print(f"   Output Base Directory: {output_dir}\n")

    # Locate metadata CSV
    metadata_path = os.path.join(raw_dir, "HAM10000_metadata.csv")
    if not os.path.exists(metadata_path):
        # Look recursively inside raw_dir for metadata CSV
        found = False
        for root, _, files in os.walk(raw_dir):
            if "HAM10000_metadata.csv" in files:
                metadata_path = os.path.join(root, "HAM10000_metadata.csv")
                raw_dir = root
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"❌ HAM10000_metadata.csv not found in {raw_dir}.\n"
                f"Please ensure raw dataset is downloaded and extracted into {raw_dir}."
            )

    print(f"📄 Found metadata at: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    # Locate image files
    image_paths = {}
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_id = os.path.splitext(file)[0]
                image_paths[image_id] = os.path.join(root, file)

    print(f"🖼️ Found {len(image_paths)} image files.")
    if len(image_paths) == 0:
        raise FileNotFoundError(f"❌ No image files found in {raw_dir}")

    # Map image paths to metadata
    metadata['image_path'] = metadata['image_id'].map(image_paths)
    missing_images = metadata['image_path'].isnull().sum()
    if missing_images > 0:
        print(f"⚠️ Warning: {missing_images} metadata entries do not have corresponding image files.")
        metadata = metadata.dropna(subset=['image_path'])

    # Stratified Split: Train (80%), Temp (20%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        metadata,
        test_size=temp_ratio,
        stratify=metadata['dx'],
        random_state=random_state
    )

    # Split Temp into Val (50% of temp = 10% total) and Test (50% of temp = 10% total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / temp_ratio),
        stratify=temp_df['dx'],
        random_state=random_state
    )

    splits = {
        'train_dir': train_df,
        'val_dir': val_df,
        'test_dir': test_df
    }

    # Create destination directory structure
    for split_name in splits.keys():
        for cls in CLASSES:
            os.makedirs(os.path.join(output_dir, split_name, cls), exist_ok=True)

    # Copy files
    print("\n📂 Copying images to structured directories...")
    for split_name, df in splits.items():
        print(f"   Processing {split_name} ({len(df)} images)...")
        for _, row in df.iterrows():
            src_path = row['image_path']
            cls = row['dx']
            filename = os.path.basename(src_path)
            dest_path = os.path.join(output_dir, split_name, cls, filename)
            shutil.copy2(src_path, dest_path)

    print("\n✅ Dataset Preparation Complete! Summary:")
    print("-" * 50)
    for split_name in splits.keys():
        split_path = os.path.join(output_dir, split_name)
        total_images = 0
        print(f"📁 {split_name}:")
        for cls in sorted(CLASSES):
            count = len(os.listdir(os.path.join(split_path, cls)))
            total_images += count
            print(f"   - {cls:<8}: {count} images")
        print(f"   Total: {total_images} images\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated HAM10000 Dataset Preprocessor")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "raw"),
        help="Path to directory containing raw HAM10000 files (HAM10000_metadata.csv and image folders)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "skin_disease_dataset", "base_dir"),
        help="Path to output base_dir where train_dir, val_dir, test_dir will be created"
    )
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)
