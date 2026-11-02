"""
HAM10000 Automated Dataset Preparation Script
----------------------------------------------
This script processes the raw HAM10000 dataset downloaded from Kaggle
(HAM10000_metadata.csv and image folders) and automatically splits and organizes
the images into structured train, validation, and test directories with class subfolders
ready for model training.

Classes (7): akiec, bcc, bkl, df, mel, nv, vasc

Output Directory Structure:
ml_engine/data/skin_disease_dataset/base_dir/
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
import zipfile
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# HAM10000 7 Diagnostic Classes
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "ml_engine" / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ml_engine" / "data" / "skin_disease_dataset" / "base_dir"


def auto_extract_archives(raw_dir: str):
    """
    Detects and extracts any nested .zip archives (e.g., HAM10000_images_part_1.zip,
    HAM10000_images_part_2.zip) found within the raw directory.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return

    for item in raw_path.rglob("*.zip"):
        extract_target = item.parent
        print(f"[ARCHIVE] Found archive: {item.name}. Extracting into {extract_target}...")
        try:
            with zipfile.ZipFile(item, "r") as zf:
                zf.extractall(extract_target)
            print(f"[OK] Extracted: {item.name}")
        except Exception as e:
            print(f"[WARNING] Could not extract {item.name}: {e}")


def prepare_dataset(raw_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Reads metadata from raw_dir, splits images into train/val/test using stratified split,
    and copies images into organized subfolders under output_dir.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    print(f"[START] Starting HAM10000 Dataset Preparation...")
    print(f"   Raw Input Directory: {raw_path}")
    print(f"   Output Base Directory: {output_path}\n")

    if not raw_path.exists():
        raise FileNotFoundError(f"[ERROR] Raw dataset directory does not exist: {raw_path}")

    # Check for and unpack any nested zip files
    auto_extract_archives(str(raw_path))

    # Locate metadata CSV
    metadata_path = raw_path / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        # Look recursively inside raw_path
        csv_matches = list(raw_path.rglob("HAM10000_metadata.csv"))
        if csv_matches:
            metadata_path = csv_matches[0]
            raw_path = metadata_path.parent
        else:
            raise FileNotFoundError(
                f"[ERROR] HAM10000_metadata.csv not found in {raw_dir}.\n"
                f"Please ensure raw dataset is downloaded and extracted into {raw_dir}."
            )

    print(f"[METADATA] Found metadata at: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    # Locate image files
    image_paths = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_file in raw_path.rglob(ext):
            image_paths[img_file.stem] = str(img_file)

    print(f"[IMAGES] Found {len(image_paths)} image files.")
    if len(image_paths) == 0:
        raise FileNotFoundError(f"[ERROR] No image files found in {raw_path}")

    # Map image paths to metadata
    metadata['image_path'] = metadata['image_id'].map(image_paths)
    missing_images = metadata['image_path'].isnull().sum()
    if missing_images > 0:
        print(f"[WARNING] {missing_images} metadata entries do not have corresponding image files.")
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
            (output_path / split_name / cls).mkdir(parents=True, exist_ok=True)

    # Copy files
    print("\n[COPYING] Copying images to structured directories...")
    for split_name, df in splits.items():
        print(f"   Processing {split_name} ({len(df)} images)...")
        for _, row in df.iterrows():
            src_path = row['image_path']
            cls = row['dx']
            filename = os.path.basename(src_path)
            dest_path = output_path / split_name / cls / filename
            shutil.copy2(src_path, dest_path)

    print("\n[COMPLETE] Dataset Preparation Complete! Summary:")
    print("-" * 50)
    for split_name in splits.keys():
        split_path = output_path / split_name
        total_images = 0
        print(f"Split: {split_name}:")
        for cls in sorted(CLASSES):
            cls_folder = split_path / cls
            count = len(list(cls_folder.iterdir())) if cls_folder.exists() else 0
            total_images += count
            print(f"   - {cls:<8}: {count} images")
        print(f"   Total: {total_images} images\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated HAM10000 Dataset Preprocessor")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Path to directory containing raw HAM10000 files (HAM10000_metadata.csv and image folders)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Path to output base_dir where train_dir, val_dir, test_dir will be created"
    )
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)
