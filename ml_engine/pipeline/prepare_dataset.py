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
    Provides real-time progress logging so the terminal never appears frozen.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return

    for item in raw_path.rglob("*.zip"):
        extract_target = item.parent
        print(f"[ARCHIVE] Inspecting archive: {item.name}...", flush=True)
        try:
            with zipfile.ZipFile(item, "r") as zf:
                members = zf.infolist()
                total = len(members)
                if total == 0:
                    continue

                # Quick check: if first and last non-directory files already exist in target, skip extraction
                non_dir_members = [m for m in members if not m.is_dir()]
                if non_dir_members:
                    samples = [non_dir_members[0], non_dir_members[-1]]
                    if all((extract_target / s.filename).exists() for s in samples):
                        print(f"[ARCHIVE] {item.name} already unpacked ({total} files present). Skipping re-extraction.\n", flush=True)
                        continue

                print(f"[ARCHIVE] Extracting {total} files from {item.name} into {extract_target}...", flush=True)
                for idx, member in enumerate(members, 1):
                    zf.extract(member, extract_target)
                    if idx % 1000 == 0 or idx == total:
                        pct = (idx / total) * 100
                        print(f"   -> [{idx}/{total}] files unpacked ({pct:.1f}%)...", flush=True)
            print(f"[OK] Completed extraction: {item.name}\n", flush=True)
        except Exception as e:
            print(f"[WARNING] Could not extract {item.name}: {e}\n", flush=True)


def prepare_dataset(raw_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Reads metadata from raw_dir, splits images into train/val/test using stratified split,
    and copies images into organized subfolders under output_dir with streaming progress.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    print(f"[START] Starting HAM10000 Dataset Preparation...", flush=True)
    print(f"   Raw Input Directory: {raw_path}", flush=True)
    print(f"   Output Base Directory: {output_path}\n", flush=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"[ERROR] Raw dataset directory does not exist: {raw_path}")

    # Check for and unpack any nested zip files with live progress
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

    print(f"[METADATA] Found metadata at: {metadata_path}", flush=True)
    metadata = pd.read_csv(metadata_path)

    # Locate image files
    print(f"[SCANNING] Indexing image files in {raw_path}...", flush=True)
    image_paths = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_file in raw_path.rglob(ext):
            image_paths[img_file.stem] = str(img_file)

    print(f"[IMAGES] Found {len(image_paths)} image files.", flush=True)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"[ERROR] No image files found in {raw_path}")

    # Map image paths to metadata
    metadata['image_path'] = metadata['image_id'].map(image_paths)
    missing_images = metadata['image_path'].isnull().sum()
    if missing_images > 0:
        print(f"[WARNING] {missing_images} metadata entries do not have corresponding image files.", flush=True)
        metadata = metadata.dropna(subset=['image_path'])

    # Stratified Split: Train (80%), Temp (20%)
    print("[SPLIT] Computing stratified 80% train / 10% validation / 10% test split...", flush=True)
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

    # Copy files with real-time streaming progress logs
    print("\n[COPYING] Copying images to structured directories...", flush=True)
    total_to_copy = sum(len(df) for df in splits.values())
    copied_so_far = 0

    for split_name, df in splits.items():
        split_count = len(df)
        print(f"\n   Processing {split_name} ({split_count} images)...", flush=True)
        split_copied = 0
        for _, row in df.iterrows():
            src_path = row['image_path']
            cls = row['dx']
            filename = os.path.basename(src_path)
            dest_path = output_path / split_name / cls / filename
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
            split_copied += 1
            copied_so_far += 1
            if split_copied % 500 == 0 or split_copied == split_count:
                pct = (copied_so_far / total_to_copy) * 100
                print(f"      -> [{split_copied}/{split_count}] ({split_name}) | Total: [{copied_so_far}/{total_to_copy}] ({pct:.1f}%)", flush=True)

    print("\n[COMPLETE] Dataset Preparation Complete! Summary:", flush=True)
    print("-" * 50, flush=True)
    for split_name in splits.keys():
        split_path = output_path / split_name
        total_images = 0
        print(f"Split: {split_name}:", flush=True)
        for cls in sorted(CLASSES):
            cls_folder = split_path / cls
            count = len(list(cls_folder.iterdir())) if cls_folder.exists() else 0
            total_images += count
            print(f"   - {cls:<8}: {count} images", flush=True)
        print(f"   Total: {total_images} images\n", flush=True)


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
