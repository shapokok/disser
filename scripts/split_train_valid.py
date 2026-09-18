#!/usr/bin/env python3
"""
Split train folder into train/valid (80/20) if valid folder doesn't exist
Use this if you only downloaded train folder
"""

import os
import shutil
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "PlantVillage" / "train"
VALID_DIR = PROJECT_ROOT / "data" / "PlantVillage" / "valid"

SPLIT_RATIO = 0.2  # 20% for validation

def split_dataset():
    """Split train into train/valid"""

    if not TRAIN_DIR.exists():
        print(f"❌ Train directory not found: {TRAIN_DIR}")
        return

    if VALID_DIR.exists():
        response = input(f"⚠️  Valid directory already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        shutil.rmtree(VALID_DIR)

    print("=" * 60)
    print("🔀 Splitting train dataset into train/valid")
    print("=" * 60)
    print(f"Split ratio: {int((1-SPLIT_RATIO)*100)}% train, {int(SPLIT_RATIO*100)}% valid")
    print()

    # Create valid directory
    VALID_DIR.mkdir(parents=True, exist_ok=True)

    # Get all class folders
    class_folders = [f for f in TRAIN_DIR.iterdir() if f.is_dir()]

    total_moved = 0
    total_train = 0

    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"Processing: {class_name}")

        # Get all images in this class
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG"))

        # Shuffle
        random.shuffle(images)

        # Calculate split point
        num_valid = int(len(images) * SPLIT_RATIO)

        # Create valid class folder
        valid_class_folder = VALID_DIR / class_name
        valid_class_folder.mkdir(exist_ok=True)

        # Move validation images
        for img in images[:num_valid]:
            dest = valid_class_folder / img.name
            shutil.move(str(img), str(dest))

        total_moved += num_valid
        total_train += len(images) - num_valid

        print(f"  Train: {len(images) - num_valid}, Valid: {num_valid}")

    print()
    print("=" * 60)
    print("✅ Split Complete!")
    print("=" * 60)
    print(f"📁 Train images: {total_train:,}")
    print(f"📁 Valid images: {total_moved:,}")
    print()
    print("Ready to train! Run:")
    print("  python scripts/train_models.py")

if __name__ == "__main__":
    split_dataset()
