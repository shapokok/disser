#!/usr/bin/env python3
"""
Split train dataset into train/valid sets with proper error handling.
"""

import shutil
from pathlib import Path
import random


def split_dataset(
    train_dir: Path = Path("data/PlantVillage/train"),
    valid_dir: Path = Path("data/PlantVillage/valid"),
    split_ratio: float = 0.8,
    seed: int = 42
):
    """
    Split training dataset into train and validation sets.

    Args:
        train_dir: Path to training directory
        valid_dir: Path to validation directory
        split_ratio: Ratio of data to keep in training (default: 0.8 = 80%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print("=" * 60)
    print("🔀 Splitting train dataset into train/valid")
    print("=" * 60)
    print(f"Split ratio: {int(split_ratio*100)}% train, {int((1-split_ratio)*100)}% valid\n")

    # Check if train directory exists
    if not train_dir.exists():
        print(f"❌ Error: Training directory not found: {train_dir}")
        print(f"   Please ensure the dataset is downloaded and extracted.")
        return

    # Get all class directories
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"❌ Error: No class directories found in {train_dir}")
        return

    print(f"Found {len(class_dirs)} classes\n")

    total_moved = 0
    total_skipped = 0
    total_errors = 0

    # Process each class
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"Processing: {class_name}")

        # Get all images in this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))

        if not image_files:
            print(f"  ⚠️  No images found in {class_name}, skipping")
            continue

        # Filter to only existing files (handle corrupted/missing entries)
        existing_files = [f for f in image_files if f.exists()]
        missing_count = len(image_files) - len(existing_files)

        if missing_count > 0:
            print(f"  ⚠️  Found {missing_count} missing files (listed but don't exist), skipping them")

        if not existing_files:
            print(f"  ⚠️  No valid images found in {class_name}, skipping")
            continue

        # Shuffle and split ONLY existing files
        random.shuffle(existing_files)
        split_idx = int(len(existing_files) * split_ratio)
        valid_images = existing_files[split_idx:]

        if not valid_images:
            print(f"  ⚠️  Not enough images to split (only {len(existing_files)}), skipping")
            continue

        # Create validation directory for this class
        valid_class_dir = valid_dir / class_name
        valid_class_dir.mkdir(parents=True, exist_ok=True)

        # Move images to validation set
        moved = 0
        skipped = 0
        errors = 0

        for img in valid_images:
            dest = valid_class_dir / img.name

            # Check if destination already exists
            if dest.exists():
                skipped += 1
                continue

            try:
                # Use absolute paths for Windows compatibility
                shutil.move(str(img.absolute()), str(dest.absolute()))
                moved += 1
            except FileNotFoundError:
                # File doesn't exist - skip silently (already filtered above)
                errors += 1
            except Exception as e:
                # Only report unexpected errors
                print(f"  ⚠️  Unexpected error moving {img.name}: {e}")
                errors += 1

        print(f"  ✓ Moved: {moved}, Skipped: {skipped}, Errors: {errors}")
        print(f"  → Train: {split_idx}, Valid: {moved}")

        total_moved += moved
        total_skipped += skipped
        total_errors += errors

    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"Total images moved to validation: {total_moved}")
    print(f"Total images skipped: {total_skipped}")
    print(f"Total errors: {total_errors}")

    if total_errors > 0:
        print("\n⚠️  Some errors occurred. Please check the messages above.")
    else:
        print("\n✅ Dataset split completed successfully!")


if __name__ == "__main__":
    # Use paths relative to project root
    project_root = Path(__file__).parent.parent
    train_dir = project_root / "data" / "PlantVillage" / "train"
    valid_dir = project_root / "data" / "PlantVillage" / "valid"

    split_dataset(train_dir, valid_dir, split_ratio=0.8)
