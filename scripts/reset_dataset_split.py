#!/usr/bin/env python3
"""
Reset the dataset split by moving all valid images back to train.
"""

import shutil
from pathlib import Path


def reset_split(
    train_dir: Path = Path("data/PlantVillage/train"),
    valid_dir: Path = Path("data/PlantVillage/valid")
):
    """Move all images from valid back to train directory."""

    print("=" * 60)
    print("🔄 Resetting dataset split")
    print("=" * 60)
    print("This will move all images from valid/ back to train/\n")

    if not valid_dir.exists():
        print(f"✓ No validation directory found. Nothing to reset.")
        return

    # Get all class directories in valid
    valid_classes = [d for d in valid_dir.iterdir() if d.is_dir()]

    if not valid_classes:
        print(f"✓ Validation directory is empty. Nothing to reset.")
        # Remove empty valid directory
        try:
            shutil.rmtree(valid_dir)
            print(f"✓ Removed empty validation directory")
        except Exception as e:
            print(f"⚠️  Could not remove valid directory: {e}")
        return

    print(f"Found {len(valid_classes)} classes in validation set\n")

    total_moved = 0
    total_errors = 0

    # Process each class
    for valid_class_dir in sorted(valid_classes):
        class_name = valid_class_dir.name
        print(f"Processing: {class_name}")

        # Get all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            images.extend(list(valid_class_dir.glob(ext)))

        if not images:
            print(f"  ✓ No images found, skipping")
            continue

        # Ensure train class directory exists
        train_class_dir = train_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)

        # Move images back to train
        moved = 0
        errors = 0

        for img in images:
            dest = train_class_dir / img.name

            # Check if destination already exists
            if dest.exists():
                # File already in train, just remove from valid
                try:
                    img.unlink()
                    moved += 1  # Count as moved (already in correct location)
                except Exception as e:
                    # If file doesn't exist, it's fine - already removed
                    if img.exists():
                        print(f"  ⚠️  Could not remove duplicate {img.name}: {e}")
                        errors += 1
                    else:
                        moved += 1  # File doesn't exist, count as success
                continue

            try:
                # Use absolute paths for Windows compatibility
                shutil.move(str(img.absolute()), str(dest.absolute()))
                moved += 1
            except FileNotFoundError:
                # File already moved or doesn't exist
                if dest.exists():
                    moved += 1  # Already in destination
                else:
                    errors += 1
            except Exception as e:
                print(f"  ⚠️  Error moving {img.name}: {e}")
                errors += 1

        print(f"  ✓ Moved: {moved}, Errors: {errors}")

        # Remove empty class directory
        try:
            if not any(valid_class_dir.iterdir()):
                valid_class_dir.rmdir()
        except Exception:
            pass

        total_moved += moved
        total_errors += errors

    # Try to remove the valid directory if it's empty
    try:
        if not any(valid_dir.iterdir()):
            valid_dir.rmdir()
            print(f"\n✓ Removed empty validation directory")
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"Total images moved back to train: {total_moved}")
    print(f"Total errors: {total_errors}")

    if total_errors > 0:
        print("\n⚠️  Some errors occurred. Please check the messages above.")
    else:
        print("\n✅ Dataset reset completed successfully!")
        print("   You can now run split_train_valid.py again.")


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    train_dir = project_root / "data" / "PlantVillage" / "train"
    valid_dir = project_root / "data" / "PlantVillage" / "valid"

    # Ask for confirmation
    if len(sys.argv) > 1 and sys.argv[1] == "--yes":
        reset_split(train_dir, valid_dir)
    else:
        print("=" * 60)
        print("⚠️  WARNING: This will move all images from valid/ to train/")
        print("=" * 60)
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            print()
            reset_split(train_dir, valid_dir)
        else:
            print("\n❌ Reset cancelled.")
