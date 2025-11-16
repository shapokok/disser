#!/usr/bin/env python3
"""
Atomic reset and split - ensures clean 80/20 split
Forcefully removes valid directory and immediately splits
"""

import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.split_train_valid import split_dataset


def force_remove_valid(valid_dir: Path):
    """Forcefully remove valid directory"""
    if valid_dir.exists():
        print(f"🗑️  Removing existing valid directory: {valid_dir}")
        try:
            shutil.rmtree(valid_dir, ignore_errors=True)
            print(f"✓ Removed {valid_dir}")
        except Exception as e:
            print(f"⚠️  Error removing valid directory: {e}")
            print("Trying to continue anyway...")

    # Double check it's gone
    if valid_dir.exists():
        print(f"❌ Warning: {valid_dir} still exists!")
        print("Attempting manual cleanup...")

        # Try to remove all subdirectories first
        for item in valid_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink()
            except:
                pass

        # Try to remove the directory itself
        try:
            valid_dir.rmdir()
        except:
            pass

    if not valid_dir.exists():
        print("✓ Valid directory removed successfully\n")
    else:
        print("⚠️  Could not fully remove valid directory")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return False

    return True


def main():
    print("=" * 60)
    print("🔄 Atomic Reset and Split")
    print("=" * 60)
    print("This will:")
    print("1. Forcefully remove valid/ directory")
    print("2. Immediately split train/ into 80% train / 20% valid")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    train_dir = project_root / "data" / "PlantVillage" / "train"
    valid_dir = project_root / "data" / "PlantVillage" / "valid"

    if not train_dir.exists():
        print(f"❌ Error: Training directory not found: {train_dir}")
        return

    # Ask for confirmation
    print(f"\nTrain directory: {train_dir}")
    print(f"Valid directory: {valid_dir}")
    response = input("\nProceed? (yes/no): ")

    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return

    print()

    # Step 1: Force remove valid
    if not force_remove_valid(valid_dir):
        return

    # Step 2: Immediately split (no pause for Windows to recreate directory)
    print("=" * 60)
    print("🚀 Starting split immediately...")
    print("=" * 60)
    print()

    split_dataset(train_dir, valid_dir, split_ratio=0.8, seed=42)


if __name__ == "__main__":
    main()
