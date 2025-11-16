#!/usr/bin/env python3
"""
Check the current state of the dataset split.
"""

from pathlib import Path


def check_dataset_state(
    train_dir: Path = Path("data/PlantVillage/train"),
    valid_dir: Path = Path("data/PlantVillage/valid")
):
    """Check and display the current state of train/valid split."""

    print("=" * 60)
    print("📊 Dataset State Check")
    print("=" * 60)

    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return

    valid_exists = valid_dir.exists()

    # Get class directories
    train_classes = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    if not train_classes:
        print(f"❌ No classes found in {train_dir}")
        return

    print(f"\n📁 Found {len(train_classes)} classes\n")
    print(f"{'Class':<40} {'Train':>10} {'Valid':>10} {'Total':>10}")
    print("-" * 72)

    total_train = 0
    total_valid = 0

    for class_dir in train_classes:
        class_name = class_dir.name

        # Count train images
        train_images = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            train_images.extend(list(class_dir.glob(ext)))
        train_count = len(train_images)

        # Count valid images
        valid_count = 0
        if valid_exists:
            valid_class_dir = valid_dir / class_name
            if valid_class_dir.exists():
                valid_images = []
                for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                    valid_images.extend(list(valid_class_dir.glob(ext)))
                valid_count = len(valid_images)

        total = train_count + valid_count

        # Truncate class name if too long
        display_name = class_name[:37] + "..." if len(class_name) > 40 else class_name

        print(f"{display_name:<40} {train_count:>10} {valid_count:>10} {total:>10}")

        total_train += train_count
        total_valid += valid_count

    print("-" * 72)
    print(f"{'TOTAL':<40} {total_train:>10} {total_valid:>10} {total_train + total_valid:>10}")

    # Calculate percentages
    if total_train + total_valid > 0:
        train_pct = (total_train / (total_train + total_valid)) * 100
        valid_pct = (total_valid / (total_train + total_valid)) * 100

        print("\n" + "=" * 60)
        print(f"📈 Split Ratio: {train_pct:.1f}% train / {valid_pct:.1f}% valid")
        print("=" * 60)

        if total_valid == 0:
            print("\n⚠️  No validation set found. Run split_train_valid.py to create it.")
        elif abs(train_pct - 80) > 5:
            print(f"\n⚠️  Split ratio is not 80/20. You may want to re-split.")
        else:
            print("\n✅ Dataset split looks good!")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    train_dir = project_root / "data" / "PlantVillage" / "train"
    valid_dir = project_root / "data" / "PlantVillage" / "valid"

    check_dataset_state(train_dir, valid_dir)
