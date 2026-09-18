#!/usr/bin/env python3
"""
Download and prepare PlantVillage dataset for training
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Dataset info
DATASET_URL = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
DATASET_NAME = "PlantVillage"
DATASET_ZIP = "plantvillage.zip"

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "PlantVillage"


def download_file(url, destination):
    """Download file with progress bar"""
    print(f"📥 Downloading dataset to {destination}...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    print("✓ Download complete!")


def extract_zip(zip_path, extract_to):
    """Extract ZIP file with progress"""
    print(f"📦 Extracting to {extract_to}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting") as progress_bar:
            for member in members:
                zip_ref.extract(member, extract_to)
                progress_bar.update(1)

    print("✓ Extraction complete!")


def main():
    print("=" * 60)
    print("🌱 PlantVillage Dataset Downloader")
    print("=" * 60)
    print()

    # Create directories
    DATA_DIR.mkdir(exist_ok=True)

    zip_path = DATA_DIR / DATASET_ZIP

    # Check if already downloaded
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print("✓ Dataset already exists!")
        print(f"📁 Location: {DATASET_DIR}")

        # Count images
        image_count = sum(1 for _ in DATASET_DIR.rglob("*.jpg")) + \
                      sum(1 for _ in DATASET_DIR.rglob("*.JPG"))
        print(f"📊 Total images: {image_count:,}")

        response = input("\nRe-download dataset? (y/n): ")
        if response.lower() != 'y':
            print("Using existing dataset.")
            return

    # Alternative: Manual download instructions
    print("=" * 60)
    print("📋 DATASET DOWNLOAD OPTIONS")
    print("=" * 60)
    print()
    print("Option 1: Kaggle (Recommended)")
    print("   1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("   2. Click 'Download' (requires Kaggle account)")
    print("   3. Move downloaded file to:", DATA_DIR)
    print("   4. Run this script again to extract")
    print()
    print("Option 2: Direct Download")
    print("   1. Download from: https://data.mendeley.com/datasets/tywbtsjrjv")
    print("   2. Save to:", DATA_DIR)
    print("   3. Run this script again")
    print()
    print("=" * 60)

    # Check if user already has the zip
    if zip_path.exists():
        print(f"\n✓ Found ZIP file: {zip_path}")
        extract_zip(zip_path, DATA_DIR)

        # Count images
        image_count = sum(1 for _ in DATASET_DIR.rglob("*.jpg")) + \
                      sum(1 for _ in DATASET_DIR.rglob("*.JPG"))
        print()
        print("=" * 60)
        print("✅ Dataset Ready!")
        print("=" * 60)
        print(f"📁 Location: {DATASET_DIR}")
        print(f"📊 Total images: {image_count:,}")
        print()
        print("Next step: Run training script")
        print("   python scripts/train_models.py")
        print()
    else:
        print("\n⚠️  Please download the dataset manually (see instructions above)")
        print(f"    Save it to: {zip_path}")
        print("    Then run this script again.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
