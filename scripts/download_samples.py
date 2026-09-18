#!/usr/bin/env python3
"""
Sample Images Downloader for Crop Disease Detection System
Downloads sample plant disease images for testing and demonstration

This script helps you download sample images from the PlantVillage dataset
or provides links to download them manually.
"""

import os
import sys


def print_banner():
    print("=" * 70)
    print("  Sample Images Downloader - Crop Disease Detection System")
    print("=" * 70)
    print()


def print_instructions():
    print("📸 SAMPLE IMAGES SETUP INSTRUCTIONS")
    print()
    print("Since we cannot automatically download copyrighted images, here are")
    print("three ways to get sample images for testing:")
    print()
    print("=" * 70)
    print()

    # Option 1
    print("OPTION 1: Use Your Own Images (Recommended for Demo)")
    print("-" * 70)
    print("✓ Take photos of plant leaves with your phone/camera")
    print("✓ Download from free stock photo sites:")
    print("   - Unsplash: https://unsplash.com/s/photos/plant-disease")
    print("   - Pexels: https://www.pexels.com/search/plant%20disease/")
    print("   - Pixabay: https://pixabay.com/images/search/plant%20disease/")
    print()
    print("✓ Save images to: data/sample_images/")
    print("✓ Supported formats: JPG, PNG")
    print()

    # Option 2
    print("OPTION 2: PlantVillage Dataset (For Training/Research)")
    print("-" * 70)
    print(
        "1. Visit Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
    )
    print("2. Download the dataset (requires Kaggle account - free)")
    print("3. Extract sample images to data/sample_images/")
    print()
    print("Recommended samples to include:")
    print("  • Apple - Apple scab (2-3 images)")
    print("  • Apple - Black rot (2-3 images)")
    print("  • Apple - Healthy (2-3 images)")
    print("  • Tomato - Early blight (2-3 images)")
    print("  • Tomato - Late blight (2-3 images)")
    print("  • Tomato - Healthy (2-3 images)")
    print("  • Potato - Early blight (2-3 images)")
    print("  • Corn - Common rust (2-3 images)")
    print()

    # Option 3
    print("OPTION 3: Quick Test Images (Create Placeholder)")
    print("-" * 70)
    print("For quick testing without real images, you can:")
    print("1. Use any plant/leaf photo you have")
    print("2. The system will still demonstrate all features")
    print("3. Predictions won't be accurate but functionality works")
    print()
    print("=" * 70)
    print()


def create_directories():
    """Create necessary directories for sample images"""
    dirs = ["data/sample_images", "data/field_images", "data/uploads"]

    print("📁 Creating directories...")
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ {dir_path}")
    print()


def create_readme():
    """Create README in sample images directory"""
    readme_path = "data/sample_images/README.md"

    readme_content = """# Sample Images for Testing

## Quick Start

Place your test images in this directory. Recommended structure:

```
sample_images/
├── apple_scab_01.jpg
├── apple_scab_02.jpg
├── tomato_early_blight_01.jpg
├── tomato_late_blight_01.jpg
├── potato_early_blight_01.jpg
├── healthy_leaf_01.jpg
└── ...
```

## Image Requirements

- **Format:** JPG or PNG
- **Size:** At least 224×224 pixels (higher is better)
- **File Size:** Maximum 16MB
- **Quality:** Clear, well-lit photos of leaves

## Where to Get Images

### Free Stock Photos
- **Unsplash:** https://unsplash.com/s/photos/plant-disease
- **Pexels:** https://www.pexels.com/search/plant%20leaf/
- **Pixabay:** https://pixabay.com/images/search/plant/

### Research Dataset
- **PlantVillage:** https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
  (Requires free Kaggle account)

### Your Own Photos
- Take photos of your own plants
- Ensure good lighting and focus
- Capture the affected area clearly

## Recommended Sample Set (15 images)

1. **Apple Diseases** (3 images)
   - Apple scab
   - Black rot
   - Healthy apple leaf

2. **Tomato Diseases** (4 images)
   - Early blight
   - Late blight
   - Leaf mold
   - Healthy tomato leaf

3. **Potato Diseases** (3 images)
   - Early blight
   - Late blight
   - Healthy potato leaf

4. **Other Crops** (5 images)
   - Corn common rust
   - Grape black rot
   - Pepper bacterial spot
   - Any other plant disease
   - Healthy plant

## For Thesis Defense

Having 10-15 diverse sample images will allow you to demonstrate:
- Different disease types
- Model accuracy across crops
- Explainable AI visualizations
- Model comparison features

## Testing the System

Once you have images:
1. Start the backend: `cd backend && python app.py`
2. Open frontend: `frontend/index.html`
3. Go to "Analyze" page
4. Upload your sample images
5. Test different models and explanation methods
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"  ✓ Created {readme_path}")
    print()


def print_next_steps():
    print("🎯 NEXT STEPS")
    print("-" * 70)
    print("1. Download or create 10-15 sample images")
    print("2. Place them in: data/sample_images/")
    print("3. Start the backend server:")
    print("   cd backend")
    print("   python app.py")
    print()
    print("4. Open the frontend in your browser:")
    print("   frontend/index.html")
    print()
    print("5. Test the system with your images!")
    print()
    print("=" * 70)
    print()
    print("✨ Your application is ready to use!")
    print("📧 For questions, refer to README.md or SETUP.md")
    print()


def main():
    print_banner()
    create_directories()
    create_readme()
    print_instructions()
    print_next_steps()


if __name__ == "__main__":
    main()
