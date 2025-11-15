# Sample Images Directory

This directory should contain sample plant disease images for testing the application.

## How to Add Sample Images

### Option 1: Download from PlantVillage Dataset

1. Visit the PlantVillage dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download sample images from various disease categories
3. Place them in this directory

### Option 2: Use Your Own Images

- Take photos of plant leaves
- Ensure good lighting and focus
- Supported formats: JPG, PNG
- Recommended resolution: At least 224×224 pixels
- Maximum file size: 16MB

## Recommended Sample Images

For demonstration purposes, include at least one image from each category:

### Apple Diseases
- `apple_scab_sample.jpg`
- `apple_black_rot_sample.jpg`
- `apple_healthy_sample.jpg`

### Tomato Diseases
- `tomato_early_blight_sample.jpg`
- `tomato_late_blight_sample.jpg`
- `tomato_leaf_mold_sample.jpg`
- `tomato_healthy_sample.jpg`

### Corn Diseases
- `corn_common_rust_sample.jpg`
- `corn_healthy_sample.jpg`

### Potato Diseases
- `potato_early_blight_sample.jpg`
- `potato_late_blight_sample.jpg`
- `potato_healthy_sample.jpg`

### Other Crops
- `grape_black_rot_sample.jpg`
- `pepper_bacterial_spot_sample.jpg`
- `strawberry_leaf_scorch_sample.jpg`

## Image Naming Convention

Use descriptive names that indicate:
1. Plant type
2. Disease name (or "healthy")
3. Optional: additional identifier

Example: `tomato_early_blight_sample_01.jpg`

## Testing the Application

Once you have sample images:

1. Start the backend server: `python backend/app.py`
2. Open the frontend: `frontend/analyze.html`
3. Upload sample images
4. Test different models and explanation methods
5. Verify visualizations are generated correctly

## Quick Test Images

For quick testing, you can use any plant leaf image. The model will make predictions based on the trained weights (or random initialization for demo purposes).
