# Sample Images for Testing

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
