# Model Training

This directory contains scripts for training all crop disease detection models.

## Quick Start

```bash
# Install dependencies
pip install torch torchvision matplotlib scikit-learn tqdm

# Download PlantVillage dataset
# https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

# Extract to: data/PlantVillage/
#   ├── train/
#   │   ├── Apple___Apple_scab/
#   │   └── ... (38 classes)
#   └── valid/
#       └── ... (38 classes)

# Run training
python training/train_models.py
```

## What It Does

The `train_models.py` script:

1. **Trains all 4 models** in sequence:
   - Baseline CNN
   - MobileNetV2
   - EfficientNet-B0
   - Hybrid CNN-Transformer

2. **Evaluates ensemble** performance:
   - Combines all 4 models using weighted voting
   - Tests on validation set
   - Computes real ensemble accuracy

3. **Auto-updates metrics** files:
   - `models/model_metrics.json` - Real validation metrics
   - `models/training_history/*.json` - Training curves
   - `models/class_names.json` - Class list

4. **Saves trained weights**:
   - `models/baseline_model.pth`
   - `models/efficientnet_model.pth`
   - `models/mobilenet_model.pth`
   - `models/hybrid_model.pth`

## Configuration

Edit these constants in `train_models.py`:

```python
BATCH_SIZE = 16          # Lower for smaller GPUs
NUM_EPOCHS = 5           # Increase for better results
LEARNING_RATE = 0.001
NUM_WORKERS = 4          # CPU cores for data loading
```

## Output Files

### Model Weights
```
models/
├── baseline_model.pth         # Trained weights
├── efficientnet_model.pth
├── mobilenet_model.pth
└── hybrid_model.pth
```

### Metrics (Auto-Generated)
```
models/
├── model_metrics.json         # Real validation metrics
├── class_names.json           # List of 38 classes
└── training_history/
    ├── baseline_history.json     # Training curves
    ├── efficientnet_history.json
    ├── mobilenet_history.json
    └── hybrid_history.json
```

### model_metrics.json Format

```json
{
  "baseline": {
    "accuracy": 0.9862,              // Real validation accuracy
    "precision": 0.9850,
    "recall": 0.9845,
    "f1_score": 0.9847,
    "inference_time_ms": 45,
    "parameters": "1.2M",
    "size_mb": 2.1,                  // Actual file size
    "validation_samples": 7600,
    "notes": "Real validation accuracy: 98.62% from 5 epochs training"
  },
  // ... other models ...
  "ensemble": {
    "accuracy": 0.9905,              // Real ensemble accuracy
    "precision": 0.9900,
    "recall": 0.9895,
    "f1_score": 0.9897,
    "inference_time_ms": 95.5,       // Measured on validation set
    "parameters": "33.1M",
    "size_mb": 309.4,
    "description": "Weighted ensemble of all 4 models",
    "models_combined": 4,
    "validation_samples": 7600,
    "notes": "Real ensemble performance from validation set"
  }
}
```

## Training Process

1. **Data Loading**
   - Reads from `data/PlantVillage/train/` and `data/PlantVillage/valid/`
   - Applies augmentation to training set (flip, rotation, color jitter)
   - No augmentation on validation set

2. **Training Each Model**
   - 5 epochs (configurable)
   - Adam optimizer with learning rate scheduler
   - Cross-entropy loss
   - Best model saved based on validation accuracy

3. **GPU Management**
   - Automatic GPU detection
   - 30-second cooling period between models
   - Fallback to CPU if no GPU available

4. **Ensemble Evaluation**
   - Loads all 4 trained models
   - Weighted voting based on model accuracies
   - Evaluates on full validation set
   - Measures real inference time

5. **Metrics Export**
   - Auto-generates `model_metrics.json` with real values
   - Saves training curves to `training_history/`
   - Ready to use in web application immediately

## Resume Training

The script saves checkpoints every 5 epochs in `results/checkpoints/`. To resume after interruption, simply run the script again - it will detect existing checkpoints.

## Expected Training Time

On RTX 3070:
- Baseline CNN: ~20 minutes
- MobileNetV2: ~40 minutes
- EfficientNet-B0: ~50 minutes
- Hybrid: ~2-3 hours

Total: ~4-5 hours for all models

## After Training

The web application will automatically use your trained models:

1. **Pull latest code** (if working with git)
2. **Restart backend**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```
3. **Check logs**:
   ```bash
   docker logs crop-disease-backend
   ```
   You should see:
   ```
   ✓ Successfully loaded trained weights from /app/models/baseline_model.pth
   ✓ Successfully loaded trained weights from /app/models/efficientnet_model.pth
   ✓ Successfully loaded trained weights from /app/models/mobilenet_model.pth
   ✓ Successfully loaded trained weights from /app/models/hybrid_model.pth
   ✓ Loaded model metrics from /app/models/model_metrics.json
   ```

4. **Visit stats page**: `http://localhost/stats.html`
   - Shows real validation accuracies
   - Displays actual model sizes
   - Shows ensemble performance

## Troubleshooting

### CUDA out of memory
- Reduce `BATCH_SIZE` from 16 to 8 or 4
- Train one model at a time

### Dataset not found
- Ensure structure is:
  ```
  data/PlantVillage/
  ├── train/
  │   ├── Apple___Apple_scab/
  │   └── ... (38 folders)
  └── valid/
      └── ... (38 folders)
  ```

### Models not loading in web app
- Check file names end with `_model.pth`
- Verify files are in `models/` directory
- Restart Docker containers

## Advanced: Custom Training

To train with different hyperparameters:

```python
# Edit train_models.py
BATCH_SIZE = 32           # Larger batch
NUM_EPOCHS = 20           # More epochs
LEARNING_RATE = 0.0001    # Lower LR

# Add early stopping
# Modify optimizer
# Change data augmentation
# etc.
```

## Notes

- Training script is GPU-optimized but works on CPU (slower)
- Models are saved with best validation accuracy
- Ensemble metrics are computed on same validation set
- All metrics are automatically used by web application
- No manual updates needed to `model_metrics.json`
