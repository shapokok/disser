# Models Directory

This directory contains the trained models for the Crop Disease Detection system.

## Adding Your Trained Models

Place your trained model files (`.pth`) in this directory. The system supports multiple naming conventions:

### Recommended Naming (Priority 1)
- `baseline_model.pth` - Baseline CNN model
- `efficientnet_model.pth` - EfficientNet-B0 model
- `mobilenet_model.pth` - MobileNetV2 model
- `hybrid_model.pth` - Hybrid CNN-Transformer model

### Alternative Naming (Priority 2)
- `baseline.pth`
- `efficientnet.pth`
- `mobilenet.pth`
- `hybrid.pth`

### Also Supported (Priority 3)
- `model_baseline.pth`
- `model_efficientnet.pth`
- `model_mobilenet.pth`
- `model_hybrid.pth`

## Model Checkpoint Format

The system supports two checkpoint formats:

### 1. State Dict Only (Recommended)
```python
torch.save(model.state_dict(), 'baseline_model.pth')
```

### 2. Full Checkpoint
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'baseline_model.pth')
```

Or with key name `state_dict`:
```python
torch.save({
    'state_dict': model.state_dict(),
    'epoch': epoch,
    # ... other training info
}, 'baseline_model.pth')
```

## Model Architectures

Make sure your saved models match these architectures:

### Baseline CNN
- Custom 4-block CNN with BatchNorm
- Input: 224x224x3
- Output: 38 classes (PlantVillage dataset)

### EfficientNet
- EfficientNet-B0 backbone
- Modified classifier head for 38 classes
- Pre-trained on ImageNet

### MobileNet
- MobileNetV2 backbone
- Modified classifier head for 38 classes
- Pre-trained on ImageNet

### Hybrid CNN-Transformer
- ResNet-50 CNN backbone
- Transformer encoder (2 layers, 8 heads)
- Combined architecture for 38 classes

## Verification

After adding your models to this directory:

1. **Restart the Docker containers** (if using Docker):
   ```bash
   docker-compose down
   docker-compose up -d
   ```

2. **Or restart the backend** (if running locally):
   ```bash
   cd backend
   python app.py
   ```

3. **Check the logs** for confirmation:
   - ✓ Successfully loaded trained weights from...
   - Or warnings if models weren't found

## Model Performance Metrics

To display real validation metrics instead of placeholder values:

### Option 1: Manual Entry (Quick)
Edit `model_metrics.json` and replace the values with your actual validation results:

```json
{
  "baseline": {
    "accuracy": 0.892,
    "precision": 0.885,
    "recall": 0.878,
    "f1_score": 0.881,
    ...
  }
}
```

### Option 2: Automatic Computation (Recommended)
If you have a validation dataset, run the metrics computation script:

```bash
cd backend
python compute_metrics.py --val_dir /path/to/validation/dataset
```

The script will:
- Evaluate all models on your validation set
- Compute accuracy, precision, recall, F1-score
- Measure actual inference times
- Calculate model sizes and parameter counts
- Save results to `model_metrics.json`

**Validation dataset structure:**
```
validation/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
├── Apple___Black_rot/
│   ├── image1.jpg
├── ...
```

After updating metrics, restart the server to see real statistics on the stats page.

## Training History

Detailed training curves and metadata are stored in `training_history/`:

```
training_history/
├── baseline_history.json
├── efficientnet_history.json
├── mobilenet_history.json
└── README.md
```

Each file contains:
- Loss curves (train/validation per epoch)
- Accuracy curves (train/validation per epoch)
- Training times
- Best epoch information

Access via API:
- `/api/training_history/<model_name>` - Get specific model history
- `/api/training_history/all` - Get all training histories

## Required Files

- `class_names.json` - List of 38 disease classes (auto-generated)
- `model_metrics.json` - Validation metrics (optional, defaults provided)
- `training_history/*.json` - Training curves and history (optional)
- Your trained `.pth` model files

## Notes

- Models are loaded on server startup
- If a trained model is not found, the system will use:
  - ImageNet pre-trained weights (for EfficientNet, MobileNet, Hybrid)
  - Random initialization (for Baseline CNN)
- All models are automatically moved to GPU if available
- Models are set to evaluation mode (`model.eval()`)
- Statistics are loaded from `model_metrics.json` if available
