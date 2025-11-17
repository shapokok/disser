# Training History

This directory contains detailed training history for each model, including:
- Loss curves (training and validation)
- Accuracy curves (training and validation)
- Per-epoch timing information
- Best performing epoch
- Training metadata

## File Format

Each `{model_name}_history.json` file contains:

```json
{
  "model_name": "...",
  "architecture": "...",
  "num_classes": 38,
  "train_losses": [...],        // Loss per epoch
  "val_losses": [...],          // Validation loss per epoch
  "train_accuracies": [...],    // Training accuracy per epoch
  "val_accuracies": [...],      // Validation accuracy per epoch
  "epoch_times": [...],         // Time per epoch in seconds
  "best_val_accuracy": 0.989,   // Best validation accuracy achieved
  "best_epoch": 5,              // Epoch with best validation accuracy
  "final_train_accuracy": 0.982,
  "total_epochs": 5,
  "total_time_minutes": 47.67,
  "training_date": "2025-11-18"
}
```

## Usage

### In Python Backend

```python
import json
import os

def load_training_history(model_name):
    history_file = f'../models/training_history/{model_name}_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return None

# Example: Get EfficientNet training history
history = load_training_history('efficientnet')
if history:
    print(f"Best validation accuracy: {history['best_val_accuracy']:.2%}")
    print(f"Achieved at epoch: {history['best_epoch']}")
```

### Visualization

Use this data to create training curves showing:
- Loss over epochs
- Accuracy over epochs
- Training time analysis
- Overfitting detection (train vs. val accuracy gap)

## Adding New Model History

When training a new model, save its history in the same format:

```python
import json

history = {
    "model_name": "your_model",
    "train_losses": [...],
    "val_losses": [...],
    # ... other fields
}

with open('models/training_history/your_model_history.json', 'w') as f:
    json.dump(history, f, indent=2)
```
