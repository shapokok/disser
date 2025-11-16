#!/usr/bin/env python3
"""
Training script for crop disease classification models
Trains all 4 architectures: BaselineCNN, EfficientNet, MobileNet, Hybrid CNN-Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json
import time
from datetime import datetime
import sys

# Import model architectures
sys.path.append('backend')
from model import BaselineCNN, EfficientNetModel, MobileNetModel, HybridCNNTransformer


# =======================
# Configuration
# =======================

CONFIG = {
    'data_dir': 'data/PlantVillage',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'early_stopping_patience': 10,
    'save_best_only': True,

    # Image preprocessing
    'image_size': 224,
    'normalize_mean': [0.485, 0.456, 0.406],  # ImageNet stats
    'normalize_std': [0.229, 0.224, 0.225],

    # Models to train
    'models': {
        'baseline': {'name': 'baseline_model', 'class': BaselineCNN, 'lr': 0.001},
        'efficientnet': {'name': 'efficientnet_model', 'class': EfficientNetModel, 'lr': 0.0001},
        'mobilenet': {'name': 'mobilenet_model', 'class': MobileNetModel, 'lr': 0.0001},
        'hybrid': {'name': 'hybrid_model', 'class': HybridCNNTransformer, 'lr': 0.00005},
    }
}


# =======================
# Data Loading
# =======================

def get_data_transforms(image_size=224):
    """Create data augmentation transforms for train and validation"""

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG['normalize_mean'], CONFIG['normalize_std'])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG['normalize_mean'], CONFIG['normalize_std'])
    ])

    return train_transform, valid_transform


def load_datasets(data_dir, batch_size, num_workers=4):
    """Load training and validation datasets"""

    data_path = Path(data_dir)
    train_dir = data_path / 'train'
    valid_dir = data_path / 'valid'

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if not valid_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}\n"
                                f"Please run: python scripts/split_train_valid.py")

    print(f"Loading datasets from {data_dir}")

    train_transform, valid_transform = get_data_transforms(CONFIG['image_size'])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Valid samples: {len(valid_dataset)}")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Batch size: {batch_size}\n")

    return train_loader, valid_loader, class_names, num_classes


# =======================
# Training Functions
# =======================

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.0 * correct / total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, valid_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def train_model(model, model_name, train_loader, valid_loader, num_epochs,
                learning_rate, device, models_dir, logs_dir):
    """Train a single model"""

    print("=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])

    # Move model to device
    model = model.to(device)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'learning_rates': []
    }

    best_valid_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] LR: {current_lr:.6f}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        # Update learning rate
        scheduler.step(valid_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['learning_rates'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")
        print(f"  Epoch Time: {epoch_time:.2f}s")

        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch + 1

            if CONFIG['save_best_only']:
                model_path = Path(models_dir) / f"{model_name}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"  ✓ Saved best model: {model_path} (Acc: {valid_acc:.2f}%)")

        # Early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
            break

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"Training completed for {model_name}")
    print(f"Best Valid Acc: {best_valid_acc:.2f}% at epoch {best_epoch}")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print("=" * 80 + "\n")

    # Save training history
    history_path = Path(logs_dir) / f"{model_name}_history.json"
    history['best_valid_acc'] = best_valid_acc
    history['best_epoch'] = best_epoch
    history['total_time'] = total_time
    history['model_name'] = model_name

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


# =======================
# Main Training Pipeline
# =======================

def main():
    """Main training pipeline for all models"""

    print("=" * 80)
    print("🚀 Crop Disease Classification - Model Training")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Create directories
    models_dir = Path(CONFIG['models_dir'])
    logs_dir = Path(CONFIG['logs_dir'])
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Load data
    try:
        train_loader, valid_loader, class_names, num_classes = load_datasets(
            CONFIG['data_dir'],
            CONFIG['batch_size'],
            CONFIG['num_workers']
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Save class names
    class_names_file = models_dir / 'class_names.json'
    with open(class_names_file, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ Saved class names to {class_names_file}\n")

    # Training results
    all_results = {}

    # Train each model
    for model_type, config in CONFIG['models'].items():
        print(f"\n{'=' * 80}")
        print(f"🔧 Preparing to train: {model_type.upper()}")
        print(f"{'=' * 80}\n")

        # Create model
        if model_type == 'baseline':
            model = config['class'](num_classes=num_classes)
        elif model_type in ['efficientnet', 'mobilenet']:
            model = config['class'](num_classes=num_classes, pretrained=True)
        elif model_type == 'hybrid':
            model = config['class'](num_classes=num_classes)
        else:
            print(f"❌ Unknown model type: {model_type}")
            continue

        # Train model
        try:
            trained_model, history = train_model(
                model=model,
                model_name=config['name'],
                train_loader=train_loader,
                valid_loader=valid_loader,
                num_epochs=CONFIG['num_epochs'],
                learning_rate=config['lr'],
                device=device,
                models_dir=models_dir,
                logs_dir=logs_dir
            )

            all_results[model_type] = {
                'best_acc': history['best_valid_acc'],
                'best_epoch': history['best_epoch'],
                'total_time': history['total_time']
            }

        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print final summary
    print("\n" + "=" * 80)
    print("📊 Training Summary")
    print("=" * 80)
    print(f"{'Model':<20} {'Best Accuracy':<15} {'Best Epoch':<12} {'Time (min)':<12}")
    print("-" * 80)

    for model_type, results in all_results.items():
        print(f"{model_type:<20} {results['best_acc']:>13.2f}% "
              f"{results['best_epoch']:>11} {results['total_time']/60:>11.2f}")

    print("=" * 80)
    print(f"\n✅ Training completed!")
    print(f"Models saved to: {models_dir.absolute()}")
    print(f"Logs saved to: {logs_dir.absolute()}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
