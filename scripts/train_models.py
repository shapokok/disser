#!/usr/bin/env python3
"""
Training script for all 4 models with checkpoint support
Supports resume, GPU monitoring, and automatic metric export
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import BaselineCNN, EfficientNetModel, MobileNetModel, HybridCNNTransformer

# Configuration
BATCH_SIZE = 16  # Lower for RTX 3070 to prevent overheating
NUM_EPOCHS = 30  # Can increase to 50 for better results
LEARNING_RATE = 0.001
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class TrainingLogger:
    """Log training progress and metrics"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epoch_times = []

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.epoch_times.append(epoch_time)

    def plot_metrics(self):
        """Generate and save metric plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} Training Metrics', fontsize=16)

        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Epoch time plot
        axes[1, 0].bar(epochs, self.epoch_times, color='green', alpha=0.7)
        axes[1, 0].set_title('Time per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)

        # Summary stats
        summary_text = f"""
        Best Train Acc: {max(self.train_accs):.2f}%
        Best Val Acc: {max(self.val_accs):.2f}%
        Final Train Loss: {self.train_losses[-1]:.4f}
        Final Val Loss: {self.val_losses[-1]:.4f}
        Avg Time/Epoch: {np.mean(self.epoch_times):.1f}s
        Total Time: {sum(self.epoch_times)/60:.1f} min
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                        verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plot_path = METRICS_DIR / f'{self.model_name}_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Metrics plot saved: {plot_path}")

    def save_json(self):
        """Save metrics as JSON"""
        metrics = {
            'model_name': self.model_name,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accs,
            'val_accuracies': self.val_accs,
            'epoch_times': self.epoch_times,
            'best_val_accuracy': max(self.val_accs),
            'final_train_accuracy': self.train_accs[-1],
            'total_time_minutes': sum(self.epoch_times) / 60
        }

        json_path = METRICS_DIR / f'{self.model_name}_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"✓ Metrics JSON saved: {json_path}")


def get_data_loaders():
    """Prepare data loaders with augmentation"""
    print("📁 Loading dataset...")

    # Check for pre-split dataset (train/valid folders)
    train_dir = DATA_DIR / "train"
    valid_dir = DATA_DIR / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        print(f"❌ Dataset not found!")
        print(f"   Expected structure:")
        print(f"   {DATA_DIR}/")
        print(f"   ├── train/")
        print(f"   │   ├── Apple___Apple_scab/")
        print(f"   │   └── ... (38 classes)")
        print(f"   └── valid/")
        print(f"       └── ... (38 classes)")
        print()
        print("   Download from: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        sys.exit(1)

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load train and validation datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(valid_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)

    print(f"✓ Found {len(train_dataset)} training images")
    print(f"✓ Found {len(val_dataset)} validation images")
    print(f"✓ Number of classes: {num_classes}")

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # Save class names
    class_names_path = MODELS_DIR / "class_names.json"
    with open(class_names_path, 'w') as f:
        json.dump(train_dataset.classes, f, indent=2)
    print(f"✓ Class names saved: {class_names_path}")

    return train_loader, val_loader, num_classes


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/len(val_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, model_name, best_val_acc):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }

    checkpoint_path = CHECKPOINT_DIR / f'{model_name}_checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Keep only last 3 checkpoints
    checkpoints = sorted(CHECKPOINT_DIR.glob(f'{model_name}_checkpoint_*.pth'))
    for old_checkpoint in checkpoints[:-3]:
        old_checkpoint.unlink()

    return checkpoint_path


def load_checkpoint(model, optimizer, model_name):
    """Load latest checkpoint if exists"""
    checkpoints = sorted(CHECKPOINT_DIR.glob(f'{model_name}_checkpoint_*.pth'))

    if not checkpoints:
        return 0, 0.0

    latest_checkpoint = checkpoints[-1]
    print(f"📂 Loading checkpoint: {latest_checkpoint.name}")

    checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['best_val_acc']


def train_model(model_name, model, num_classes, train_loader, val_loader, resume=True):
    """Train a single model"""
    print("\n" + "=" * 70)
    print(f"🚀 Training {model_name}")
    print("=" * 70)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    logger = TrainingLogger(model_name)

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0

    if resume:
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, model_name)
        if start_epoch > 0:
            print(f"✓ Resuming from epoch {start_epoch} (best val acc: {best_val_acc:.2f}%)")

    # Training loop
    print(f"\n⏱️  Training on: {DEVICE}")
    print(f"📊 Batch size: {BATCH_SIZE}")
    print(f"📈 Epochs: {NUM_EPOCHS} (starting from {start_epoch + 1})")
    print()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()

        print(f"\n📍 Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Log metrics
        logger.log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc, epoch_time)

        # Print summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.1f}s")

        # Update learning rate
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != prev_lr:
            print(f"📉 Learning rate reduced: {prev_lr:.6f} → {new_lr:.6f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, model_name, best_val_acc)
            print(f"💾 Checkpoint saved: {checkpoint_path.name}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = MODELS_DIR / f'{model_name}_model.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best! Saved to: {best_model_path.name}")

    # Training complete
    print("\n" + "=" * 70)
    print(f"✅ {model_name} Training Complete!")
    print("=" * 70)
    print(f"🏆 Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"⏱️  Total Time: {sum(logger.epoch_times)/60:.1f} minutes")

    # Save final model
    final_model_path = MODELS_DIR / f'{model_name}_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved: {final_model_path}")

    # Generate plots
    logger.plot_metrics()
    logger.save_json()

    return best_val_acc


def main():
    print("=" * 70)
    print("🌱 CROP DISEASE DETECTION - MODEL TRAINING")
    print("=" * 70)
    print()

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU found, using CPU (will be slower)")

    print()

    # Load data
    train_loader, val_loader, num_classes = get_data_loaders()

    # Define models
    models_to_train = [
        ("baseline", BaselineCNN(num_classes=num_classes)),
        ("mobilenet", MobileNetModel(num_classes=num_classes)),
        ("efficientnet", EfficientNetModel(num_classes=num_classes)),
        ("hybrid", HybridCNNTransformer(num_classes=num_classes))
    ]

    # Train all models
    results = {}

    for model_name, model in models_to_train:
        try:
            best_acc = train_model(model_name, model, num_classes, train_loader, val_loader)
            results[model_name] = best_acc

            # Cool down GPU (prevent overheating)
            if DEVICE.type == 'cuda':
                print("\n❄️  Cooling down GPU for 30 seconds...")
                time.sleep(30)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Training interrupted for {model_name}")
            save_checkpoint(model, None, 0, model_name, 0.0)
            print("   Progress saved. You can resume later.")
            sys.exit(0)

        except Exception as e:
            print(f"\n\n❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print final summary
    print("\n" + "=" * 70)
    print("🏆 TRAINING SUMMARY")
    print("=" * 70)

    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name.ljust(15)}: {acc:.2f}%")

    print("\n✅ All models trained successfully!")
    print(f"📁 Models saved in: {MODELS_DIR}")
    print(f"📊 Metrics saved in: {METRICS_DIR}")


if __name__ == "__main__":
    main()
