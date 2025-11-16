#!/usr/bin/env python3
"""
Train a single model (quick training script)
Usage: python scripts/train_single_model.py --model baseline --epochs 10
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main training script
from train import (
    CONFIG, load_datasets, train_model,
    BaselineCNN, EfficientNetModel, MobileNetModel, HybridCNNTransformer
)
import torch
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Train a single crop disease classification model')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'efficientnet', 'mobilenet', 'hybrid'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: model-specific)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick training mode (10 epochs, small batch)')

    args = parser.parse_args()

    print("=" * 80)
    print(f"🚀 Training Single Model: {args.model.upper()}")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Quick mode
    if args.quick:
        args.epochs = 10
        args.batch_size = 16
        print("⚡ Quick training mode enabled: 10 epochs, batch size 16\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Create directories
    models_dir = Path(CONFIG['models_dir'])
    logs_dir = Path(CONFIG['logs_dir'])
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Load data
    try:
        train_loader, valid_loader, class_names, num_classes = load_datasets(
            CONFIG['data_dir'],
            args.batch_size,
            CONFIG['num_workers']
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Save class names
    class_names_file = models_dir / 'class_names.json'
    with open(class_names_file, 'w') as f:
        json.dump(class_names, f, indent=2)

    # Get model configuration
    model_config = CONFIG['models'][args.model]
    learning_rate = args.lr if args.lr else model_config['lr']

    # Create model
    print(f"Creating model: {args.model}")
    if args.model == 'baseline':
        model = BaselineCNN(num_classes=num_classes)
    elif args.model == 'efficientnet':
        model = EfficientNetModel(num_classes=num_classes, pretrained=True)
    elif args.model == 'mobilenet':
        model = MobileNetModel(num_classes=num_classes, pretrained=True)
    elif args.model == 'hybrid':
        model = HybridCNNTransformer(num_classes=num_classes)

    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {args.epochs}\n")

    # Train model
    try:
        trained_model, history = train_model(
            model=model,
            model_name=model_config['name'],
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=args.epochs,
            learning_rate=learning_rate,
            device=device,
            models_dir=models_dir,
            logs_dir=logs_dir
        )

        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print(f"Best Accuracy: {history['best_valid_acc']:.2f}%")
        print(f"Best Epoch: {history['best_epoch']}")
        print(f"Model saved to: {models_dir / model_config['name']}.pth")
        print(f"History saved to: {logs_dir / model_config['name']}_history.json")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
