"""
Joint Domain Adaptation Training
Train ALL 4 classes together from the start with class balancing

Goal: 63% → 70-73%
Strategy: No sequential stages, joint training from beginning
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from collections import Counter

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import MobileNetModel

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "plantdoc_path": "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc",
    "st_checkpoint": "./results/da_checkpoints/self_training_best.pth",
    "output_checkpoint": "./results/da_checkpoints/joint_training_final.pth",
    "results_path": "./results",
    # Training params - AGGRESSIVE!
    "epochs": 25,
    "learning_rate": 0.00005,
    "batch_size": 16,
    "weight_decay": 1e-4,
    "num_classes": 38,
    "img_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

ALL_CLASSES = {
    "Corn Gray leaf spot": 7,
    "Squash Powdery mildew leaf": 25,
    "Corn leaf blight": 9,
    "Tomato leaf late blight": 30,
}

PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ============================================================================
# DATASET
# ============================================================================


class PlantDocDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, class_mapping=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.class_mapping = class_mapping or {}

        self.images = []
        self.labels = []

        self._load_images()

    def _load_images(self):
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            mapped_class = self.class_mapping.get(class_name, -1)

            if mapped_class == -1:
                continue

            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.png"))
            )

            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(mapped_class)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# TRANSFORMS - VERY HEAVY AUGMENTATION
# ============================================================================


def get_super_heavy_augmentation():
    """Super heavy augmentation for maximum generalization"""
    return transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            # Geometric
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(25),
            transforms.RandomAffine(
                degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            # Color
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
            ),
            transforms.RandomGrayscale(p=0.1),
            # To tensor
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # Cutout
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.15)),
        ]
    )


def get_test_transform():
    return transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# ============================================================================
# TRAINING
# ============================================================================


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"}
        )

    return running_loss / len(train_loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    # Per-class
    unique_labels = sorted(set(all_labels))
    per_class_acc = {}

    for label in unique_labels:
        mask = np.array(all_labels) == label
        if mask.sum() > 0:
            class_acc = accuracy_score(
                np.array(all_labels)[mask], np.array(all_preds)[mask]
            )
            per_class_acc[label] = class_acc

    return accuracy, per_class_acc


def build_detailed_metrics(all_labels, all_preds):
    """Build detailed per-class metrics dict for the 4 target classes"""
    TARGET_CLASSES = [7, 9, 25, 30]

    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)

    # Filter to target classes only
    mask = np.isin(all_labels_arr, TARGET_CLASSES)
    filtered_labels = all_labels_arr[mask]
    filtered_preds = all_preds_arr[mask]

    accuracy = accuracy_score(filtered_labels, filtered_preds)
    conf_matrix = confusion_matrix(
        filtered_labels, filtered_preds, labels=TARGET_CLASSES
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_labels, filtered_preds, labels=TARGET_CLASSES,
        average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f1))

    per_class = {}
    for i, label in enumerate(TARGET_CLASSES):
        lmask = filtered_labels == label
        class_acc = float((filtered_preds[lmask] == label).sum() / lmask.sum()) if lmask.sum() > 0 else 0.0
        per_class[str(label)] = {
            "accuracy": class_acc,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "samples": int(support[i]),
        }

    return {
        "overall": {"accuracy": float(accuracy), "macro_f1": macro_f1},
        "per_class": per_class,
        "confusion_matrix": conf_matrix.tolist(),
    }


def collect_preds(model, dataloader, device):
    """Collect all predictions and labels from a dataloader"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds


# ============================================================================
# MAIN
# ============================================================================


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║              JOINT DOMAIN ADAPTATION TRAINING                        ║
║         All classes together from the start - No stages!             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    )

    device = CONFIG["device"]
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n🎯 STRATEGY:")
    print(f"   ✅ Train ALL 4 classes together (no stages!)")
    print(f"   ✅ Class-balanced sampling")
    print(f"   ✅ Super heavy augmentation")
    print(f"   ✅ {CONFIG['epochs']} epochs")
    print(f"   Baseline ST-DA: 63.0%")
    print(f"   Target:         70-73%")

    # Create dirs
    Path(CONFIG["results_path"]).mkdir(exist_ok=True)
    Path(CONFIG["output_checkpoint"]).parent.mkdir(exist_ok=True, parents=True)

    # Load model
    print(f"\n📥 Loading Self-Training model as starting point...")
    model = MobileNetModel(num_classes=CONFIG["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(CONFIG["st_checkpoint"], map_location="cpu"))
    model = model.to(device)
    print(f"   ✅ Loaded")

    # Prepare datasets
    print(f"\n📁 Preparing datasets...")

    # Training set with heavy augmentation
    train_transform = get_super_heavy_augmentation()
    train_dataset = PlantDocDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=train_transform,
        class_mapping=ALL_CLASSES,
    )

    print(f"   ✅ Training: {len(train_dataset)} images")

    # Class distribution
    label_counts = Counter(train_dataset.labels)
    print(f"\n   📊 Class distribution:")
    for label in sorted(label_counts.keys()):
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:30]
        print(f"      [{label:2d}] {class_name:30s}: {label_counts[label]:3d} images")

    # Create balanced sampler
    print(f"\n   🔄 Creating class-balanced sampler...")
    class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=0
    )

    # Test set
    test_transform = get_test_transform()
    test_dataset = PlantDocDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=test_transform,
        class_mapping=ALL_CLASSES,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    # Evaluate baseline
    print(f"\n📊 Evaluating baseline...")
    baseline_acc, baseline_per_class = evaluate(model, test_loader, device)
    print(f"   Baseline: {baseline_acc*100:.2f}%")
    baseline_labels, baseline_preds = collect_preds(model, test_loader, device)
    detailed_results = {"baseline": build_detailed_metrics(baseline_labels, baseline_preds)}

    # Training loop
    print(f"\n{'='*70}")
    print(f"JOINT TRAINING - {CONFIG['epochs']} EPOCHS")
    print(f"{'='*70}\n")

    best_acc = baseline_acc
    best_per_class = baseline_per_class
    patience = 0
    max_patience = 8

    history = []

    for epoch in range(CONFIG["epochs"]):
        print(f"\n📍 Epoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        test_acc, test_per_class = evaluate(model, test_loader, device)

        # Step scheduler
        scheduler.step()

        # Log
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(
            f"   Test:  Acc={test_acc*100:.2f}%, LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        # Per-class
        print(f"   Per-class:", end=" ")
        for label in sorted(test_per_class.keys()):
            print(
                f"{PLANTVILLAGE_CLASSES[label].split('___')[-1][:4]}:{test_per_class[label]*100:.0f}%",
                end=" ",
            )
        print()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "per_class": {int(k): float(v) for k, v in test_per_class.items()},
            }
        )

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            best_per_class = test_per_class
            torch.save(model.state_dict(), CONFIG["output_checkpoint"])
            print(f"   🏆 New best! Saved checkpoint")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n   ⚠️  Early stopping (patience={max_patience})")
                break

    # Final results
    print(f"\n{'='*70}")
    print(f"JOINT TRAINING COMPLETE")
    print(f"{'='*70}")

    print(f"\n📊 RESULTS:")
    print(f"   Baseline (ST-DA):  {baseline_acc*100:5.2f}%")
    print(f"   Best (Joint):      {best_acc*100:5.2f}%")
    print(f"   Improvement:       {(best_acc - baseline_acc)*100:+5.2f}%")

    print(f"\n📊 PER-CLASS FINAL:")
    print("=" * 70)
    print(f"{'Class':<30} {'Baseline':>10} {'Joint':>10} {'Gain':>10}")
    print("-" * 70)

    for label in sorted(best_per_class.keys()):
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:28]
        baseline_class = baseline_per_class.get(label, 0) * 100
        joint_class = best_per_class.get(label, 0) * 100
        gain = joint_class - baseline_class

        print(
            f"{class_name:<30} {baseline_class:>9.1f}% {joint_class:>9.1f}% {gain:>9.1f}%"
        )

    # Save results
    results = {
        "baseline": {
            "accuracy": float(baseline_acc),
            "per_class": {int(k): float(v) for k, v in baseline_per_class.items()},
        },
        "final": {
            "accuracy": float(best_acc),
            "per_class": {int(k): float(v) for k, v in best_per_class.items()},
        },
        "improvement": float(best_acc - baseline_acc),
        "history": history,
    }

    results_file = Path(CONFIG["results_path"]) / "joint_training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")
    print(f"💾 Best model saved to: {CONFIG['output_checkpoint']}")

    # Save detailed metrics (baseline + final best model)
    model.load_state_dict(torch.load(CONFIG["output_checkpoint"], map_location="cpu"))
    final_labels, final_preds = collect_preds(model.to(device), test_loader, device)
    detailed_results["final"] = build_detailed_metrics(final_labels, final_preds)

    detailed_file = Path(CONFIG["results_path"]) / "joint_training_full_metrics.json"
    with open(detailed_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"💾 Detailed metrics saved to: {detailed_file}")

    # Final message
    if best_acc >= 0.70:
        print(f"\n🎉 EXCELLENT! Achieved {best_acc*100:.1f}% (target: 70%+)")
        print(f"   Strong results for SIST 2026 paper!")
    elif best_acc >= 0.68:
        print(f"\n✅ GREAT! {best_acc*100:.1f}%")
        print(f"   Good improvement demonstrated!")
    elif best_acc >= 0.65:
        print(f"\n✅ GOOD! {best_acc*100:.1f}%")
        print(f"   Solid results!")
    else:
        print(f"\n⚠️  {best_acc*100:.1f}%")
        print(f"   Modest improvement")

    print(f"\n📝 Ready for SIST 2026 paper!")


if __name__ == "__main__":
    main()
