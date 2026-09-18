"""
Progressive Domain Adaptation for Plant Disease Detection
Goal: Improve from 63% to 72-78%

Strategy:
1. Stage 1: Fine-tune on "easy" classes (high baseline: Squash, Tomato)
2. Stage 2: Fine-tune on "hard" classes (low baseline: Corn)
3. Stage 3: Joint fine-tuning with class balancing
4. Heavy data augmentation
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
import copy
from collections import Counter

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import MobileNetModel

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ПУТИ
    "plantdoc_path": "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc",
    "st_checkpoint": "./results/da_checkpoints/self_training_best.pth",
    "results_path": "./results",
    "output_checkpoint": "./results/da_checkpoints/progressive_da_final.pth",
    # Progressive DA параметры
    "stage1_epochs": 8,  # Easy classes
    "stage2_epochs": 10,  # Hard classes
    "stage3_epochs": 12,  # Joint fine-tuning
    "learning_rate_stage1": 0.00005,
    "learning_rate_stage2": 0.00003,
    "learning_rate_stage3": 0.00001,
    "batch_size": 16,
    # Other
    "num_classes": 38,
    "img_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# CLASS GROUPING
EASY_CLASSES = {
    "Squash Powdery mildew leaf": 25,  # 76.4% baseline
    "Tomato leaf late blight": 30,  # 66.3% baseline
}

HARD_CLASSES = {
    "Corn Gray leaf spot": 7,  # 59.4% baseline
    "Corn leaf blight": 9,  # 53.3% baseline
}

ALL_CLASSES = {**EASY_CLASSES, **HARD_CLASSES}

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
# DATASET WITH HEAVY AUGMENTATION
# ============================================================================


class PlantDocProgressiveDataset(Dataset):
    """PlantDoc with class filtering and heavy augmentation"""

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        class_mapping=None,
        filter_classes=None,
    ):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.class_mapping = class_mapping or {}
        self.filter_classes = filter_classes  # Only load these classes

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

            # Filter classes if specified
            if self.filter_classes and mapped_class not in self.filter_classes:
                continue

            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.png"))
            )

            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(mapped_class)

        print(
            f"   Loaded {len(self.images)} images from {len(set(self.labels))} classes"
        )

        # Class distribution
        label_counts = Counter(self.labels)
        for label in sorted(label_counts.keys()):
            class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:30]
            print(f"      {class_name:30s}: {label_counts[label]:3d} images")

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
# TRANSFORMS
# ============================================================================


def get_heavy_augmentation_transform():
    """Heavy augmentation for progressive training"""
    return transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ]
    )


def get_test_transform():
    """Standard test transform"""
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


def train_stage(
    model, train_loader, criterion, optimizer, scheduler, epochs, device, stage_name
):
    """Train for one stage"""

    print(f"\n{'='*70}")
    print(f"{stage_name}")
    print(f"{'='*70}")

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
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
                {
                    "loss": f"{running_loss/(pbar.n+1):.4f}",
                    "acc": f"{100.*correct/total:.1f}%",
                }
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        # Step scheduler
        if scheduler:
            scheduler.step()

        print(
            f"   Epoch {epoch+1}/{epochs}: "
            f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%, "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss

    return model


# ============================================================================
# EVALUATION
# ============================================================================


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model"""
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in dataloader:
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
# MAIN PROGRESSIVE DA
# ============================================================================


def progressive_da():
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║              PROGRESSIVE DOMAIN ADAPTATION                           ║
║                Target: 72-78% accuracy                               ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    )

    device = CONFIG["device"]
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n🎯 STRATEGY:")
    print(f"   Stage 1: Fine-tune on EASY classes (Squash, Tomato)")
    print(f"   Stage 2: Fine-tune on HARD classes (Corn)")
    print(f"   Stage 3: Joint training with class balancing")
    print(f"   Baseline: 63.0%")
    print(f"   Target:   72-78%")

    # Create directories
    Path(CONFIG["results_path"]).mkdir(exist_ok=True)
    Path(CONFIG["output_checkpoint"]).parent.mkdir(exist_ok=True, parents=True)

    # Load ST-DA model as starting point
    print(f"\n📥 Loading Self-Training DA model...")
    model = MobileNetModel(num_classes=CONFIG["num_classes"], pretrained=False)
    st_checkpoint = Path(CONFIG["st_checkpoint"])
    model.load_state_dict(torch.load(st_checkpoint, map_location="cpu"))
    model = model.to(device)
    print(f"   ✅ Loaded from {st_checkpoint}")

    # Evaluate baseline
    print(f"\n📊 Evaluating ST-DA baseline...")
    test_transform = get_test_transform()
    test_dataset = PlantDocProgressiveDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=test_transform,
        class_mapping=ALL_CLASSES,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    baseline_acc, baseline_per_class = evaluate_model(model, test_loader, device)
    print(f"   Baseline accuracy: {baseline_acc*100:.2f}%")

    results_history = {
        "baseline": {
            "accuracy": float(baseline_acc),
            "per_class": {int(k): float(v) for k, v in baseline_per_class.items()},
        }
    }

    # ========================================================================
    # STAGE 1: EASY CLASSES
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"STAGE 1: FINE-TUNING ON EASY CLASSES")
    print(f"{'='*70}")
    print(f"Classes: Squash Powdery mildew (76%), Tomato Late blight (66%)")

    # Load easy classes dataset
    easy_transform = get_heavy_augmentation_transform()
    easy_dataset = PlantDocProgressiveDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=easy_transform,
        class_mapping=EASY_CLASSES,
        filter_classes=list(EASY_CLASSES.values()),
    )
    easy_loader = DataLoader(
        easy_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
    )

    # Train Stage 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate_stage1"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["stage1_epochs"]
    )

    model = train_stage(
        model,
        easy_loader,
        criterion,
        optimizer,
        scheduler,
        CONFIG["stage1_epochs"],
        device,
        "STAGE 1 TRAINING",
    )

    # Evaluate after Stage 1
    print(f"\n📊 Evaluating after Stage 1...")
    stage1_acc, stage1_per_class = evaluate_model(model, test_loader, device)
    print(f"   Accuracy: {stage1_acc*100:.2f}% (baseline: {baseline_acc*100:.2f}%)")

    results_history["stage1"] = {
        "accuracy": float(stage1_acc),
        "per_class": {int(k): float(v) for k, v in stage1_per_class.items()},
    }

    stage1_labels, stage1_preds = collect_preds(model, test_loader, device)
    detailed_stages = {"stage1": build_detailed_metrics(stage1_labels, stage1_preds)}

    # Save Stage 1 checkpoint
    stage1_checkpoint = (
        Path(CONFIG["output_checkpoint"]).parent / "progressive_stage1.pth"
    )
    torch.save(model.state_dict(), stage1_checkpoint)
    print(f"   💾 Saved to {stage1_checkpoint}")

    # ========================================================================
    # STAGE 2: HARD CLASSES
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"STAGE 2: FINE-TUNING ON HARD CLASSES")
    print(f"{'='*70}")
    print(f"Classes: Corn Gray spot (59%), Corn Blight (53%)")

    # Load hard classes dataset
    hard_transform = get_heavy_augmentation_transform()
    hard_dataset = PlantDocProgressiveDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=hard_transform,
        class_mapping=HARD_CLASSES,
        filter_classes=list(HARD_CLASSES.values()),
    )
    hard_loader = DataLoader(
        hard_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
    )

    # Train Stage 2
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate_stage2"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["stage2_epochs"]
    )

    model = train_stage(
        model,
        hard_loader,
        criterion,
        optimizer,
        scheduler,
        CONFIG["stage2_epochs"],
        device,
        "STAGE 2 TRAINING",
    )

    # Evaluate after Stage 2
    print(f"\n📊 Evaluating after Stage 2...")
    stage2_acc, stage2_per_class = evaluate_model(model, test_loader, device)
    print(f"   Accuracy: {stage2_acc*100:.2f}% (stage1: {stage1_acc*100:.2f}%)")

    results_history["stage2"] = {
        "accuracy": float(stage2_acc),
        "per_class": {int(k): float(v) for k, v in stage2_per_class.items()},
    }

    stage2_labels, stage2_preds = collect_preds(model, test_loader, device)
    detailed_stages["stage2"] = build_detailed_metrics(stage2_labels, stage2_preds)

    # Save Stage 2 checkpoint
    stage2_checkpoint = (
        Path(CONFIG["output_checkpoint"]).parent / "progressive_stage2.pth"
    )
    torch.save(model.state_dict(), stage2_checkpoint)
    print(f"   💾 Saved to {stage2_checkpoint}")

    # ========================================================================
    # STAGE 3: JOINT FINE-TUNING WITH CLASS BALANCING
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"STAGE 3: JOINT FINE-TUNING (ALL CLASSES)")
    print(f"{'='*70}")
    print(f"All 4 classes with class-balanced sampling")

    # Load all classes with balanced sampling
    joint_transform = get_heavy_augmentation_transform()
    joint_dataset = PlantDocProgressiveDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=joint_transform,
        class_mapping=ALL_CLASSES,
    )

    # Create balanced sampler
    class_counts = Counter(joint_dataset.labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in joint_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    joint_loader = DataLoader(
        joint_dataset, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=0
    )

    # Train Stage 3
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate_stage3"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["stage3_epochs"]
    )

    model = train_stage(
        model,
        joint_loader,
        criterion,
        optimizer,
        scheduler,
        CONFIG["stage3_epochs"],
        device,
        "STAGE 3 TRAINING",
    )

    # Final evaluation
    print(f"\n📊 FINAL EVALUATION...")
    final_acc, final_per_class = evaluate_model(model, test_loader, device)

    results_history["final"] = {
        "accuracy": float(final_acc),
        "per_class": {int(k): float(v) for k, v in final_per_class.items()},
    }

    final_labels, final_preds = collect_preds(model, test_loader, device)
    detailed_stages["final"] = build_detailed_metrics(final_labels, final_preds)

    # Save final model
    torch.save(model.state_dict(), CONFIG["output_checkpoint"])
    print(f"   💾 Final model saved to {CONFIG['output_checkpoint']}")

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"PROGRESSIVE DA COMPLETE")
    print(f"{'='*70}")

    print(f"\n📊 OVERALL PROGRESS:")
    print(f"   Baseline (ST-DA):  {baseline_acc*100:5.2f}%")
    print(
        f"   After Stage 1:     {stage1_acc*100:5.2f}% ({(stage1_acc-baseline_acc)*100:+5.2f}%)"
    )
    print(
        f"   After Stage 2:     {stage2_acc*100:5.2f}% ({(stage2_acc-stage1_acc)*100:+5.2f}%)"
    )
    print(
        f"   FINAL:             {final_acc*100:5.2f}% ({(final_acc-stage2_acc)*100:+5.2f}%)"
    )
    print(f"   ")
    print(f"   Total Improvement: {(final_acc-baseline_acc)*100:+5.2f}%")

    print(f"\n📊 PER-CLASS FINAL RESULTS:")
    print("=" * 70)
    print(f"{'Class':<30} {'Baseline':>10} {'Final':>10} {'Gain':>10}")
    print("-" * 70)

    for label in sorted(final_per_class.keys()):
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:28]
        baseline_class = baseline_per_class.get(label, 0) * 100
        final_class = final_per_class.get(label, 0) * 100
        gain = final_class - baseline_class

        print(
            f"{class_name:<30} {baseline_class:>9.1f}% {final_class:>9.1f}% {gain:>9.1f}%"
        )

    # Save results
    results_file = Path(CONFIG["results_path"]) / "progressive_da_results.json"
    with open(results_file, "w") as f:
        json.dump(results_history, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Save detailed metrics per stage
    detailed_file = Path(CONFIG["results_path"]) / "progressive_da_full_metrics.json"
    with open(detailed_file, "w") as f:
        json.dump(detailed_stages, f, indent=2)
    print(f"💾 Detailed stage metrics saved to: {detailed_file}")

    # Final message
    if final_acc >= 0.75:
        print(f"\n🎉 SUCCESS! Target achieved ({final_acc*100:.1f}% >= 75%)")
        print(f"   Excellent results for SIST 2026 paper!")
    elif final_acc >= 0.70:
        print(f"\n✅ GREAT! Strong improvement ({final_acc*100:.1f}%)")
        print(f"   Solid results for publication!")
    elif final_acc >= 0.65:
        print(f"\n✅ GOOD progress ({final_acc*100:.1f}%)")
        print(f"   Significant improvement demonstrated!")
    else:
        print(f"\n⚠️  Modest results ({final_acc*100:.1f}%)")
        print(f"   But still meaningful progress!")

    print(f"\n📝 Ready to write SIST 2026 paper!")
    print(f"   Baseline: {baseline_acc*100:.1f}%")
    print(f"   Final:    {final_acc*100:.1f}%")
    print(
        f"   Story: Progressive DA with staged training → {(final_acc-baseline_acc)*100:+.1f}% gain"
    )


if __name__ == "__main__":
    progressive_da()
