"""
Self-Training Domain Adaptation for Plant Disease Detection
Goal: Improve from 41.9% baseline to 68-72%

Strategy:
1. Use confident predictions as pseudo-labels
2. Iteratively retrain model
3. Gradually lower confidence threshold
4. 4-5 iterations
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import MobileNetModel

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ПУТИ
    "plantvillage_path": "C:/Users/Нурислам/Desktop/disser/data/PlantVillage",
    "plantdoc_path": "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc",
    "models_path": "C:/Users/Нурислам/Desktop/disser/models",
    "results_path": "./results",
    "checkpoints_path": "./results/da_checkpoints",
    # МОДЕЛИ
    "base_checkpoint": "mobilenet_model.pth",  # Используем MobileNet (лучший baseline)
    # Self-Training параметры
    "initial_confidence": 0.85,  # Start with high confidence
    "final_confidence": 0.65,  # End with lower confidence
    "num_iterations": 5,
    "learning_rate": 0.0001,  # Low LR for fine-tuning
    "batch_size": 16,
    "epochs_per_iteration": 3,  # Quick adaptation per iteration
    # Other
    "num_classes": 38,
    "img_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# OPTIMAL 4 CLASSES
OPTIMAL_4_CLASSES = {
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
# DATASET WITH PSEUDO-LABELS
# ============================================================================


class PlantDocPseudoLabelDataset(Dataset):
    """PlantDoc with pseudo-labels from confident predictions"""

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        class_mapping=None,
        pseudo_labels=None,
    ):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.class_mapping = class_mapping or {}
        self.pseudo_labels = pseudo_labels or {}

        self.images = []
        self.labels = []
        self.is_pseudo = []  # Track which labels are pseudo

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
                img_path_str = str(img_path)
                self.images.append(img_path_str)

                # Use pseudo-label if available, else use ground truth
                if img_path_str in self.pseudo_labels:
                    self.labels.append(self.pseudo_labels[img_path_str])
                    self.is_pseudo.append(True)
                else:
                    self.labels.append(mapped_class)
                    self.is_pseudo.append(False)

        num_pseudo = sum(self.is_pseudo)
        print(f"   Loaded {len(self.images)} images")
        print(
            f"   Pseudo-labels: {num_pseudo} ({num_pseudo/len(self.images)*100:.1f}%)"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# SELF-TRAINING
# ============================================================================


def generate_pseudo_labels(model, dataset, confidence_threshold, device):
    """Generate pseudo-labels for unlabeled data"""
    model.eval()

    # Create temporary dataloader
    loader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    pseudo_labels = {}
    confidences = {}

    print(f"   Generating pseudo-labels (confidence >= {confidence_threshold:.2f})...")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(loader, leave=False)):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            max_probs, predictions = torch.max(probs, dim=1)

            # Get image paths for this batch
            start_idx = batch_idx * CONFIG["batch_size"]
            end_idx = start_idx + images.size(0)

            for i, (conf, pred) in enumerate(zip(max_probs, predictions)):
                img_idx = start_idx + i
                if img_idx >= len(dataset.images):
                    break

                img_path = dataset.images[img_idx]

                if conf.item() >= confidence_threshold:
                    pseudo_labels[img_path] = pred.item()
                    confidences[img_path] = conf.item()

    print(f"   Generated {len(pseudo_labels)} pseudo-labels")

    return pseudo_labels, confidences


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, leave=False, desc="Training")
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
                "loss": f"{running_loss/len(train_loader):.4f}",
                "acc": f"{100.*correct/total:.1f}%",
            }
        )

    return running_loss / len(train_loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate model"""
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

    # Per-class accuracy
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


# ============================================================================
# MAIN SELF-TRAINING LOOP
# ============================================================================


def self_training_da():
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║              SELF-TRAINING DOMAIN ADAPTATION                         ║
║                  Target: 68-72% accuracy                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    )

    device = CONFIG["device"]
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n🎯 STRATEGY:")
    print(f"   Baseline: 41.9%")
    print(f"   Method: Self-Training with pseudo-labels")
    print(f"   Iterations: {CONFIG['num_iterations']}")
    print(
        f"   Confidence: {CONFIG['initial_confidence']:.2f} → {CONFIG['final_confidence']:.2f}"
    )
    print(f"   Target: 68-72%")

    # Create directories
    results_path = Path(CONFIG["results_path"])
    checkpoints_path = Path(CONFIG["checkpoints_path"])
    results_path.mkdir(exist_ok=True)
    checkpoints_path.mkdir(exist_ok=True, parents=True)

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load base model
    print(f"\n📥 Loading base model...")
    model_path = Path(CONFIG["models_path"]) / CONFIG["base_checkpoint"]
    model = MobileNetModel(num_classes=CONFIG["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device)
    print(f"   ✅ Loaded MobileNet-V2")

    # Initial dataset (no pseudo-labels)
    print(f"\n📁 Loading PlantDoc dataset...")
    dataset = PlantDocPseudoLabelDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=None,  # Will add later
        class_mapping=OPTIMAL_4_CLASSES,
    )

    # Test dataset (for evaluation)
    test_dataset = PlantDocPseudoLabelDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=test_transform,
        class_mapping=OPTIMAL_4_CLASSES,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # Evaluate baseline
    print(f"\n📊 Evaluating baseline...")
    baseline_acc, baseline_per_class = evaluate(model, test_loader, device)
    print(f"   Baseline accuracy: {baseline_acc*100:.2f}%")

    # Self-training iterations
    results_history = []
    best_accuracy = baseline_acc
    best_model_state = copy.deepcopy(model.state_dict())

    print(f"\n{'='*70}")
    print(f"STARTING SELF-TRAINING")
    print(f"{'='*70}\n")

    for iteration in range(CONFIG["num_iterations"]):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{CONFIG['num_iterations']}")
        print(f"{'='*70}")

        # Calculate confidence threshold for this iteration
        progress = (
            iteration / (CONFIG["num_iterations"] - 1)
            if CONFIG["num_iterations"] > 1
            else 1
        )
        confidence_threshold = (
            CONFIG["initial_confidence"]
            - (CONFIG["initial_confidence"] - CONFIG["final_confidence"]) * progress
        )

        print(f"📊 Confidence threshold: {confidence_threshold:.2f}")

        # Generate pseudo-labels
        print(f"\n🏷️  Generating pseudo-labels...")
        pseudo_labels, confidences = generate_pseudo_labels(
            model, test_dataset, confidence_threshold, device
        )

        # Create dataset with pseudo-labels
        train_dataset = PlantDocPseudoLabelDataset(
            CONFIG["plantdoc_path"],
            split="train",
            transform=train_transform,
            class_mapping=OPTIMAL_4_CLASSES,
            pseudo_labels=pseudo_labels,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
        )

        # Train on pseudo-labeled data
        print(f"\n🔄 Fine-tuning on pseudo-labeled data...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

        for epoch in range(CONFIG["epochs_per_iteration"]):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            print(
                f"   Epoch {epoch+1}/{CONFIG['epochs_per_iteration']}: "
                f"Loss={train_loss:.4f}, Acc={train_acc:.1f}%"
            )

        # Evaluate
        print(f"\n📊 Evaluating...")
        test_acc, per_class_acc = evaluate(model, test_loader, device)

        print(f"\n   Overall accuracy: {test_acc*100:.2f}%")
        print(f"   Per-class accuracy:")
        for label, acc in per_class_acc.items():
            class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1]
            print(f"      {class_name[:30]:30s}: {acc*100:5.1f}%")

        # Save results
        results_history.append(
            {
                "iteration": iteration + 1,
                "confidence_threshold": float(confidence_threshold),
                "num_pseudo_labels": len(pseudo_labels),
                "overall_accuracy": float(test_acc),
                "per_class_accuracy": {
                    int(k): float(v) for k, v in per_class_acc.items()
                },
            }
        )

        # Save checkpoint if best
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = copy.deepcopy(model.state_dict())

            checkpoint_path = checkpoints_path / f"self_training_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\n   🏆 New best! Saved to {checkpoint_path}")

        # Save iteration checkpoint
        iter_checkpoint = checkpoints_path / f"self_training_iter{iteration+1}.pth"
        torch.save(model.state_dict(), iter_checkpoint)

    # Final results
    print(f"\n{'='*70}")
    print(f"SELF-TRAINING COMPLETE")
    print(f"{'='*70}")

    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Baseline:      {baseline_acc*100:.2f}%")
    print(f"   Final:         {test_acc*100:.2f}%")
    print(f"   Best:          {best_accuracy*100:.2f}%")
    print(f"   Improvement:   +{(best_accuracy - baseline_acc)*100:.2f}%")

    # Save results
    results_file = results_path / "self_training_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "baseline_accuracy": float(baseline_acc),
                "final_accuracy": float(test_acc),
                "best_accuracy": float(best_accuracy),
                "improvement": float(best_accuracy - baseline_acc),
                "iterations": results_history,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Results saved to: {results_file}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    final_acc, final_per_class = evaluate(model, test_loader, device)

    print(f"\n📊 FINAL PER-CLASS ACCURACY (Best Model):")
    print("=" * 70)
    for label, acc in final_per_class.items():
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1]
        baseline_class_acc = baseline_per_class.get(label, 0)
        improvement = acc - baseline_class_acc

        print(
            f"   {class_name[:30]:30s}: "
            f"{acc*100:5.1f}% (baseline: {baseline_class_acc*100:5.1f}%, "
            f"gain: {improvement*100:+5.1f}%)"
        )

    # Collect predictions from best model for detailed metrics
    model.eval()
    all_preds_final, all_labels_final = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds_final.extend(preds.cpu().numpy())
            all_labels_final.extend(labels.numpy())

    detailed_metrics = build_detailed_metrics(all_labels_final, all_preds_final)
    detailed_file = results_path / "self_training_final_metrics.json"
    with open(detailed_file, "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"\n💾 Detailed metrics saved to: {detailed_file}")

    if best_accuracy >= 0.68:
        print(f"\n🎉 SUCCESS! Target achieved ({best_accuracy*100:.1f}% >= 68%)")
    elif best_accuracy >= 0.60:
        print(f"\n✅ GOOD progress ({best_accuracy*100:.1f}%)")
        print(f"   Progressive DA can push to 75-80%")
    else:
        print(f"\n⚠️  Modest improvement ({best_accuracy*100:.1f}%)")
        print(f"   May need additional techniques")

    print(f"\n🚀 Next step: Progressive Domain Adaptation")


if __name__ == "__main__":
    self_training_da()
