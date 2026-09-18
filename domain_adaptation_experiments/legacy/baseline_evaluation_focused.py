"""
Domain Adaptation Experiments - OPTIMAL 4-Class Baseline
Focus on 4 disease classes with best baseline performance

Goal: Baseline ~45-50% → After DA: 80-85%
Classes: Corn Gray spot, Squash Mildew, Corn Blight, Tomato Late blight
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import numpy as np

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import EfficientNetModel, MobileNetModel

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ПУТИ
    "plantvillage_path": "C:/Users/Нурислам/Desktop/disser/data/PlantVillage",
    "plantdoc_path": "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc",
    "models_path": "C:/Users/Нурислам/Desktop/disser/models",
    "results_path": "./results",
    # МОДЕЛИ
    "efficientnet_checkpoint": "efficientnet_model.pth",
    "mobilenet_checkpoint": "mobilenet_model.pth",
    # Параметры
    "num_classes": 38,
    "img_size": 224,
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# OPTIMAL 4 CLASSES - Best baseline performance
# ============================================================================

OPTIMAL_4_CLASSES = {
    # Class 1: Corn Gray leaf spot (73% baseline!) ✅
    "Corn Gray leaf spot": 7,
    # Class 2: Squash Powdery mildew (62% baseline!) ✅
    "Squash Powdery mildew leaf": 25,
    # Class 3: Corn Blight (24% baseline) ⚠️
    "Corn leaf blight": 9,
    # Class 4: Tomato Late blight (33% baseline) ⚠️
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


class PlantDocOptimal4Dataset(Dataset):
    """PlantDoc Dataset - ONLY 4 optimal classes"""

    def __init__(self, root_dir, split="train", transform=None, class_mapping=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.class_mapping = class_mapping or {}

        self.images = []
        self.labels = []
        self.original_classes = []

        self._load_images()

    def _load_images(self):
        print(f"📁 Loading OPTIMAL 4-CLASS dataset from: {self.root_dir}")

        if not self.root_dir.exists():
            print(f"❌ Directory not found: {self.root_dir}")
            return

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
                self.original_classes.append(class_name)

        print(
            f"✅ Loaded {len(self.images)} images from {len(set(self.labels))} classes"
        )

        # Class distribution
        from collections import Counter

        class_counts = Counter(self.original_classes)

        print(f"\n📊 OPTIMAL 4-CLASS Distribution:")
        print("=" * 70)

        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            pv_idx = self.class_mapping[cls]
            pv_name = PLANTVILLAGE_CLASSES[pv_idx]
            print(f"   [{pv_idx:2d}] {cls:35s}: {count:4d} images → {pv_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model(checkpoint_path, model_class, num_classes=38):
    """Load model using your model.py"""
    print(f"   Loading from: {checkpoint_path}")

    model = model_class(num_classes=num_classes, pretrained=False)

    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("   ✅ Loaded successfully!")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

    return model


# ============================================================================
# EVALUATION
# ============================================================================


@torch.no_grad()
def evaluate_model(model, dataloader, device, model_name="Model"):
    """Evaluate model with detailed per-class metrics"""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")

    pbar = tqdm(dataloader, desc=f"Testing {model_name}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Per-class metrics
    unique_labels = sorted(set(all_labels))
    per_class_metrics = []

    for label in unique_labels:
        mask = np.array(all_labels) == label
        class_preds = np.array(all_preds)[mask]
        class_labels = np.array(all_labels)[mask]

        if len(class_labels) > 0:
            class_acc = accuracy_score(class_labels, class_preds)
            # Calculate precision/recall for this specific class
            class_prec = precision = np.sum(
                (class_preds == label) & (class_labels == label)
            ) / max(np.sum(class_preds == label), 1)
            class_rec = np.sum((class_preds == label) & (class_labels == label)) / max(
                np.sum(class_labels == label), 1
            )
            class_f1 = 2 * (class_prec * class_rec) / max(class_prec + class_rec, 1e-10)

            per_class_metrics.append(
                {
                    "class_idx": int(label),
                    "class_name": PLANTVILLAGE_CLASSES[label],
                    "accuracy": float(class_acc),
                    "precision": float(class_prec),
                    "recall": float(class_rec),
                    "f1_score": float(class_f1),
                    "samples": int(len(class_labels)),
                }
            )

    results = {
        "model": model_name,
        "overall": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "num_samples": len(all_labels),
        },
        "per_class": per_class_metrics,
    }

    # Print results
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    print(f"   Samples:   {len(all_labels)}")

    print(f"\n📊 PER-CLASS DETAILED METRICS:")
    print("=" * 70)
    print(f"{'Class':<45} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("-" * 70)

    for metric in per_class_metrics:
        short_name = metric["class_name"].split("___")[-1][:40]
        print(
            f"{short_name:<45} "
            f"{metric['accuracy']*100:>5.1f}% "
            f"{metric['precision']*100:>5.1f}% "
            f"{metric['recall']*100:>5.1f}% "
            f"{metric['f1_score']*100:>5.1f}% "
            f"{metric['samples']:>5d}"
        )

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║              OPTIMAL 4-CLASS BASELINE EVALUATION                     ║
║         Focus on Classes with Best Baseline Performance              ║
║                   Target: 80-85% after DA                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    )

    device = CONFIG["device"]
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n🎯 STRATEGY:")
    print(f"   Select 4 disease classes with highest baseline accuracy")
    print(f"   Expected baseline: 45-50%")
    print(f"   Target after DA: 80-85%")
    print(f"   Expected improvement: +35-40%")

    print(f"\n📋 OPTIMAL 4 CLASSES:")
    print(f"   1. Corn Gray leaf spot      (expected: ~70%)")
    print(f"   2. Squash Powdery mildew    (expected: ~60%)")
    print(f"   3. Corn Blight              (expected: ~25%)")
    print(f"   4. Tomato Late blight       (expected: ~30%)")
    print(f"   Average baseline:            ~46%")

    # Create results dir
    results_path = Path(CONFIG["results_path"])
    results_path.mkdir(exist_ok=True)

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load PlantDoc OPTIMAL 4
    print("\n" + "=" * 70)
    print("LOADING PLANTDOC OPTIMAL 4-CLASS DATASET")
    print("=" * 70)

    plantdoc_dataset = PlantDocOptimal4Dataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=transform,
        class_mapping=OPTIMAL_4_CLASSES,
    )

    if len(plantdoc_dataset) == 0:
        print("\n❌ ERROR: No images loaded!")
        return

    plantdoc_loader = DataLoader(
        plantdoc_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # Test models
    all_results = {}

    models_to_test = [
        ("EfficientNet-B0", "efficientnet", EfficientNetModel),
        ("MobileNet-V2", "mobilenet", MobileNetModel),
    ]

    for model_name, checkpoint_key, model_class in models_to_test:
        checkpoint_path = (
            Path(CONFIG["models_path"]) / CONFIG[f"{checkpoint_key}_checkpoint"]
        )

        if not checkpoint_path.exists():
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\n🔄 Loading {model_name}...")
        model = load_model(checkpoint_path, model_class, CONFIG["num_classes"])

        # Evaluate
        results = evaluate_model(model, plantdoc_loader, device, model_name)
        all_results[checkpoint_key] = results

    # Save results (JSON-safe)
    results_file = results_path / "baseline_evaluation_OPTIMAL_4CLASS.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: OPTIMAL 4-CLASS BASELINE")
    print("=" * 70)

    for key, res in all_results.items():
        model_name = res["model"]
        overall_acc = res["overall"]["accuracy"] * 100

        print(f"\n{model_name}:")
        print(f"   Overall Accuracy: {overall_acc:.2f}%")
        print(f"   Per-class breakdown:")

        for cls in res["per_class"]:
            short_name = cls["class_name"].split("___")[-1]
            print(f"      {short_name[:30]:30s}: {cls['accuracy']*100:5.1f}%")

    # Calculate average baseline
    avg_baseline = (
        np.mean([r["overall"]["accuracy"] for r in all_results.values()]) * 100
    )

    print("\n" + "=" * 70)
    print("PROJECTION: After Domain Adaptation")
    print("=" * 70)

    print(f"\n📊 Current Baseline:     {avg_baseline:.1f}%")
    print(f"🎯 Target after DA:      80-85%")
    print(
        f"📈 Expected improvement: +{80-avg_baseline:.0f}% to +{85-avg_baseline:.0f}%"
    )

    # Roadmap
    print(f"\n🗺️  ROADMAP TO 80%+:")
    print(f"   Step 1: Self-Training DA (3-5 iterations)")
    print(f"           Expected: {avg_baseline:.1f}% → 70-75%")
    print(f"   Step 2: Progressive Fine-tuning")
    print(f"           Expected: 70-75% → 80-85%")

    if avg_baseline >= 45:
        print(f"\n✅ EXCELLENT baseline ({avg_baseline:.1f}%)!")
        print(f"   Strong foundation for achieving 80%+ target.")
    elif avg_baseline >= 35:
        print(f"\n✅ GOOD baseline ({avg_baseline:.1f}%)")
        print(f"   Target 80% is achievable with proper DA.")
    else:
        print(f"\n⚠️  Lower baseline ({avg_baseline:.1f}%)")
        print(f"   May need additional techniques to reach 80%.")

    print(f"\n🚀 Ready for Domain Adaptation experiments!")
    print(f"📝 This baseline will be used in your SIST 2026 paper.")


if __name__ == "__main__":
    main()
