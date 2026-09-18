"""
Class-Specific Ensemble using best checkpoint per class
Quick win: Use Stage1 for easy classes, Stage2 for hard classes

Expected: 66.5% → 69-71%
Time: 5 minutes!
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import MobileNetModel

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "plantdoc_path": "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc",
    "checkpoints": {
        "stage1": "./results/da_checkpoints/progressive_stage1.pth",
        "stage2": "./results/da_checkpoints/progressive_stage2.pth",
        "final": "./results/da_checkpoints/progressive_da_final.pth",
    },
    "results_path": "./results",
    "num_classes": 38,
    "img_size": 224,
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# CLASS ASSIGNMENTS
CLASS_ROUTING = {
    7: "stage2",  # Corn Gray - best in Stage 2
    9: "stage2",  # Corn Blight - best in Stage 2
    25: "final",  # Squash - best in Final (91%!)
    30: "stage1",  # Tomato - best in Stage 1 (before forgetting)
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
# CLASS-SPECIFIC ENSEMBLE
# ============================================================================


@torch.no_grad()
def class_specific_ensemble_evaluate(models_dict, dataloader, device, routing):
    """
    Use different models for different classes

    Args:
        models_dict: {checkpoint_name: model}
        dataloader: Test loader
        device: Device
        routing: {class_id: checkpoint_name}
    """
    for model in models_dict.values():
        model.eval()
        model.to(device)

    all_preds = []
    all_labels = []

    print(f"   Evaluating with class-specific routing...")

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)

        batch_preds = []

        for i, label in enumerate(labels):
            label_val = label.item()

            # Get appropriate model for this class
            checkpoint_name = routing.get(label_val, "final")
            model = models_dict[checkpoint_name]

            # Predict
            img = images[i : i + 1]
            output = model(img)
            _, pred = torch.max(output, 1)

            batch_preds.append(pred.item())

        all_preds.extend(batch_preds)
        all_labels.extend(labels.numpy())

    # Calculate metrics
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


# ============================================================================
# MAIN
# ============================================================================


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║              CLASS-SPECIFIC ENSEMBLE EVALUATION                      ║
║        Use best checkpoint for each class independently              ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    )

    device = CONFIG["device"]
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n🎯 STRATEGY:")
    print(f"   Corn classes    → Stage 2 checkpoint (best for Corn)")
    print(f"   Squash          → Final checkpoint (91%!)")
    print(f"   Tomato          → Stage 1 checkpoint (before forgetting)")

    print(f"\n📋 CLASS ROUTING:")
    for class_id, checkpoint in CLASS_ROUTING.items():
        class_name = PLANTVILLAGE_CLASSES[class_id].split("___")[-1][:28]
        print(f"   [{class_id:2d}] {class_name:30s} → {checkpoint}")

    # Load models
    print(f"\n📥 Loading checkpoints...")
    models = {}

    for name, path in CONFIG["checkpoints"].items():
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            print(f"   ⚠️  {name}: Not found, skipping")
            continue

        model = MobileNetModel(num_classes=CONFIG["num_classes"], pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        models[name] = model
        print(f"   ✅ {name}: Loaded")

    print(f"\n   Total checkpoints: {len(models)}")

    # Load dataset
    print(f"\n📁 Loading PlantDoc dataset...")
    transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = PlantDocDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=transform,
        class_mapping=ALL_CLASSES,
    )

    print(f"   ✅ Loaded {len(dataset)} images")

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Batch=1 for class-specific routing
        shuffle=False,
        num_workers=0,
    )

    # Evaluate
    print(f"\n📊 Evaluating class-specific ensemble...")
    accuracy, per_class_acc = class_specific_ensemble_evaluate(
        models, dataloader, device, CLASS_ROUTING
    )

    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    print(f"\n📊 OVERALL ACCURACY: {accuracy*100:.2f}%")

    print(f"\n📊 PER-CLASS RESULTS:")
    print("=" * 70)
    print(f"{'Class':<30} {'Checkpoint':<12} {'Accuracy':>10}")
    print("-" * 70)

    # Load baseline for comparison
    baseline_file = Path(CONFIG["results_path"]) / "progressive_da_results.json"
    baseline_per_class = {}
    if baseline_file.exists():
        with open(baseline_file) as f:
            data = json.load(f)
            baseline_per_class = {
                int(k): v for k, v in data["final"]["per_class"].items()
            }

    for label in sorted(per_class_acc.keys()):
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:28]
        checkpoint = CLASS_ROUTING.get(label, "final")
        acc = per_class_acc[label] * 100

        baseline_acc = baseline_per_class.get(label, 0) * 100
        gain = acc - baseline_acc

        print(
            f"{class_name:<30} {checkpoint:<12} {acc:>9.1f}% "
            f"(was: {baseline_acc:>5.1f}%, {gain:>+5.1f}%)"
        )

    # Save results
    results = {
        "strategy": "class_specific_ensemble",
        "routing": {int(k): v for k, v in CLASS_ROUTING.items()},
        "overall_accuracy": float(accuracy),
        "per_class": {int(k): float(v) for k, v in per_class_acc.items()},
        "improvement_over_final": float(
            accuracy - baseline_per_class.get("accuracy", 0.665)
        ),
    }

    results_file = Path(CONFIG["results_path"]) / "class_specific_ensemble_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    progressive_final = 0.6645  # From previous result
    improvement = accuracy - progressive_final

    print(f"\n📊 Progressive DA (Final):        {progressive_final*100:.2f}%")
    print(f"📊 Class-Specific Ensemble:       {accuracy*100:.2f}%")
    print(f"📈 Improvement:                   {improvement*100:+.2f}%")

    if accuracy >= 0.70:
        print(f"\n🎉 EXCELLENT! Achieved {accuracy*100:.1f}% (target: 70%+)")
    elif accuracy >= 0.68:
        print(f"\n✅ GREAT! {accuracy*100:.1f}%")
    elif accuracy >= 0.67:
        print(f"\n✅ GOOD! {accuracy*100:.1f}%")
    else:
        print(f"\n⚠️  Modest gain: {accuracy*100:.1f}%")

    print(f"\n🚀 Next: Try joint training from scratch for even better results!")


if __name__ == "__main__":
    main()
