"""
Ensemble Domain Adaptation
Combine EfficientNet + MobileNet for improved accuracy

Goal: 63% → 66-68% (quick win!)
Time: ~30 minutes to run
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
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

from model import MobileNetModel, EfficientNetModel

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ПУТИ
    "plantdoc_path": "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc",
    "models_path": "C:/Users/Нурислам/Desktop/disser/models",
    "st_checkpoint": "./results/da_checkpoints/self_training_best.pth",
    "results_path": "./results",
    # МОДЕЛИ
    "efficientnet_checkpoint": "efficientnet_model.pth",
    "mobilenet_base": "mobilenet_model.pth",
    # Parameters
    "num_classes": 38,
    "img_size": 224,
    "batch_size": 16,
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
# DATASET
# ============================================================================


class PlantDocDataset(Dataset):
    """Simple PlantDoc Dataset"""

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
        except Exception as e:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# ENSEMBLE EVALUATION
# ============================================================================


@torch.no_grad()
def ensemble_evaluate(models, dataloader, device, weights=None):
    """
    Evaluate ensemble of models

    Args:
        models: List of models
        dataloader: Test dataloader
        device: Device
        weights: Optional weights for each model (default: equal weights)

    Returns:
        accuracy, per_class_accuracy
    """
    for model in models:
        model.eval()
        model.to(device)

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    all_preds = []
    all_labels = []

    print(f"   Evaluating ensemble of {len(models)} models...")

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)

        # Get predictions from all models
        ensemble_probs = None

        for model, weight in zip(models, weights):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if ensemble_probs is None:
                ensemble_probs = probs * weight
            else:
                ensemble_probs += probs * weight

        # Get final predictions
        _, preds = torch.max(ensemble_probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    # Overall accuracy
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


# ============================================================================
# MAIN
# ============================================================================


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║                    ENSEMBLE DOMAIN ADAPTATION                        ║
║              Quick improvement: 63% → 66-68%                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    )

    device = CONFIG["device"]
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n🎯 STRATEGY:")
    print(f"   Combine EfficientNet + MobileNet predictions")
    print(f"   Current best (ST-DA MobileNet): 63.0%")
    print(f"   Target: 66-68%")
    print(f"   Expected gain: +3-5%")

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    print(f"\n📁 Loading PlantDoc dataset...")
    dataset = PlantDocDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=transform,
        class_mapping=OPTIMAL_4_CLASSES,
    )

    print(f"   ✅ Loaded {len(dataset)} images")

    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # Load models
    print(f"\n📥 Loading models...")
    models = []
    model_names = []

    # Model 1: EfficientNet (base)
    print(f"   1. EfficientNet-B0 (base)...")
    efficientnet_path = Path(CONFIG["models_path"]) / CONFIG["efficientnet_checkpoint"]
    efficientnet = EfficientNetModel(
        num_classes=CONFIG["num_classes"], pretrained=False
    )
    efficientnet.load_state_dict(torch.load(efficientnet_path, map_location="cpu"))
    models.append(efficientnet)
    model_names.append("EfficientNet")
    print(f"      ✅ Loaded")

    # Model 2: MobileNet (ST-DA)
    print(f"   2. MobileNet-V2 (Self-Training DA)...")
    st_checkpoint = Path(CONFIG["st_checkpoint"])
    mobilenet_st = MobileNetModel(num_classes=CONFIG["num_classes"], pretrained=False)
    mobilenet_st.load_state_dict(torch.load(st_checkpoint, map_location="cpu"))
    models.append(mobilenet_st)
    model_names.append("MobileNet-ST-DA")
    print(f"      ✅ Loaded")

    print(f"\n   Total models in ensemble: {len(models)}")

    # Evaluate individual models
    print(f"\n{'='*70}")
    print(f"INDIVIDUAL MODEL PERFORMANCE")
    print(f"{'='*70}")

    individual_results = {}

    for model, name in zip(models, model_names):
        print(f"\n📊 {name}:")
        acc, per_class = ensemble_evaluate([model], dataloader, device)

        individual_results[name] = {
            "accuracy": float(acc),
            "per_class": {int(k): float(v) for k, v in per_class.items()},
        }

        print(f"   Overall: {acc*100:.2f}%")
        for label, class_acc in per_class.items():
            class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:28]
            print(f"      {class_name:30s}: {class_acc*100:5.1f}%")

    # Test ensemble strategies
    print(f"\n{'='*70}")
    print(f"ENSEMBLE STRATEGIES")
    print(f"{'='*70}")

    # Strategy 1: Equal weights (0.5, 0.5)
    print(f"\n1️⃣  EQUAL WEIGHTS (0.5, 0.5)")
    acc_equal, per_class_equal = ensemble_evaluate(
        models, dataloader, device, [0.5, 0.5]
    )
    print(f"   Overall: {acc_equal*100:.2f}%")

    # Strategy 2: Favor ST-DA (0.3, 0.7)
    print(f"\n2️⃣  ST-DA FOCUSED (0.3, 0.7)")
    acc_focused, per_class_focused = ensemble_evaluate(
        models, dataloader, device, [0.3, 0.7]
    )
    print(f"   Overall: {acc_focused*100:.2f}%")

    # Strategy 3: Heavy ST-DA (0.2, 0.8)
    print(f"\n3️⃣  HEAVY ST-DA (0.2, 0.8)")
    acc_heavy, per_class_heavy = ensemble_evaluate(
        models, dataloader, device, [0.2, 0.8]
    )
    print(f"   Overall: {acc_heavy*100:.2f}%")

    # Find best
    strategies = [
        ("Equal", acc_equal, per_class_equal, [0.5, 0.5]),
        ("ST-Focused", acc_focused, per_class_focused, [0.3, 0.7]),
        ("Heavy-ST", acc_heavy, per_class_heavy, [0.2, 0.8]),
    ]

    best_strategy = max(strategies, key=lambda x: x[1])
    best_name, best_acc, best_per_class, best_weights = best_strategy

    print(f"\n{'='*70}")
    print(f"BEST ENSEMBLE")
    print(f"{'='*70}")
    print(f"\n🏆 Strategy: {best_name}")
    print(
        f"   Weights: EfficientNet={best_weights[0]:.1f}, MobileNet={best_weights[1]:.1f}"
    )
    print(f"   Overall accuracy: {best_acc*100:.2f}%")

    print(f"\n📊 Per-class accuracy:")
    st_baseline = individual_results["MobileNet-ST-DA"]["per_class"]

    for label, class_acc in best_per_class.items():
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:28]
        baseline = st_baseline.get(label, 0.0)
        gain = class_acc - baseline

        print(
            f"   {class_name:30s}: {class_acc*100:5.1f}% "
            f"(ST-DA: {baseline*100:5.1f}%, {gain*100:+5.1f}%)"
        )

    # Save results
    results = {
        "individual": individual_results,
        "ensemble": {
            "strategy": best_name,
            "weights": best_weights,
            "accuracy": float(best_acc),
            "per_class": {int(k): float(v) for k, v in best_per_class.items()},
        },
        "improvement": float(
            best_acc - individual_results["MobileNet-ST-DA"]["accuracy"]
        ),
    }

    results_file = Path(CONFIG["results_path"]) / "ensemble_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    baseline = individual_results["MobileNet-ST-DA"]["accuracy"]
    improvement = best_acc - baseline

    print(f"\n📊 ST-DA (MobileNet):  {baseline*100:.2f}%")
    print(f"📊 Ensemble (best):    {best_acc*100:.2f}%")
    print(f"📈 Improvement:        {improvement*100:+.2f}%")

    if best_acc >= 0.66:
        print(f"\n🎉 SUCCESS! Target achieved ({best_acc*100:.1f}% >= 66%)")
    elif best_acc >= 0.64:
        print(f"\n✅ GOOD progress! ({best_acc*100:.1f}%)")
    else:
        print(f"\n⚠️  Modest gain ({best_acc*100:.1f}%)")

    print(f"\n🚀 Next step: Progressive DA (target: 75-80%)")


if __name__ == "__main__":
    main()
