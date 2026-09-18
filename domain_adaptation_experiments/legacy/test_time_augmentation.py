"""
Test-Time Augmentation (TTA)
Apply multiple augmentations during inference and average predictions

Goal: 79.1% → 80-81%
Time: ~5 minutes inference
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

from model import MobileNetModel

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "plantdoc_path": "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc",
    "checkpoint": "./results/da_checkpoints/joint_training_final.pth",
    "results_path": "./results",
    # TTA parameters
    "num_tta_transforms": 10,  # Number of augmented versions per image
    "num_classes": 38,
    "img_size": 224,
    "batch_size": 16,
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
# TTA TRANSFORMS
# ============================================================================


def get_tta_transforms():
    """
    Generate multiple TTA transforms
    Each applies different augmentations to same image
    """
    base_transform = [
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    ]

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    tta_transforms = [
        # Original
        transforms.Compose(base_transform + [transforms.ToTensor(), normalize]),
        # Horizontal flip
        transforms.Compose(
            base_transform
            + [transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), normalize]
        ),
        # Vertical flip
        transforms.Compose(
            base_transform
            + [transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), normalize]
        ),
        # Both flips
        transforms.Compose(
            base_transform
            + [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # Rotation +10
        transforms.Compose(
            base_transform
            + [
                transforms.RandomRotation(degrees=(10, 10)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # Rotation -10
        transforms.Compose(
            base_transform
            + [
                transforms.RandomRotation(degrees=(-10, -10)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # Brightness adjustment
        transforms.Compose(
            base_transform
            + [transforms.ColorJitter(brightness=0.2), transforms.ToTensor(), normalize]
        ),
        # Contrast adjustment
        transforms.Compose(
            base_transform
            + [transforms.ColorJitter(contrast=0.2), transforms.ToTensor(), normalize]
        ),
        # Scale 0.9
        transforms.Compose(
            base_transform
            + [
                transforms.RandomAffine(degrees=0, scale=(0.9, 0.9)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        # Scale 1.1
        transforms.Compose(
            base_transform
            + [
                transforms.RandomAffine(degrees=0, scale=(1.1, 1.1)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ]

    return tta_transforms


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

    def get_raw_image(self, idx):
        """Get PIL image without transform"""
        img_path = self.images[idx]
        try:
            return Image.open(img_path).convert("RGB")
        except:
            return Image.new("RGB", (224, 224))


# ============================================================================
# TTA EVALUATION
# ============================================================================


@torch.no_grad()
def evaluate_with_tta(model, dataset, tta_transforms, device):
    """
    Evaluate with Test-Time Augmentation

    For each image:
    1. Apply multiple augmentations
    2. Get predictions for each
    3. Average probabilities
    4. Take argmax
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    print(f"   Evaluating with {len(tta_transforms)} TTA transforms...")

    for idx in tqdm(range(len(dataset)), desc="TTA Inference"):
        # Get raw image
        raw_image = dataset.get_raw_image(idx)
        label = dataset.labels[idx]

        # Apply all TTA transforms and collect predictions
        tta_probs = []

        for transform in tta_transforms:
            image = transform(raw_image).unsqueeze(0).to(device)
            output = model(image)
            probs = torch.softmax(output, dim=1)
            tta_probs.append(probs.cpu().numpy())

        # Average probabilities across all augmentations
        avg_probs = np.mean(tta_probs, axis=0)
        pred = np.argmax(avg_probs)

        all_preds.append(pred)
        all_labels.append(label)

    # Calculate accuracy
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


@torch.no_grad()
def evaluate_standard(model, dataloader, device):
    """Standard evaluation without TTA"""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)

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
║              TEST-TIME AUGMENTATION EVALUATION                       ║
║           Apply multiple transforms and average predictions          ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    )

    device = CONFIG["device"]
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n🎯 STRATEGY:")
    print(f"   Apply {CONFIG['num_tta_transforms']} different augmentations per image")
    print(f"   Average predictions across all versions")
    print(f"   Expected: +1-2% improvement")

    # Load model
    print(f"\n📥 Loading Joint Training model...")
    model = MobileNetModel(num_classes=CONFIG["num_classes"], pretrained=False)
    checkpoint = Path(CONFIG["checkpoint"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)
    print(f"   ✅ Loaded from {checkpoint}")

    # Load dataset
    print(f"\n📁 Loading dataset...")

    # For standard evaluation
    standard_transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    standard_dataset = PlantDocDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=standard_transform,
        class_mapping=ALL_CLASSES,
    )

    standard_loader = DataLoader(
        standard_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # For TTA evaluation (no transform, will apply multiple)
    tta_dataset = PlantDocDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=None,  # Will apply transforms manually
        class_mapping=ALL_CLASSES,
    )

    print(f"   ✅ Loaded {len(tta_dataset)} images")

    # Get TTA transforms
    tta_transforms = get_tta_transforms()
    print(f"\n   📋 TTA transforms: {len(tta_transforms)}")

    # Evaluate standard (baseline)
    print(f"\n📊 Standard Evaluation (no TTA)...")
    standard_acc, standard_per_class = evaluate_standard(model, standard_loader, device)
    print(f"   Accuracy: {standard_acc*100:.2f}%")

    # Evaluate with TTA
    print(f"\n📊 TTA Evaluation...")
    tta_acc, tta_per_class = evaluate_with_tta(
        model, tta_dataset, tta_transforms, device
    )

    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    print(f"\n📊 OVERALL ACCURACY:")
    print(f"   Standard (no TTA):  {standard_acc*100:5.2f}%")
    print(f"   With TTA:           {tta_acc*100:5.2f}%")
    print(f"   Improvement:        {(tta_acc - standard_acc)*100:+5.2f}%")

    print(f"\n📊 PER-CLASS COMPARISON:")
    print("=" * 70)
    print(f"{'Class':<30} {'Standard':>10} {'TTA':>10} {'Gain':>10}")
    print("-" * 70)

    for label in sorted(tta_per_class.keys()):
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:28]
        std_acc = standard_per_class.get(label, 0) * 100
        tta_class_acc = tta_per_class.get(label, 0) * 100
        gain = tta_class_acc - std_acc

        print(f"{class_name:<30} {std_acc:>9.1f}% {tta_class_acc:>9.1f}% {gain:>9.1f}%")

    # Save results
    results = {
        "standard": {
            "accuracy": float(standard_acc),
            "per_class": {int(k): float(v) for k, v in standard_per_class.items()},
        },
        "tta": {
            "accuracy": float(tta_acc),
            "per_class": {int(k): float(v) for k, v in tta_per_class.items()},
            "num_transforms": len(tta_transforms),
        },
        "improvement": float(tta_acc - standard_acc),
    }

    results_file = Path(CONFIG["results_path"]) / "tta_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    improvement = (tta_acc - standard_acc) * 100

    if improvement >= 2.0:
        print(f"\n🎉 EXCELLENT! TTA boosted accuracy by {improvement:+.2f}%")
    elif improvement >= 1.0:
        print(f"\n✅ GOOD! TTA improved by {improvement:+.2f}%")
    elif improvement >= 0.5:
        print(f"\n✅ Modest gain: {improvement:+.2f}%")
    else:
        print(f"\n⚠️  Small gain: {improvement:+.2f}%")

    print(f"\n📊 Final accuracy with TTA: {tta_acc*100:.2f}%")
    print(f"\n🚀 Next step: Fine-tune with micro LR for final push!")


if __name__ == "__main__":
    main()
