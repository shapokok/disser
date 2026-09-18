"""
Evaluate the best Joint Training checkpoint (79.06%) on PlantDoc 4-class test set.

Checkpoint: results/da_checkpoints/joint_training_final.pth
  (saved by joint_training_da.py whenever a new best accuracy was reached —
   this IS the 79.06% model)

Output: results/joint_training_79percent_metrics.json
"""

import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import json

# Add backend to path so MobileNetModel can be imported
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import MobileNetModel

# ============================================================================
# CONFIG
# ============================================================================

CHECKPOINT = Path(__file__).parent / "results/da_checkpoints/joint_training_final.pth"
PLANTDOC_PATH = "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc"
RESULTS_PATH = Path(__file__).parent / "results"
OUTPUT_FILE = RESULTS_PATH / "joint_training_79percent_metrics.json"

NUM_CLASSES = 38
BATCH_SIZE = 16
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_CLASSES = [7, 9, 25, 30]

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
# DATASET  (identical to joint_training_da.py)
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
        try:
            image = Image.open(self.images[idx]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ============================================================================
# METRICS
# ============================================================================


def build_detailed_metrics(all_labels, all_preds):
    """Detailed per-class metrics for the 4 target classes only"""
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
        class_acc = (
            float((filtered_preds[lmask] == label).sum() / lmask.sum())
            if lmask.sum() > 0 else 0.0
        )
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
# MAIN
# ============================================================================


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║        EVALUATE JOINT TRAINING BEST CHECKPOINT (79.06%)              ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    )

    print(f"🖥️  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Verify checkpoint exists
    if not CHECKPOINT.exists():
        print(f"\n❌ Checkpoint not found: {CHECKPOINT}")
        print("   Run joint_training_da.py first to generate it.")
        return

    print(f"\n📥 Loading checkpoint: {CHECKPOINT}")
    model = MobileNetModel(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()
    print("   ✅ Loaded successfully")

    # Dataset
    print(f"\n📁 Loading PlantDoc (4 classes: {TARGET_CLASSES})...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = PlantDocDataset(
        PLANTDOC_PATH,
        split="train",
        transform=transform,
        class_mapping=ALL_CLASSES,
    )
    print(f"   ✅ {len(dataset)} images loaded")

    from collections import Counter
    label_counts = Counter(dataset.labels)
    for label in sorted(label_counts):
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1]
        print(f"      [{label:2d}] {class_name:<35}: {label_counts[label]:3d} images")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Collect predictions
    print(f"\n🔍 Running inference...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute metrics
    metrics = build_detailed_metrics(all_labels, all_preds)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"   Overall accuracy: {metrics['overall']['accuracy']*100:.2f}%")
    print(f"   Macro-F1:         {metrics['overall']['macro_f1']*100:.2f}%")
    print(f"\n   Per-class (labels=[7,9,25,30]):")
    print(f"   {'Class':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print(f"   {'-'*65}")
    for label in TARGET_CLASSES:
        c = metrics["per_class"][str(label)]
        class_name = PLANTVILLAGE_CLASSES[label].split("___")[-1][:38]
        print(
            f"   [{label:2d}] {class_name:<36}"
            f" {c['accuracy']*100:5.1f}%"
            f" {c['precision']*100:5.1f}%"
            f" {c['recall']*100:5.1f}%"
            f" {c['f1_score']*100:5.1f}%"
            f" {c['samples']:5d}"
        )

    print(f"\n   Confusion matrix (rows=true, cols=pred, order={TARGET_CLASSES}):")
    for row in metrics["confusion_matrix"]:
        print(f"      {row}")

    # Save
    RESULTS_PATH.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
