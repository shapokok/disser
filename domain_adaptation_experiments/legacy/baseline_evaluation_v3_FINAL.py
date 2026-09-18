"""
Domain Adaptation Experiments - Baseline Evaluation v3 FINAL
Использует ТВОЙ model.py для правильной загрузки моделей
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
    # PlantVillage классы
    "plantvillage_classes": [
        "Apple___Apple_scab",  # 0
        "Apple___Black_rot",  # 1
        "Apple___Cedar_apple_rust",  # 2
        "Apple___healthy",  # 3
        "Blueberry___healthy",  # 4
        "Cherry_(including_sour)___Powdery_mildew",  # 5
        "Cherry_(including_sour)___healthy",  # 6
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",  # 7
        "Corn_(maize)___Common_rust_",  # 8
        "Corn_(maize)___Northern_Leaf_Blight",  # 9
        "Corn_(maize)___healthy",  # 10
        "Grape___Black_rot",  # 11
        "Grape___Esca_(Black_Measles)",  # 12
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",  # 13
        "Grape___healthy",  # 14
        "Orange___Haunglongbing_(Citrus_greening)",  # 15
        "Peach___Bacterial_spot",  # 16
        "Peach___healthy",  # 17
        "Pepper,_bell___Bacterial_spot",  # 18
        "Pepper,_bell___healthy",  # 19
        "Potato___Early_blight",  # 20
        "Potato___Late_blight",  # 21
        "Potato___healthy",  # 22
        "Raspberry___healthy",  # 23
        "Soybean___healthy",  # 24
        "Squash___Powdery_mildew",  # 25
        "Strawberry___Leaf_scorch",  # 26
        "Strawberry___healthy",  # 27
        "Tomato___Bacterial_spot",  # 28
        "Tomato___Early_blight",  # 29
        "Tomato___Late_blight",  # 30
        "Tomato___Leaf_Mold",  # 31
        "Tomato___Septoria_leaf_spot",  # 32
        "Tomato___Spider_mites Two-spotted_spider_mite",  # 33
        "Tomato___Target_Spot",  # 34
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",  # 35
        "Tomato___Tomato_mosaic_virus",  # 36
        "Tomato___healthy",  # 37
    ],
}

# ============================================================================
# CLASS MAPPING
# ============================================================================


def create_class_mapping():
    """Маппинг PlantDoc → PlantVillage"""
    mapping = {
        # Apple
        "Apple Scab Leaf": 0,
        "Apple rust leaf": 2,
        "Apple leaf": 3,
        # Blueberry
        "Blueberry leaf": 4,
        # Cherry
        "Cherry leaf": 6,
        # Corn
        "Corn Gray leaf spot": 7,
        "Corn rust leaf": 8,
        "Corn leaf blight": 9,
        # Grape
        "grape leaf black rot": 11,
        "grape leaf": 14,
        # Peach
        "Peach leaf": 17,
        # Bell Pepper
        "Bell_pepper leaf spot": 18,
        "Bell_pepper leaf": 19,
        # Potato
        "Potato leaf early blight": 20,
        "Potato leaf late blight": 21,
        # Raspberry
        "Raspberry leaf": 23,
        # Soybean
        "Soyabean leaf": 24,
        # Squash
        "Squash Powdery mildew leaf": 25,
        # Strawberry
        "Strawberry leaf": 27,
        # Tomato
        "Tomato leaf bacterial spot": 28,
        "Tomato Early blight leaf": 29,
        "Tomato leaf late blight": 30,
        "Tomato mold leaf": 31,
        "Tomato Septoria leaf spot": 32,
        "Tomato leaf yellow virus": 35,
        "Tomato leaf mosaic virus": 36,
        "Tomato leaf": 37,
    }

    return mapping


# ============================================================================
# DATASET
# ============================================================================


class PlantDocDataset(Dataset):
    """PlantDoc Dataset"""

    def __init__(self, root_dir, split="train", transform=None, class_mapping=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.class_mapping = class_mapping or {}

        self.images = []
        self.labels = []
        self.original_classes = []

        self._load_images()

    def _load_images(self):
        print(f"📁 Loading from: {self.root_dir}")

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

        print(f"✅ Loaded {len(self.images)} images")
        print(f"✅ Unique PlantVillage classes: {len(set(self.labels))}")

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
# MODEL LOADING - ИСПОЛЬЗУЕМ ТВОЙ model.py!
# ============================================================================


def load_efficientnet_from_your_code(checkpoint_path, num_classes=38):
    """Загрузить через ТВОЙ model.py"""
    print(f"   Loading from: {checkpoint_path}")

    # Использовать ТВОЮ архитектуру из model.py
    model = EfficientNetModel(num_classes=num_classes, pretrained=False)

    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("   ✅ Loaded successfully!")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print("   Using randomly initialized model")

    return model


def load_mobilenet_from_your_code(checkpoint_path, num_classes=38):
    """Загрузить через ТВОЙ model.py"""
    print(f"   Loading from: {checkpoint_path}")

    model = MobileNetModel(num_classes=num_classes, pretrained=False)

    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("   ✅ Loaded successfully!")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print("   Using randomly initialized model")

    return model


# ============================================================================
# EVALUATION
# ============================================================================


@torch.no_grad()
def evaluate_model(model, dataloader, device, model_name="Model"):
    """Оценить модель"""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    pbar = tqdm(dataloader, desc=f"Testing {model_name}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Метрики
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    results = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "num_samples": len(all_labels),
    }

    print(f"\n📊 RESULTS:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    print(f"   Samples:   {len(all_labels)}")

    return results


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
# MAIN
# ============================================================================


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║     DOMAIN ADAPTATION EXPERIMENTS - BASELINE EVALUATION      ║
║                  PlantVillage → PlantDoc                     ║
║                      v3 FINAL - Using model.py               ║
╚══════════════════════════════════════════════════════════════╝
"""
    )

    device = CONFIG["device"]
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Создать папки
    results_path = Path(CONFIG["results_path"])
    results_path.mkdir(exist_ok=True)

    # Transforms - ВАЖНО: использовать ТУ ЖЕ нормализацию что при тренировке!
    transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # ImageNet norm
        ]
    )

    # Загрузить PlantDoc
    print("\n" + "=" * 60)
    print("LOADING PLANTDOC DATASET")
    print("=" * 60)

    class_mapping = create_class_mapping()

    plantdoc_dataset = PlantDocDataset(
        CONFIG["plantdoc_path"],
        split="train",
        transform=transform,
        class_mapping=class_mapping,
    )

    if len(plantdoc_dataset) == 0:
        print("\n❌ ERROR: No images loaded!")
        return

    plantdoc_loader = DataLoader(
        plantdoc_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # Тестировать модели
    all_results = {}

    models_to_test = [
        ("EfficientNet-B0", "efficientnet", load_efficientnet_from_your_code),
        ("MobileNet-V2", "mobilenet", load_mobilenet_from_your_code),
    ]

    mobilenet_detailed = None

    for model_name, checkpoint_key, load_fn in models_to_test:
        checkpoint_path = (
            Path(CONFIG["models_path"]) / CONFIG[f"{checkpoint_key}_checkpoint"]
        )

        if not checkpoint_path.exists():
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\n🔄 Loading {model_name}...")
        model = load_fn(checkpoint_path, CONFIG["num_classes"])

        # Evaluate (standard)
        results = evaluate_model(model, plantdoc_loader, device, model_name)
        all_results[checkpoint_key] = results

        # Collect detailed metrics for MobileNet baseline
        if checkpoint_key == "mobilenet":
            model.eval()
            model.to(device)
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in plantdoc_loader:
                    outputs = model(images.to(device))
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            mobilenet_detailed = build_detailed_metrics(all_labels, all_preds)

    # Сохранить результаты
    results_file = results_path / "baseline_evaluation_FINAL.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Сохранить detailed metrics для MobileNet baseline
    if mobilenet_detailed is not None:
        detailed_file = results_path / "baseline_metrics_4class.json"
        with open(detailed_file, "w") as f:
            json.dump(mobilenet_detailed, f, indent=2)
        print(f"💾 Detailed baseline metrics saved to: {detailed_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: DOMAIN SHIFT (PlantVillage → PlantDoc)")
    print("=" * 60)
    print(f"{'Model':<20} {'PV (train)':<12} {'PlantDoc':<12} {'Gap':<12}")
    print("-" * 60)

    baseline_acc = {"efficientnet": 0.9893, "mobilenet": 0.9887}

    for key, res in all_results.items():
        pv_acc = baseline_acc.get(key, 0) * 100
        pd_acc = res["accuracy"] * 100
        gap = pv_acc - pd_acc

        model_name = res["model"]
        print(
            f"{model_name:<20} {pv_acc:>6.2f}%      {pd_acc:>6.2f}%      {gap:>+6.2f}%"
        )

    print("\n✅ Baseline evaluation complete!")

    if all_results:
        avg_acc = (
            sum(r["accuracy"] for r in all_results.values()) / len(all_results) * 100
        )
        if avg_acc > 80:
            print("\n🎉 EXCELLENT! Models work well on field data!")
            print("   Domain shift is minimal.")
        elif avg_acc > 60:
            print("\n✅ GOOD! Moderate domain shift.")
            print("   Domain Adaptation can improve by +10-15%")
        elif avg_acc > 40:
            print("\n⚠️  SIGNIFICANT domain shift detected!")
            print("   Domain Adaptation is essential!")
            print("   Expected improvement: +15-25%")
        else:
            print("\n🚨 SEVERE domain shift!")
            print("   Strong Domain Adaptation needed!")
            print("   Expected improvement: +20-30%")


if __name__ == "__main__":
    main()
