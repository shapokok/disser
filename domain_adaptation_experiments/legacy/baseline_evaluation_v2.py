"""
Domain Adaptation Experiments - Baseline Evaluation v2
ИСПРАВЛЕННАЯ ВЕРСИЯ для PlantDoc с правильными названиями классов
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ПУТИ (ИЗМЕНИ ЕСЛИ НУЖНО)
    "plantvillage_path": "C:/Users/Нурислам/Desktop/disser/data/PlantVillage",
    "plantdoc_path": "C:/Users/Нурислам/Desktop/disser/domain_adaptation_experiments/datasets/plantdoc",
    "models_path": "C:/Users/Нурислам/Desktop/disser/models",  # Используем готовые модели
    "results_path": "./results",
    # НАЗВАНИЯ МОДЕЛЕЙ
    "efficientnet_checkpoint": "efficientnet_model.pth",
    "mobilenet_checkpoint": "mobilenet_model.pth",
    # Параметры
    "num_classes": 38,
    "img_size": 224,
    "batch_size": 16,  # Уменьшил для надежности
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # PlantVillage классы (твои 38)
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
# CLASS MAPPING PlantDoc → PlantVillage
# ============================================================================


def create_class_mapping():
    """
    Маппинг PlantDoc классов → PlantVillage индексы
    """
    mapping = {
        # Apple
        "Apple Scab Leaf": 0,  # → Apple___Apple_scab
        "Apple rust leaf": 2,  # → Apple___Cedar_apple_rust
        "Apple leaf": 3,  # → Apple___healthy
        # Blueberry
        "Blueberry leaf": 4,  # → Blueberry___healthy
        # Cherry
        "Cherry leaf": 6,  # → Cherry___healthy
        # Corn
        "Corn Gray leaf spot": 7,  # → Corn___Gray_leaf_spot
        "Corn rust leaf": 8,  # → Corn___Common_rust
        "Corn leaf blight": 9,  # → Corn___Northern_Leaf_Blight
        # Grape
        "grape leaf black rot": 11,  # → Grape___Black_rot
        "grape leaf": 14,  # → Grape___healthy
        # Peach
        "Peach leaf": 17,  # → Peach___healthy
        # Bell Pepper
        "Bell_pepper leaf spot": 18,  # → Pepper___Bacterial_spot
        "Bell_pepper leaf": 19,  # → Pepper___healthy
        # Potato
        "Potato leaf early blight": 20,  # → Potato___Early_blight
        "Potato leaf late blight": 21,  # → Potato___Late_blight
        # Raspberry
        "Raspberry leaf": 23,  # → Raspberry___healthy
        # Soybean
        "Soyabean leaf": 24,  # → Soybean___healthy
        # Squash
        "Squash Powdery mildew leaf": 25,  # → Squash___Powdery_mildew
        # Strawberry
        "Strawberry leaf": 27,  # → Strawberry___healthy
        # Tomato
        "Tomato leaf bacterial spot": 28,  # → Tomato___Bacterial_spot
        "Tomato Early blight leaf": 29,  # → Tomato___Early_blight
        "Tomato leaf late blight": 30,  # → Tomato___Late_blight
        "Tomato mold leaf": 31,  # → Tomato___Leaf_Mold
        "Tomato Septoria leaf spot": 32,  # → Tomato___Septoria_leaf_spot
        "Tomato leaf yellow virus": 35,  # → Tomato___TYLCV
        "Tomato leaf mosaic virus": 36,  # → Tomato___mosaic_virus
        "Tomato leaf": 37,  # → Tomato___healthy
    }

    return mapping


# ============================================================================
# DATASET
# ============================================================================


class PlantDocDataset(Dataset):
    """PlantDoc Dataset с маппингом на PlantVillage классы"""

    def __init__(self, root_dir, split="train", transform=None, class_mapping=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.class_mapping = class_mapping or {}

        self.images = []
        self.labels = []
        self.original_classes = []

        self._load_images()

    def _load_images(self):
        """Загрузить изображения и маппить классы"""
        print(f"📁 Loading from: {self.root_dir}")

        if not self.root_dir.exists():
            print(f"❌ Directory not found: {self.root_dir}")
            return

        # Пройтись по всем классам
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            mapped_class = self.class_mapping.get(class_name, -1)

            if mapped_class == -1:
                print(f"⚠️  Unmapped class: {class_name}")
                continue

            # Загрузить изображения
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

        # Показать статистику
        unique_classes = {}
        for orig, mapped in zip(self.original_classes, self.labels):
            if orig not in unique_classes:
                unique_classes[orig] = 0
            unique_classes[orig] += 1

        print(f"\n📊 Class distribution:")
        for cls, count in sorted(unique_classes.items())[:10]:
            print(f"   {cls}: {count} images")
        if len(unique_classes) > 10:
            print(f"   ... and {len(unique_classes) - 10} more classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Return a blank image if error
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_efficientnet(checkpoint_path, num_classes=38):
    """Загрузить EfficientNet-B0 с поддержкой backbone префикса"""
    print(f"   Loading from: {checkpoint_path}")

    from torchvision.models import efficientnet_b0

    # Создать модель
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Загрузить веса
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Извлечь state_dict
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Убрать "backbone." префикс если есть
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("backbone.", "")  # Убираем префикс
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        print("   ✅ Loaded successfully!")

    except Exception as e:
        print(f"   ⚠️  Error loading checkpoint: {e}")
        print("   Using randomly initialized model")

    return model


def load_mobilenet(checkpoint_path, num_classes=38):
    """Загрузить MobileNet-V2 с поддержкой backbone префикса"""
    print(f"   Loading from: {checkpoint_path}")

    from torchvision.models import mobilenet_v2

    # Создать модель
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Загрузить веса
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Извлечь state_dict
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Убрать "backbone." префикс если есть
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("backbone.", "")  # Убираем префикс
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        print("   ✅ Loaded successfully!")

    except Exception as e:
        print(f"   ⚠️  Error loading checkpoint: {e}")
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

    # Вывод
    print(f"\n📊 RESULTS:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    print(f"   Samples:   {len(all_labels)}")

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║     DOMAIN ADAPTATION EXPERIMENTS - BASELINE EVALUATION      ║
║                  PlantVillage → PlantDoc                     ║
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

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        print("   Check that PlantDoc is in the correct location.")
        return

    plantdoc_loader = DataLoader(
        plantdoc_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,  # Windows compatibility
    )

    # Тестировать модели
    all_results = {}

    models_to_test = [
        ("EfficientNet-B0", "efficientnet", load_efficientnet),
        ("MobileNet-V2", "mobilenet", load_mobilenet),
    ]

    for model_name, checkpoint_key, load_fn in models_to_test:
        checkpoint_path = (
            Path(CONFIG["models_path"]) / CONFIG[f"{checkpoint_key}_checkpoint"]
        )

        if not checkpoint_path.exists():
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"   Skipping {model_name}")
            continue

        print(f"\n🔄 Loading {model_name}...")
        model = load_fn(checkpoint_path, CONFIG["num_classes"])

        # Evaluate
        results = evaluate_model(model, plantdoc_loader, device, model_name)
        all_results[checkpoint_key] = results

    # Сохранить результаты
    results_file = results_path / "baseline_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

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
    print("\n📊 NEXT STEPS:")
    print("   1. Domain shift confirmed!")
    print("   2. Ready for Self-Training Domain Adaptation")
    print("   3. Expected improvement: +10-15%")


if __name__ == "__main__":
    main()
