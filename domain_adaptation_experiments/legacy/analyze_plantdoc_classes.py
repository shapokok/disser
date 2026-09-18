from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from model import EfficientNetModel

# PlantVillage классы (твой порядок)
PV_CLASSES = [
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

# Загрузить модель
model = EfficientNetModel(num_classes=38, pretrained=False)
model.load_state_dict(
    torch.load(r"C:\Users\Нурислам\Desktop\disser\models\efficientnet_model.pth")
)
model.eval().cuda()

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Проверить все PlantDoc классы
plantdoc_train = Path(
    r"C:\Users\Нурислам\Desktop\disser\domain_adaptation_experiments\datasets\plantdoc\train"
)

print("=" * 80)
print("ANALYZING PLANTDOC CLASSES")
print("=" * 80)

results = {}

for class_dir in sorted(plantdoc_train.iterdir()):
    if not class_dir.is_dir():
        continue

    plantdoc_class = class_dir.name

    # Взять первые 10 изображений
    images = list(class_dir.glob("*.jpg"))[:10]

    if not images:
        continue

    predictions = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).cuda()

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, 1).item()
            predictions.append(pred_idx)

    # Найти самый частый класс
    from collections import Counter

    most_common = Counter(predictions).most_common(1)[0]
    predicted_class_idx = most_common[0]
    confidence = most_common[1] / len(predictions) * 100

    predicted_class_name = PV_CLASSES[predicted_class_idx]

    results[plantdoc_class] = {
        "predicted_idx": predicted_class_idx,
        "predicted_name": predicted_class_name,
        "confidence": confidence,
        "all_predictions": predictions,
    }

    print(f"\n{plantdoc_class}")
    print(f"  → [{predicted_class_idx:2d}] {predicted_class_name}")
    print(
        f"  Confidence: {confidence:.0f}% ({most_common[1]}/{len(predictions)} images)"
    )

# Создать ПРАВИЛЬНЫЙ маппинг
print("\n" + "=" * 80)
print("SUGGESTED CORRECT MAPPING:")
print("=" * 80)
print("\nmapping = {")
for plantdoc_class, result in sorted(results.items()):
    if result["confidence"] >= 70:  # Только уверенные
        print(
            f'    "{plantdoc_class}": {result["predicted_idx"]},  # {result["predicted_name"]}'
        )
print("}")
