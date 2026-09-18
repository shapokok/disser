import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import EfficientNetModel, MobileNetModel

print("=" * 60)
print("TESTING ON PLANTVILLAGE VALIDATION SET")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Load PlantVillage validation
pv_path = Path(r"C:\Users\Нурислам\Desktop\disser\data\PlantVillage")

# Найти validation set
if (pv_path / "valid").exists():
    val_path = pv_path / "valid"
elif (pv_path / "val").exists():
    val_path = pv_path / "val"
elif (pv_path / "test").exists():
    val_path = pv_path / "test"
else:
    print(f"\n❌ Can't find validation set!")
    print(f"Available folders:")
    for f in pv_path.iterdir():
        if f.is_dir():
            print(f"  - {f.name}")
    sys.exit(1)

print(f"\nUsing validation set: {val_path}")

val_dataset = datasets.ImageFolder(val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"✓ Loaded {len(val_dataset)} validation images")
print(f"✓ Classes: {len(val_dataset.classes)}")

# Test EfficientNet
print("\n" + "=" * 60)
print("TESTING EFFICIENTNET")
print("=" * 60)

model_path = Path(r"C:\Users\Нурислам\Desktop\disser\models\efficientnet_model.pth")
model = EfficientNetModel(num_classes=38, pretrained=False)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"\n📊 PlantVillage Validation Accuracy: {accuracy:.2f}%")
print(f"   Correct: {correct} / {total}")

if accuracy > 95:
    print("\n✅ MODEL WORKS PERFECTLY!")
    print("   Problem is with PlantDoc mapping or data")
elif accuracy > 80:
    print("\n✅ Model works well")
    print("   But not perfect - may need retraining")
elif accuracy > 50:
    print("\n⚠️  Model partially works")
    print("   Significant issues")
else:
    print("\n❌ MODEL DOESN'T WORK!")
    print("   Model weights are wrong or corrupted")
