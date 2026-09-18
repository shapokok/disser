import torch
from baseline_evaluation_v2 import *

# Загрузить PlantVillage TEST set
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

from torchvision import datasets

pv_test = datasets.ImageFolder(
    root=r"C:\Users\Нурислам\Desktop\disser\data\PlantVillage\test",
    transform=transform,
)

loader = torch.utils.data.DataLoader(pv_test, batch_size=32, shuffle=False)

# Загрузить модель
model_path = CONFIG["models_path"] + "/" + CONFIG["efficientnet_checkpoint"]
model = load_efficientnet(model_path, 38)
model.eval()
model.cuda()

# Тест
correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nPlantVillage TEST accuracy: {accuracy:.2f}%")
print(f"Expected: ~98-99%")

if accuracy > 90:
    print("✅ Модель работает правильно на PlantVillage!")
else:
    print("❌ Модель не работает даже на PlantVillage")
