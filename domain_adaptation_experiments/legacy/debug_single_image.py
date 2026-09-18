import torch
from baseline_evaluation_v2 import *
from PIL import Image
import matplotlib.pyplot as plt

# Загрузить данные
class_mapping = create_class_mapping()
dataset = PlantDocDataset(
    CONFIG["plantdoc_path"],
    split="train",
    transform=None,  # БЕЗ transform сначала
    class_mapping=class_mapping,
)

# Взять первое яблоко
print("Ищем Apple leaf...")
for i in range(len(dataset.images)):
    if "Apple leaf" in dataset.images[i]:
        idx = i
        break

img_path = dataset.images[idx]
true_label = dataset.labels[idx]
orig_class = dataset.original_classes[idx]

print(f"\nImage: {img_path}")
print(f"PlantDoc class: {orig_class}")
print(
    f"Mapped to PV class [{true_label}]: {CONFIG['plantvillage_classes'][true_label]}"
)

# Открыть и показать
img = Image.open(img_path)
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")

# Применить transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img_tensor = transform(img).unsqueeze(0).cuda()

# Загрузить модель
model_path = CONFIG["models_path"] + "/" + CONFIG["efficientnet_checkpoint"]
model = load_efficientnet(model_path, 38)
model.eval()
model.cuda()

# Предсказать
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)[0]

# Топ-5 предсказаний
top5_probs, top5_idx = torch.topk(probs, 5)

print(f"\n{'='*60}")
print("TOP 5 PREDICTIONS:")
print(f"{'='*60}")
for i, (prob, idx) in enumerate(zip(top5_probs, top5_idx)):
    pred_class = CONFIG["plantvillage_classes"][idx.item()]
    marker = "✅" if idx.item() == true_label else "❌"
    print(f"{i+1}. [{idx.item():2d}] {pred_class:50s} {prob.item():6.2%} {marker}")

print(f"\n{'='*60}")
print(f"TRUE CLASS: [{true_label:2d}] {CONFIG['plantvillage_classes'][true_label]}")
print(f"Probability: {probs[true_label].item():.4%}")
