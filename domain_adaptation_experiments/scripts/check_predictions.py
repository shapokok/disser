import torch
from baseline_evaluation_v2 import *

# Загрузить данные
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

class_mapping = create_class_mapping()
dataset = PlantDocDataset(
    CONFIG["plantdoc_path"],
    split="train",
    transform=transform,
    class_mapping=class_mapping,
)

# Загрузить модель
model_path = CONFIG["models_path"] + "/" + CONFIG["efficientnet_checkpoint"]
model = load_efficientnet(model_path, 38)
model.eval()
model.cuda()

# Проверить несколько предсказаний
print("\n" + "=" * 60)
print("CHECKING PREDICTIONS")
print("=" * 60)

for i in range(10):
    img, label = dataset[i]
    img_batch = img.unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img_batch)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
        confidence = probs[0, pred].item()

    true_class = CONFIG["plantvillage_classes"][label]
    pred_class = CONFIG["plantvillage_classes"][pred] if pred < 38 else "UNKNOWN"

    print(f"\nSample {i+1}:")
    print(f"  True:  [{label:2d}] {true_class}")
    print(f"  Pred:  [{pred:2d}] {pred_class}")
    print(f"  Conf:  {confidence:.2%}")
    print(f"  Match: {'✅' if pred == label else '❌'}")

# Статистика предсказаний
print("\n" + "=" * 60)
print("PREDICTION STATISTICS")
print("=" * 60)

all_preds = []
all_labels = []

for i in range(min(100, len(dataset))):
    img, label = dataset[i]
    img_batch = img.unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img_batch)
        pred = torch.argmax(output, dim=1).item()

    all_preds.append(pred)
    all_labels.append(label)

# Какие классы модель предсказывает?
unique_preds = set(all_preds)
unique_labels = set(all_labels)

print(f"\nTrue labels (PlantDoc classes): {sorted(unique_labels)}")
print(f"Total unique true classes: {len(unique_labels)}")

print(f"\nPredicted labels: {sorted(unique_preds)}")
print(f"Total unique predicted classes: {len(unique_preds)}")

# Overlap
overlap = unique_preds & unique_labels
print(f"\nOverlap: {len(overlap)} classes")
print(f"Predicted classes NOT in true labels: {sorted(unique_preds - unique_labels)}")
