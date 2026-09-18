from pathlib import Path

pv_path = Path(r"C:\Users\Нурислам\Desktop\disser\data\PlantVillage")

train_path = pv_path / "train"
if not train_path.exists():
    train_path = pv_path

print("=" * 60)
print("PLANTVILLAGE CLASS ORDER")
print("=" * 60)

classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])

for idx, cls in enumerate(classes):
    print(f"{idx:2d}: {cls}")

print(f"\nTotal: {len(classes)} classes")
