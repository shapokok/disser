import torch

# Проверим EfficientNet checkpoint
path = r"C:\Users\Нурислам\Desktop\disser\models\efficientnet_model.pth"

print("=" * 60)
print("CHECKING CHECKPOINT FORMAT")
print("=" * 60)

checkpoint = torch.load(path, map_location="cpu")

print(f"\nType: {type(checkpoint)}")
print(f"\nKeys in checkpoint:")

if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        print(f"  - {key}")
        if key in ["model_state_dict", "state_dict"]:
            state_dict = checkpoint[key]
            print(f"\n    First 5 layers in {key}:")
            for i, layer_key in enumerate(list(state_dict.keys())[:5]):
                print(f"      {layer_key}: {state_dict[layer_key].shape}")
else:
    print("\nDirect state_dict (first 5 layers):")
    for i, layer_key in enumerate(list(checkpoint.keys())[:5]):
        print(f"  {layer_key}: {checkpoint[layer_key].shape}")

print("\n" + "=" * 60)
