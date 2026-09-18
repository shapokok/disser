import torch
from baseline_evaluation_v2 import *
from PIL import Image

img_path = r"C:\Users\Нурислам\Desktop\disser\domain_adaptation_experiments\datasets\plantdoc\train\Apple leaf\Apple leaf (1).jpg"
img = Image.open(img_path)

model_path = CONFIG["models_path"] + "/" + CONFIG["efficientnet_checkpoint"]
model = load_efficientnet(model_path, 38)
model.eval().cuda()

print("Testing different normalizations:\n")

# Test 1: ImageNet norm (current)
t1 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img1 = t1(img).unsqueeze(0).cuda()
with torch.no_grad():
    out1 = model(img1)
    pred1 = torch.argmax(out1, 1).item()
print(f"1. ImageNet norm: [{pred1:2d}] {CONFIG['plantvillage_classes'][pred1]}")

# Test 2: No normalization
t2 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
img2 = t2(img).unsqueeze(0).cuda()
with torch.no_grad():
    out2 = model(img2)
    pred2 = torch.argmax(out2, 1).item()
print(f"2. No norm:       [{pred2:2d}] {CONFIG['plantvillage_classes'][pred2]}")

# Test 3: Different norm (0.5, 0.5, 0.5)
t3 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
img3 = t3(img).unsqueeze(0).cuda()
with torch.no_grad():
    out3 = model(img3)
    pred3 = torch.argmax(out3, 1).item()
print(f"3. 0.5/0.5 norm:  [{pred3:2d}] {CONFIG['plantvillage_classes'][pred3]}")

print(f"\nTrue class: [3] Apple___healthy")
