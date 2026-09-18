#!/usr/bin/env python3
"""
Robustness of the classifiers to image degradations that occur in the field:
blur, Gaussian noise, JPEG compression, darkening / over-exposure, down-scaling.

    python scripts/robustness_eval.py                    # all trained models, 1 000 valid images per corruption
    python scripts/robustness_eval.py --models efficientnet mobilenet --limit 500 --device mps

Writes results/metrics/robustness.json and results/metrics/robustness.md
(accuracy per corruption level; the "clean" row is the reference).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from explain import to_tensor
from model import MODEL_TYPES, ModelManager, pick_device

DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "results" / "metrics"


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.RandomState(0).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def downscale(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR).resize(img.size, Image.BILINEAR)


CORRUPTIONS = {
    "clean": lambda im: im,
    "blur_r1": lambda im: im.filter(ImageFilter.GaussianBlur(1)),
    "blur_r2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "blur_r4": lambda im: im.filter(ImageFilter.GaussianBlur(4)),
    "noise_s10": lambda im: noise(im, 10),
    "noise_s25": lambda im: noise(im, 25),
    "jpeg_q30": lambda im: jpeg(im, 30),
    "jpeg_q10": lambda im: jpeg(im, 10),
    "dark_0.5": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
    "dark_0.3": lambda im: ImageEnhance.Brightness(im).enhance(0.3),
    "bright_1.6": lambda im: ImageEnhance.Brightness(im).enhance(1.6),
    "lowres_64": lambda im: downscale(im, 64),
    "lowres_32": lambda im: downscale(im, 32),
}


@torch.no_grad()
def accuracy(model, device, samples, fn, batch=32) -> float:
    correct = 0
    for i in range(0, len(samples), batch):
        chunk = samples[i : i + batch]
        x = torch.cat([to_tensor(fn(Image.open(p).convert("RGB"))) for p, _ in chunk]).to(device)
        y = torch.tensor([lbl for _, lbl in chunk], device=device)
        correct += (F.softmax(model(x), 1).argmax(1) == y).sum().item()
    return correct / len(samples)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=list(MODEL_TYPES), default=None)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    manager = ModelManager(str(MODELS_DIR), device=device)

    ds = datasets.ImageFolder(DATA_DIR / "valid")
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(ds.samples))[: args.limit]
    samples = [ds.samples[i] for i in idx]
    metrics = (
        json.loads((MODELS_DIR / "model_metrics.json").read_text())
        if (MODELS_DIR / "model_metrics.json").exists()
        else {}
    )
    names = args.models or [
        n for n in MODEL_TYPES if metrics.get(n, {}).get("trained") and (MODELS_DIR / f"{n}_model.pth").exists()
    ]

    results = {"limit": len(samples), "device": str(device), "corruptions": list(CORRUPTIONS), "models": {}}
    for name in names:
        model = manager.load_model(name, name)
        results["models"][name] = {}
        for cname, fn in CORRUPTIONS.items():
            acc = accuracy(model, device, samples, fn)
            results["models"][name][cname] = acc
            print(f"{name:13s} {cname:12s} {acc * 100:6.2f}%", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "robustness.json").write_text(json.dumps(results, indent=2))
    header = "| Искажение | " + " | ".join(MODEL_TYPES[n]["label"] for n in names) + " |"
    lines = [
        "# Робастность к искажениям (accuracy, %)",
        "",
        f"{len(samples)} изображений valid, device {device}",
        "",
        header,
        "|---|" + "---|" * len(names),
    ]
    for cname in CORRUPTIONS:
        lines.append(f"| {cname} | " + " | ".join(f"{100 * results['models'][n][cname]:.1f}" for n in names) + " |")
    (OUT_DIR / "robustness.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR / 'robustness.json'}")


if __name__ == "__main__":
    main()
