#!/usr/bin/env python3
"""
Train the classification models on PlantVillage.

Examples:
    python scripts/train_models.py                       # all four models
    python scripts/train_models.py --models hybrid       # one model
    python scripts/train_models.py --models hybrid --epochs 5 --batch-size 32 --device mps
    python scripts/train_models.py --models baseline --limit 512   # quick smoke test

Outputs:
    models/<name>_model.pth                 best weights (state_dict)
    results/checkpoints/<name>_epoch_N.pth  resumable checkpoints (last 2 kept)
    results/metrics/<name>_metrics.json     training curves + summary
    results/metrics/<name>_metrics.png      plot of the curves
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import (  # noqa: E402
    BaselineCNN,
    EfficientNetModel,
    HybridCNNTransformer,
    MobileNetModel,
)

DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_FACTORIES = {
    "baseline": BaselineCNN,
    "efficientnet": EfficientNetModel,
    "mobilenet": MobileNetModel,
    "hybrid": HybridCNNTransformer,
}

# Baseline is trained from scratch, the others fine-tune ImageNet weights.
DEFAULT_LR = {
    "baseline": 1e-3,
    "efficientnet": 1e-4,
    "mobilenet": 1e-4,
    "hybrid": 1e-4,
}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(batch_size: int, workers: int, img_size: int, limit: int | None):
    train_dir, valid_dir = DATA_DIR / "train", DATA_DIR / "valid"
    if not train_dir.is_dir() or not valid_dir.is_dir():
        sys.exit(
            f"Dataset not found. Expected {train_dir} and {valid_dir} "
            "(run scripts/download_dataset.py and scripts/split_train_valid.py)."
        )

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    valid_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    valid_ds = datasets.ImageFolder(valid_dir, transform=valid_tf)
    if train_ds.classes != valid_ds.classes:
        sys.exit("train/ and valid/ class folders differ")
    class_names = train_ds.classes

    if limit:
        g = torch.Generator().manual_seed(0)
        train_ds = Subset(train_ds, torch.randperm(len(train_ds), generator=g)[:limit].tolist())
        valid_ds = Subset(valid_ds, torch.randperm(len(valid_ds), generator=g)[: max(limit // 4, 64)].tolist())

    common = dict(num_workers=workers, persistent_workers=workers > 0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False, **common)

    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))
    return train_loader, valid_loader, class_names


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, log_every=100, tag=""):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    start = time.time()
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
            if i % log_every == 0 or i == len(loader):
                rate = seen / (time.time() - start)
                print(
                    f"  {tag} batch {i}/{len(loader)}  loss {total_loss/seen:.4f}  "
                    f"acc {100*correct/seen:.2f}%  {rate:.0f} img/s",
                    flush=True,
                )
    return total_loss / seen, 100.0 * correct / seen


def save_plot(name, hist):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(hist["train_losses"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, hist["train_losses"], label="train")
    ax[0].plot(epochs, hist["val_losses"], label="valid")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(epochs, hist["train_accuracies"], label="train")
    ax[1].plot(epochs, hist["val_accuracies"], label="valid")
    ax[1].set(title="Accuracy (%)", xlabel="epoch")
    ax[1].legend()
    fig.suptitle(f"{name}: best valid acc {max(hist['val_accuracies']):.2f}%")
    fig.tight_layout()
    fig.savefig(METRICS_DIR / f"{name}_metrics.png", dpi=150)
    plt.close(fig)


def train_one(name, args, device, train_loader, valid_loader, class_names):
    print("=" * 70)
    print(f"Training {name} on {device}  epochs={args.epochs} batch={args.batch_size}")
    print("=" * 70, flush=True)

    model = MODEL_FACTORIES[name](num_classes=len(class_names)).to(device)
    lr = args.lr or DEFAULT_LR[name]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(300, steps_per_epoch)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    hist = {k: [] for k in ("train_losses", "val_losses", "train_accuracies", "val_accuracies", "epoch_times")}
    start_epoch, best_acc = 0, 0.0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(CHECKPOINT_DIR.glob(f"{name}_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    if args.resume and ckpts:
        ck = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        hist, start_epoch, best_acc = ck["history"], ck["epoch"], ck["best_val_acc"]
        print(f"Resumed from {ckpts[-1].name} (epoch {start_epoch}, best {best_acc:.2f}%)")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, tag="train")
        va_loss, va_acc = run_epoch(model, valid_loader, criterion, device, tag="valid")
        dt = time.time() - t0
        for k, v in zip(hist, (tr_loss, va_loss, tr_acc, va_acc, dt)):
            hist[k].append(v)
        print(f"==> epoch {epoch + 1}: train {tr_acc:.2f}%  valid {va_acc:.2f}%  ({dt/60:.1f} min)", flush=True)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), MODELS_DIR / f"{name}_model.pth")
            print(f"    new best -> models/{name}_model.pth", flush=True)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_acc,
                "history": hist,
            },
            CHECKPOINT_DIR / f"{name}_epoch_{epoch + 1}.pth",
        )
        for old in sorted(CHECKPOINT_DIR.glob(f"{name}_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))[:-2]:
            old.unlink()

    summary = {
        "model_name": name,
        **hist,
        "best_val_accuracy": best_acc,
        "final_train_accuracy": hist["train_accuracies"][-1] if hist["train_accuracies"] else None,
        "total_time_minutes": sum(hist["epoch_times"]) / 60,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "optimizer": "AdamW + warmup/cosine",
        "device": str(device),
        "num_classes": len(class_names),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    (METRICS_DIR / f"{name}_metrics.json").write_text(json.dumps(summary, indent=2))
    save_plot(name, hist)
    print(f"\n{name}: best valid accuracy {best_acc:.2f}%  ({summary['total_time_minutes']:.1f} min)")
    return best_acc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=list(MODEL_FACTORIES), default=list(MODEL_FACTORIES))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="override the per-model default")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    p.add_argument("--limit", type=int, default=None, help="use only N training images (smoke test)")
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(42)
    train_loader, valid_loader, class_names = build_loaders(args.batch_size, args.workers, args.img_size, args.limit)
    print(f"train {len(train_loader.dataset)} / valid {len(valid_loader.dataset)} images, {len(class_names)} classes")

    results = {}
    for name in args.models:
        results[name] = train_one(name, args, device, train_loader, valid_loader, class_names)

    print("\nSummary (best valid accuracy):")
    for name, acc in results.items():
        print(f"  {name:13s} {acc:.2f}%")


if __name__ == "__main__":
    main()
