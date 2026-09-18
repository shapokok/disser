"""
Shared code for the domain adaptation experiments (PlantVillage -> PlantDoc).

Evaluation protocol (honest version, replaces the legacy scripts that
evaluated on their own training images):

    PlantDoc "train" split of the 4 focus classes (468 images)
        -> stratified 80/20 split with a fixed seed
        -> "adapt" (375 images): used for adaptation (labels only for the
           supervised methods, never for self-training)
        -> "dev"   (93 images):  model selection / early stopping
    PlantDoc "test" split of the 4 focus classes (32 images): held-out test
    "eval" = dev + test (125 images) is reported as the headline number,
    with dev and test shown separately and Wilson 95 % intervals.

Paths are resolved relative to the repository, or overridden with
PLANTDOC_DIR / DA_RESULTS_DIR environment variables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

DA_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DA_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import MobileNetModel, pick_device  # noqa: E402

PLANTDOC_DIR = Path(os.environ.get("PLANTDOC_DIR", DA_DIR / "datasets" / "plantdoc"))
RESULTS_DIR = Path(os.environ.get("DA_RESULTS_DIR", DA_DIR / "results"))
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = DA_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
SOURCE_CHECKPOINT = MODELS_DIR / "mobilenet_model.pth"

with open(MODELS_DIR / "class_names.json", encoding="utf-8") as _f:
    CLASS_NAMES: list[str] = json.load(_f)

# PlantDoc folder name -> PlantVillage class index (27 of 38 classes overlap)
PLANTDOC_TO_PV = {
    "Apple Scab Leaf": 0,
    "Apple rust leaf": 2,
    "Apple leaf": 3,
    "Blueberry leaf": 4,
    "Cherry leaf": 6,
    "Corn Gray leaf spot": 7,
    "Corn rust leaf": 8,
    "Corn leaf blight": 9,
    "grape leaf black rot": 11,
    "grape leaf": 14,
    "Peach leaf": 17,
    "Bell_pepper leaf spot": 18,
    "Bell_pepper leaf": 19,
    "Potato leaf early blight": 20,
    "Potato leaf late blight": 21,
    "Raspberry leaf": 23,
    "Soyabean leaf": 24,
    "Squash Powdery mildew leaf": 25,
    "Strawberry leaf": 27,
    "Tomato leaf bacterial spot": 28,
    "Tomato Early blight leaf": 29,
    "Tomato leaf late blight": 30,
    "Tomato mold leaf": 31,
    "Tomato Septoria leaf spot": 32,
    "Tomato leaf yellow virus": 35,
    "Tomato leaf mosaic virus": 36,
    "Tomato leaf": 37,
}

# The four classes with enough PlantDoc images to adapt on (the thesis' focus).
FOCUS_CLASSES = {
    "Corn Gray leaf spot": 7,
    "Corn leaf blight": 9,
    "Squash Powdery mildew leaf": 25,
    "Tomato leaf late blight": 30,
}
EASY_CLASSES = {"Squash Powdery mildew leaf": 25, "Tomato leaf late blight": 30}
HARD_CLASSES = {"Corn Gray leaf spot": 7, "Corn leaf blight": 9}
FOCUS_IDX = sorted(FOCUS_CLASSES.values())

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEV_FRACTION = 0.2
SPLIT_SEED = 42
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------- utils
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def short_name(idx: int) -> str:
    return CLASS_NAMES[idx].split("___")[-1].replace("_", " ").strip()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"saved {path.relative_to(PROJECT_ROOT)}")


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- data
def list_images(split: str, mapping: dict[str, int]) -> list[tuple[str, int]]:
    root = PLANTDOC_DIR / split
    if not root.is_dir():
        raise FileNotFoundError(f"PlantDoc split not found: {root}")
    items = []
    for folder, label in mapping.items():
        d = root / folder
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXT:
                items.append((str(p.relative_to(PLANTDOC_DIR)), label))
    return items


def make_splits(mapping: dict[str, int] = FOCUS_CLASSES, dev_fraction: float = DEV_FRACTION, seed: int = SPLIT_SEED) -> dict[str, list]:
    """Stratified adapt/dev split of PlantDoc train + the official test split. Cached on disk."""
    key = hashlib.md5(json.dumps(sorted(mapping.items())).encode()).hexdigest()[:8]
    cache = RESULTS_DIR / f"splits_{key}_seed{seed}.json"
    if cache.exists():
        return load_json(cache)

    train = list_images("train", mapping)
    rng = random.Random(seed)
    adapt, dev = [], []
    by_label: dict[int, list] = {}
    for item in train:
        by_label.setdefault(item[1], []).append(item)
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        n_dev = max(1, round(len(items) * dev_fraction))
        dev += items[:n_dev]
        adapt += items[n_dev:]
    splits = {"adapt": sorted(adapt), "dev": sorted(dev), "test": list_images("test", mapping)}
    save_json({**splits, "_meta": {"seed": seed, "dev_fraction": dev_fraction, "classes": mapping}}, cache)
    return splits


class PlantDocDataset(Dataset):
    def __init__(self, items: list, transform=None, pseudo_labels: dict[str, int] | None = None):
        self.items = list(items)
        self.transform = transform
        self.pseudo_labels = pseudo_labels or {}

    @property
    def labels(self) -> list[int]:
        return [self.pseudo_labels.get(p, l) for p, l in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        label = self.pseudo_labels.get(rel, label)
        try:
            img = Image.open(PLANTDOC_DIR / rel).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, label, i


def eval_transform():
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def heavy_augment():
    """The 'super heavy' augmentation from the original joint-training experiment."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(25),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.15)),
        ]
    )


def light_augment():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def make_loader(items, transform, batch_size=32, shuffle=False, balanced=False, workers=0, pseudo_labels=None):
    ds = PlantDocDataset(items, transform, pseudo_labels)
    sampler = None
    if balanced and len(ds):
        counts = Counter(ds.labels)
        weights = [1.0 / counts[l] for l in ds.labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=workers, drop_last=False)


# --------------------------------------------------------------------------- models
def load_source_model(device) -> nn.Module:
    """MobileNet-V2 trained on PlantVillage (the source domain)."""
    return load_checkpoint(SOURCE_CHECKPOINT, device)


def load_checkpoint(path: Path, device) -> nn.Module:
    model = MobileNetModel(num_classes=len(CLASS_NAMES), pretrained=False)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def save_checkpoint(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        out = torch.softmax(model(x.to(device)), dim=1)
        probs.append(out.float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def restricted_argmax(probs: np.ndarray, allowed: list[int]) -> np.ndarray:
    masked = np.full_like(probs, -1.0)
    masked[:, allowed] = probs[:, allowed]
    return masked.argmax(1)


def compute_metrics(labels: np.ndarray, preds: np.ndarray, classes: list[int] = FOCUS_IDX) -> dict:
    correct = int((labels == preds).sum())
    n = int(len(labels))
    acc = correct / n if n else 0.0
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=classes, zero_division=0)
    per_class = {}
    for i, c in enumerate(classes):
        mask = labels == c
        per_class[str(c)] = {
            "name": short_name(c),
            "accuracy": float((preds[mask] == c).mean()) if mask.any() else 0.0,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
    # Confusion matrix over the focus classes; predictions outside them go to an "other" column.
    other = -1
    preds_cm = np.where(np.isin(preds, classes), preds, other)
    cm = confusion_matrix(labels, preds_cm, labels=classes + [other])[: len(classes)]
    lo, hi = wilson_ci(correct, n)
    return {
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "ci95": [lo, hi],
        "macro_f1": float(f.mean()) if len(f) else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": [short_name(c) for c in classes] + ["other"],
    }


def evaluate_model(model: nn.Module, device, splits: dict[str, list], batch_size=64, workers=0, classes: list[int] = FOCUS_IDX) -> dict:
    """Evaluate on dev, test and dev+test; both open-set (38-way) and restricted (4-way) argmax."""
    out = {}
    tf = eval_transform()
    cache = {}
    for name in ("dev", "test"):
        loader = make_loader(splits[name], tf, batch_size=batch_size, workers=workers)
        cache[name] = predict(model, loader, device)
    cache["eval"] = (np.concatenate([cache["dev"][0], cache["test"][0]]), np.concatenate([cache["dev"][1], cache["test"][1]]))
    for name, (probs, labels) in cache.items():
        out[name] = {
            "open": compute_metrics(labels, probs.argmax(1), classes),
            "restricted": compute_metrics(labels, restricted_argmax(probs, classes), classes),
        }
    return out


def headline(result: dict, split: str = "eval", mode: str = "open") -> str:
    m = result[split][mode]
    lo, hi = m["ci95"]
    return f"{m['accuracy']*100:.1f}% (95% CI {lo*100:.0f}–{hi*100:.0f}, n={m['n']}, macro-F1 {m['macro_f1']*100:.1f})"


# --------------------------------------------------------------------------- training
def train_epochs(model, loader, device, epochs, lr, weight_decay=1e-4, on_epoch_end=None, label_smoothing=0.0, log_prefix=""):
    """AdamW + cosine schedule. `on_epoch_end(epoch, train_loss, train_acc)` may return a dict merged into the history."""
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": loss_sum / max(1, total), "train_acc": correct / max(1, total), "seconds": time.time() - t0}
        if on_epoch_end:
            row.update(on_epoch_end(epoch, row["train_loss"], row["train_acc"]) or {})
        history.append(row)
        extra = "  ".join(f"{k} {v*100:.1f}%" for k, v in row.items() if k.endswith("_acc") and k != "train_acc")
        print(f"{log_prefix}epoch {epoch:3d}/{epochs}  loss {row['train_loss']:.4f}  train {row['train_acc']*100:.1f}%  {extra}  ({row['seconds']:.0f}s)", flush=True)
    return history


def dev_accuracy_fn(model, device, splits, batch_size=64, workers=0, mode="open"):
    loader = make_loader(splits["dev"], eval_transform(), batch_size=batch_size, workers=workers)

    def fn(*_):
        probs, labels = predict(model, loader, device)
        preds = probs.argmax(1) if mode == "open" else restricted_argmax(probs, FOCUS_IDX)
        return {"dev_acc": float((preds == labels).mean())}

    return fn


def describe_splits(splits: dict) -> dict:
    return {name: {"n": len(items), "per_class": {short_name(c): sum(1 for _, l in items if l == c) for c in FOCUS_IDX}} for name, items in splits.items() if not name.startswith("_")}


def common_args(parser):
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def setup(args):
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = pick_device(args.device)
    splits = make_splits()
    print(f"device={device}  splits={ {k: v['n'] for k, v in describe_splits(splits).items()} }")
    return device, splits
