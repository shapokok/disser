#!/usr/bin/env python3
"""
Evaluate the trained classifiers on the PlantVillage validation split and
write the real metrics the web app shows.

    python scripts/evaluate_models.py                    # all models found in models/
    python scripts/evaluate_models.py --models hybrid    # one model
    python scripts/evaluate_models.py --limit 2000       # quick run on a subset

Outputs (all consumed by the backend, nothing is hard-coded there any more):
    models/model_metrics.json                  summary per model (+ ensemble)
    results/metrics/<name>_validation.json     per-class metrics, confusion matrix,
                                               most confused pairs, timing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import MODEL_TYPES, ModelManager, count_parameters

DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(batch_size: int, workers: int, limit: int | None, class_names: list[str]):
    valid_dir = DATA_DIR / "valid"
    if not valid_dir.is_dir():
        sys.exit(f"Validation split not found at {valid_dir}")
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    ds = datasets.ImageFolder(valid_dir, transform=tf)
    if ds.classes != class_names:
        sys.exit("valid/ class folders differ from models/class_names.json")
    if limit:
        g = torch.Generator().manual_seed(0)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:limit].tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=workers > 0)


@torch.no_grad()
def collect_probs(model, loader, device, tag):
    probs, labels = [], []
    start = time.time()
    seen = 0
    for i, (x, y) in enumerate(loader, 1):
        out = model(x.to(device))
        probs.append(F.softmax(out, dim=1).cpu())
        labels.append(y)
        seen += y.size(0)
        if i % 20 == 0 or i == len(loader):
            print(f"  {tag}: {seen}/{len(loader.dataset)}  {seen / (time.time() - start):.0f} img/s", flush=True)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def summarise(name, probs, labels, class_names, elapsed_ms_per_image, model):
    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(class_names)), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    per_class = []
    for i, cname in enumerate(class_names):
        cls_acc = float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
        per_class.append(
            {
                "class_name": cname,
                "precision": float(p_cls[i]),
                "recall": float(r_cls[i]),
                "f1_score": float(f_cls[i]),
                "accuracy": cls_acc,
                "support": int(support[i]),
            }
        )

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:10], off.shape)):
        if off[i, j] == 0:
            break
        pairs.append(
            {
                "true_class": class_names[i],
                "predicted_class": class_names[j],
                "count": int(off[i, j]),
                "rate": float(off[i, j] / cm[i].sum()) if cm[i].sum() else 0.0,
            }
        )

    top5 = float(np.mean([labels[k] in np.argsort(probs[k])[::-1][:5] for k in range(len(labels))]))
    n_params = count_parameters(model)
    weights_file = MODELS_DIR / f"{name}_model.pth"

    validation = {
        "model_name": name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "num_samples": len(labels),
        "overall": {
            "accuracy": acc,
            "top5_accuracy": top5,
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f_macro),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confused_pairs": pairs,
        "inference_time_ms": elapsed_ms_per_image,
    }
    summary = {
        "accuracy": round(acc, 4),
        "top5_accuracy": round(top5, 4),
        "precision": round(float(p_macro), 4),
        "recall": round(float(r_macro), 4),
        "f1_score": round(float(f_macro), 4),
        "inference_time_ms": round(elapsed_ms_per_image, 1),
        "parameters": f"{n_params / 1e6:.1f}M",
        "parameters_count": n_params,
        "size_mb": round(weights_file.stat().st_size / 1e6, 1) if weights_file.exists() else None,
        "validation_samples": len(labels),
        "trained": acc > 0.5,
        "evaluated_at": validation["evaluated_at"],
    }
    return validation, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=list(MODEL_TYPES), default=list(MODEL_TYPES))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: leave torch's choice)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    manager = ModelManager(str(MODELS_DIR), device=device)
    class_names = manager.class_names
    loader = build_loader(args.batch_size, args.workers, args.limit, class_names)
    print(f"device={device}  samples={len(loader.dataset)}  classes={len(class_names)}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = MODELS_DIR / "model_metrics.json"
    all_metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    all_probs = {}
    labels = None

    for name in args.models:
        if not (MODELS_DIR / f"{name}_model.pth").exists():
            print(f"skip {name}: models/{name}_model.pth not found")
            continue
        model = manager.load_model(name, name)
        t0 = time.time()
        probs, labels = collect_probs(model, loader, device, name)
        ms_per_img = 1000 * (time.time() - t0) / len(labels)
        validation, summary = summarise(name, probs, labels, class_names, ms_per_img, model)
        (METRICS_DIR / f"{name}_validation.json").write_text(json.dumps(validation, indent=2))
        all_metrics[name] = summary
        all_probs[name] = probs
        print(f"==> {name}: acc {summary['accuracy']:.4f}  f1 {summary['f1_score']:.4f}  {ms_per_img:.1f} ms/img")

    # Ensemble of every *trained* model, weighted by accuracy (same rule as the backend)
    trained = {n: m for n, m in all_metrics.items() if n != "ensemble" and m.get("trained") and n in all_probs}
    if len(trained) >= 2 and labels is not None:
        w = np.array([trained[n]["accuracy"] for n in trained])
        ens = sum(all_probs[n] * wi for n, wi in zip(trained, w)) / w.sum()
        preds = ens.argmax(1)
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        all_metrics["ensemble"] = {
            "accuracy": round(float((preds == labels).mean()), 4),
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1_score": round(float(f_macro), 4),
            "inference_time_ms": round(sum(trained[n]["inference_time_ms"] for n in trained), 1),
            "parameters": f"{sum(trained[n]['parameters_count'] for n in trained) / 1e6:.1f}M",
            "size_mb": round(sum(trained[n]["size_mb"] or 0 for n in trained), 1),
            "models_combined": list(trained),
            "method": "accuracy-weighted average of softmax outputs",
            "validation_samples": len(labels),
            "trained": True,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"==> ensemble({', '.join(trained)}): acc {all_metrics['ensemble']['accuracy']:.4f}")

    metrics_file.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nwrote {metrics_file} and results/metrics/*_validation.json")


if __name__ == "__main__":
    main()
