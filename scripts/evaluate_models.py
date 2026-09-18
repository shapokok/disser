#!/usr/bin/env python3
"""
Evaluate the trained classifiers on the PlantVillage validation split and
write the real metrics the web app shows.

    python scripts/evaluate_models.py                    # all models found in models/
    python scripts/evaluate_models.py --models hybrid    # one model
    python scripts/evaluate_models.py --limit 2000       # quick run on a subset
    python scripts/evaluate_models.py --analysis-only    # recompute calibration / McNemar / ensemble
                                                         # from the saved probabilities, no inference

Outputs (all consumed by the backend, nothing is hard-coded there any more):
    models/model_metrics.json                  summary per model (+ ensemble), incl. calibration
    results/metrics/<name>_validation.json     per-class metrics, confusion matrix, confused pairs,
                                               reliability diagram bins, timing
    results/metrics/<name>_probs.npz           softmax outputs + labels (for post-hoc analysis)
    results/metrics/model_comparison.json      pairwise McNemar tests, de-duplicated accuracy
"""

from __future__ import annotations

import argparse
import itertools
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
TRAINED_THRESHOLD = 0.5


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- data
def build_dataset(class_names: list[str], limit: int | None):
    valid_dir = DATA_DIR / "valid"
    if not valid_dir.is_dir():
        sys.exit(f"Validation split not found at {valid_dir}")
    tf = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    ds = datasets.ImageFolder(valid_dir, transform=tf)
    if ds.classes != class_names:
        sys.exit("valid/ class folders differ from models/class_names.json")
    paths = [str(Path(p).relative_to(DATA_DIR)) for p, _ in ds.samples]
    if limit:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[:limit].tolist()
        ds = Subset(ds, idx)
        paths = [paths[i] for i in idx]
    return ds, paths


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


# --------------------------------------------------------------------------- calibration
def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0, "accuracy": None, "confidence": None})
            continue
        acc, c = float(correct[m].mean()), float(conf[m].mean())
        gap = abs(acc - c)
        ece += gap * m.sum() / len(conf)
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()), "accuracy": acc, "confidence": c})
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    brier = float(((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(1).mean())
    return {
        "ece": float(ece),
        "mce": float(mce),
        "nll": nll,
        "brier": brier,
        "mean_confidence": float(conf.mean()),
        "bins": bins,
    }


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017) on log-probabilities, fitted with LBFGS."""
    logits = torch.log(torch.tensor(probs, dtype=torch.float64).clamp_min(1e-12))
    y = torch.tensor(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1)) / t
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def calibration_report(probs: np.ndarray, labels: np.ndarray) -> dict:
    """ECE before/after temperature scaling; T is fitted on one half of valid and applied to the other."""
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(labels))
    fit_idx, test_idx = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    raw = reliability(probs, labels)
    t = fit_temperature(probs[fit_idx], labels[fit_idx])
    scaled = reliability(apply_temperature(probs[test_idx], t), labels[test_idx])
    raw_half = reliability(probs[test_idx], labels[test_idx])
    return {
        "ece": raw["ece"],
        "mce": raw["mce"],
        "nll": raw["nll"],
        "brier": raw["brier"],
        "mean_confidence": raw["mean_confidence"],
        "temperature": t,
        "ece_after_temperature": scaled["ece"],
        "ece_before_temperature_same_half": raw_half["ece"],
        "bins": raw["bins"],
    }


# --------------------------------------------------------------------------- statistics
def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    from scipy.stats import chi2

    b = int((a_correct & ~b_correct).sum())  # A right, B wrong
    c = int((~a_correct & b_correct).sum())  # A wrong, B right
    if b + c == 0:
        return {"a_only": b, "b_only": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"a_only": b, "b_only": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1))}


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
    n_params = count_parameters(model) if model is not None else None
    weights_file = MODELS_DIR / f"{name}_model.pth"
    calib = calibration_report(probs, labels)

    validation = {
        "model_name": name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "num_samples": int(len(labels)),
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
        "calibration": calib,
        "inference_time_ms": elapsed_ms_per_image,
    }
    summary = {
        "accuracy": round(acc, 4),
        "top5_accuracy": round(top5, 4),
        "precision": round(float(p_macro), 4),
        "recall": round(float(r_macro), 4),
        "f1_score": round(float(f_macro), 4),
        "ece": round(calib["ece"], 4),
        "ece_after_temperature": round(calib["ece_after_temperature"], 4),
        "temperature": round(calib["temperature"], 3),
        "nll": round(calib["nll"], 4),
        "inference_time_ms": round(elapsed_ms_per_image, 1) if elapsed_ms_per_image is not None else None,
        "parameters": f"{n_params / 1e6:.1f}M" if n_params else None,
        "parameters_count": n_params,
        "size_mb": round(weights_file.stat().st_size / 1e6, 1) if weights_file.exists() else None,
        "validation_samples": int(len(labels)),
        "trained": acc > TRAINED_THRESHOLD,
        "evaluated_at": validation["evaluated_at"],
    }
    return validation, summary


def load_saved(name):
    f = METRICS_DIR / f"{name}_probs.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return z["probs"], z["labels"], list(z["paths"]), float(z["ms_per_image"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=list(MODEL_TYPES), default=list(MODEL_TYPES))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--analysis-only", action="store_true", help="reuse results/metrics/<name>_probs.npz, skip inference"
    )
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    manager = ModelManager(str(MODELS_DIR), device=device)
    class_names = manager.class_names
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = MODELS_DIR / "model_metrics.json"
    all_metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}

    ds = paths = loader = None
    if not args.analysis_only:
        ds, paths = build_dataset(class_names, args.limit)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0
        )
        print(f"device={device}  samples={len(ds)}  classes={len(class_names)}")

    results = {}  # name -> (probs, labels, paths)
    for name in args.models:
        model = None
        if args.analysis_only:
            saved = load_saved(name)
            if saved is None:
                print(f"skip {name}: no saved probabilities")
                continue
            probs, labels, npaths, ms_per_img = saved
            if (MODELS_DIR / f"{name}_model.pth").exists():
                model = manager.load_model(name, name)
        else:
            if not (MODELS_DIR / f"{name}_model.pth").exists():
                print(f"skip {name}: models/{name}_model.pth not found")
                continue
            model = manager.load_model(name, name)
            t0 = time.time()
            probs, labels = collect_probs(model, loader, device, name)
            ms_per_img = 1000 * (time.time() - t0) / len(labels)
            npaths = paths
            np.savez_compressed(
                METRICS_DIR / f"{name}_probs.npz",
                probs=probs.astype(np.float32),
                labels=labels,
                paths=np.array(npaths),
                ms_per_image=ms_per_img,
            )
        validation, summary = summarise(name, probs, labels, class_names, ms_per_img, model)
        (METRICS_DIR / f"{name}_validation.json").write_text(json.dumps(validation, indent=2))
        all_metrics[name] = summary
        results[name] = (probs, labels, npaths)
        print(
            f"==> {name}: acc {summary['accuracy']:.4f}  f1 {summary['f1_score']:.4f}  "
            f"ECE {summary['ece']:.4f}  {ms_per_img:.1f} ms/img"
        )

    # --- ensemble of every trained model, accuracy-weighted (same rule as the backend)
    trained = [n for n in results if all_metrics[n]["trained"]]
    labels = next(iter(results.values()))[1] if results else None
    if len(trained) >= 2:
        w = np.array([all_metrics[n]["accuracy"] for n in trained])
        ens = sum(results[n][0] * wi for n, wi in zip(trained, w)) / w.sum()
        preds = ens.argmax(1)
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        calib = reliability(ens, labels)
        all_metrics["ensemble"] = {
            "accuracy": round(float((preds == labels).mean()), 4),
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1_score": round(float(f_macro), 4),
            "ece": round(calib["ece"], 4),
            "nll": round(calib["nll"], 4),
            "inference_time_ms": round(sum(all_metrics[n]["inference_time_ms"] or 0 for n in trained), 1),
            "parameters": f"{sum(all_metrics[n]['parameters_count'] or 0 for n in trained) / 1e6:.1f}M",
            "size_mb": round(sum(all_metrics[n]["size_mb"] or 0 for n in trained), 1),
            "models_combined": trained,
            "method": "accuracy-weighted average of softmax outputs",
            "validation_samples": int(len(labels)),
            "trained": True,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        results["ensemble"] = (ens, labels, results[trained[0]][2])
        print(f"==> ensemble({', '.join(trained)}): acc {all_metrics['ensemble']['accuracy']:.4f}")

    # --- pairwise McNemar tests and de-duplicated accuracy
    comparison = {"evaluated_at": datetime.now().isoformat(timespec="seconds"), "mcnemar": [], "deduplicated": {}}
    names = list(results)
    for a, b in itertools.combinations(names, 2):
        ca = results[a][0].argmax(1) == results[a][1]
        cb = results[b][0].argmax(1) == results[b][1]
        comparison["mcnemar"].append(
            {"a": a, "b": b, "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()), **mcnemar(ca, cb)}
        )
    dup_file = METRICS_DIR / "duplicates.json"
    if dup_file.exists() and results:
        leaked = {d["valid"] for d in json.loads(dup_file.read_text())["leaked"]}
        for n, (pr, lb, pth) in results.items():
            keep = np.array([pp not in leaked for pp in pth])
            if keep.any():
                comparison["deduplicated"][n] = {
                    "kept": int(keep.sum()),
                    "removed": int((~keep).sum()),
                    "accuracy_all": float((pr.argmax(1) == lb).mean()),
                    "accuracy_dedup": float((pr.argmax(1)[keep] == lb[keep]).mean()),
                }
                all_metrics[n]["accuracy_dedup"] = round(comparison["deduplicated"][n]["accuracy_dedup"], 4)
    (METRICS_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    for row in comparison["mcnemar"]:
        print(f"McNemar {row['a']} vs {row['b']}: a_only={row['a_only']} b_only={row['b_only']} p={row['p_value']:.3g}")

    metrics_file.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nwrote {metrics_file}, results/metrics/*_validation.json and model_comparison.json")


if __name__ == "__main__":
    main()
