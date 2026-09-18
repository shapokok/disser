"""
Real validation reports.

`scripts/evaluate_models.py` evaluates every model on the PlantVillage
validation split and writes results/metrics/<model>_validation.json.
This module only reads those files; nothing is simulated.
"""

from __future__ import annotations

import json
from functools import lru_cache

import config
from labels import format_class


def _read_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_validation(model_name: str):
    return _read_json(config.METRICS_DIR / f"{model_name}_validation.json")


def load_training_history(model_name: str):
    data = _read_json(config.METRICS_DIR / f"{model_name}_metrics.json")
    if not data:
        return None
    keys = ("train_losses", "val_losses", "train_accuracies", "val_accuracies", "epoch_times")
    history = {k: data.get(k, []) for k in keys}
    history.update(
        {
            "epochs": len(history["train_losses"]),
            "best_val_accuracy": data.get("best_val_accuracy"),
            "total_time_minutes": data.get("total_time_minutes"),
            "batch_size": data.get("batch_size"),
            "learning_rate": data.get("learning_rate"),
            "optimizer": data.get("optimizer"),
            "device": data.get("device"),
            "trained_at": data.get("trained_at"),
        }
    )
    return history


def build_report(model_name: str, lang: str = "en"):
    """Report for the statistics page: per-class metrics, confusion matrix, confused pairs (cached per file version)."""
    path = config.METRICS_DIR / f"{model_name}_validation.json"
    mtime = path.stat().st_mtime if path.exists() else None
    return _build_report(model_name, lang, mtime)


@lru_cache(maxsize=32)
def _build_report(model_name: str, lang: str, _mtime):
    data = load_validation(model_name)
    if not data:
        return None

    per_class = []
    for row in data["per_class"]:
        item = dict(row)
        item["label"] = format_class(row["class_name"], lang)
        per_class.append(item)

    pairs = []
    for p in data.get("top_confused_pairs", []):
        pairs.append(
            {
                **p,
                "true_label": format_class(p["true_class"], lang),
                "predicted_label": format_class(p["predicted_class"], lang),
            }
        )

    by_f1 = sorted(per_class, key=lambda r: r["f1_score"])
    return {
        "model_name": model_name,
        "evaluated_at": data.get("evaluated_at"),
        "num_samples": data.get("num_samples"),
        "overall": data["overall"],
        "per_class": per_class,
        "class_names": [r["class_name"] for r in per_class],
        "labels": [r["label"] for r in per_class],
        "confusion_matrix": data["confusion_matrix"],
        "top_confused_pairs": pairs,
        "best_classes": list(reversed(by_f1[-5:])),
        "worst_classes": by_f1[:5],
        "inference_time_ms": data.get("inference_time_ms"),
        "calibration": data.get("calibration"),
    }


def clear_cache():
    _build_report.cache_clear()
