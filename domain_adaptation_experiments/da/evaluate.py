#!/usr/bin/env python3
"""
Evaluate a checkpoint on the PlantDoc dev / test splits.

    python -m da.evaluate                                   # zero-shot source model (PlantVillage MobileNet-V2)
    python -m da.evaluate --checkpoint results/checkpoints/joint_seed42.pth --name joint_seed42
    python -m da.evaluate --classes all                     # zero-shot on all 27 overlapping classes (test split)

Writes results/eval_<name>.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from da import common as C


def evaluate_all_classes(model, device, batch_size, workers) -> dict:
    """Zero-shot accuracy on every overlapping class, official PlantDoc test split (236 images)."""
    items = C.list_images("test", C.PLANTDOC_TO_PV)
    loader = C.make_loader(items, C.eval_transform(), batch_size=batch_size, workers=workers)
    probs, labels = C.predict(model, loader, device)
    classes = sorted(set(C.PLANTDOC_TO_PV.values()))
    return {
        "open": C.compute_metrics(labels, probs.argmax(1), classes),
        "restricted": C.compute_metrics(labels, C.restricted_argmax(probs, classes), classes),
        "top5_open": float(np.mean([labels[i] in np.argsort(probs[i])[::-1][:5] for i in range(len(labels))])),
    }


def main():
    p = C.common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--checkpoint", default=None, help="state dict to evaluate (default: PlantVillage MobileNet-V2)")
    p.add_argument("--name", default=None)
    p.add_argument("--classes", default="focus", choices=["focus", "all"])
    args = p.parse_args()
    device, splits = C.setup(args)

    ckpt = Path(args.checkpoint) if args.checkpoint else C.SOURCE_CHECKPOINT
    name = args.name or (ckpt.stem if args.checkpoint else "baseline_source")
    model = C.load_checkpoint(ckpt, device)

    if args.classes == "all":
        result = {"name": name, "checkpoint": str(ckpt), "classes": "all_27_overlapping", "test": evaluate_all_classes(model, device, args.batch_size, args.workers)}
        m = result["test"]
        print(f"{name} on all 27 classes / test: open {m['open']['accuracy']*100:.1f}%  restricted {m['restricted']['accuracy']*100:.1f}%  top-5 {m['top5_open']*100:.1f}%")
        C.save_json(result, C.RESULTS_DIR / f"eval_{name}_all27.json")
        return

    result = {"name": name, "checkpoint": str(ckpt), "classes": "focus4", "splits": C.describe_splits(splits), **C.evaluate_model(model, device, splits, args.batch_size, args.workers)}
    for split in ("dev", "test", "eval"):
        print(f"{name:24s} {split:5s} open {C.headline(result, split, 'open')}   | restricted {result[split]['restricted']['accuracy']*100:.1f}%")
    C.save_json(result, C.RESULTS_DIR / f"eval_{name}.json")


if __name__ == "__main__":
    main()
