#!/usr/bin/env python3
"""
Test-time augmentation: average the softmax over 10 deterministic views
(identity, flips, ±10° rotations, brightness/contrast, 0.9x / 1.1x scale).

    python -m da.tta --checkpoint results/checkpoints/joint_seed42.pth --name tta_joint_seed42
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

from da import common as C


class _View:
    """Picklable deterministic view (DataLoader workers use spawn on macOS, so no lambdas)."""

    def __init__(self, kind: str, value: float = 0.0):
        self.kind, self.value = kind, value

    def __call__(self, im):
        if self.kind == "hflip":
            return TF.hflip(im)
        if self.kind == "vflip":
            return TF.vflip(im)
        if self.kind == "hvflip":
            return TF.vflip(TF.hflip(im))
        if self.kind == "rotate":
            return TF.rotate(im, self.value)
        if self.kind == "brightness":
            return TF.adjust_brightness(im, self.value)
        if self.kind == "contrast":
            return TF.adjust_contrast(im, self.value)
        if self.kind == "scale":
            return TF.affine(im, angle=0, translate=[0, 0], scale=self.value, shear=[0.0])
        return im


def tta_views():
    base = [transforms.Resize((C.IMG_SIZE, C.IMG_SIZE))]
    tail = [transforms.ToTensor(), transforms.Normalize(C.MEAN, C.STD)]
    ops = {
        "identity": _View("identity"),
        "hflip": _View("hflip"),
        "vflip": _View("vflip"),
        "hvflip": _View("hvflip"),
        "rot+10": _View("rotate", 10),
        "rot-10": _View("rotate", -10),
        "bright": _View("brightness", 1.2),
        "contrast": _View("contrast", 1.2),
        "scale0.9": _View("scale", 0.9),
        "scale1.1": _View("scale", 1.1),
    }
    return {k: transforms.Compose([*base, v, *tail]) for k, v in ops.items()}


@torch.no_grad()
def predict_tta(model, items, device, batch_size, workers):
    views = tta_views()
    acc = None
    labels = None
    for tf in views.values():
        loader = C.make_loader(items, tf, batch_size=batch_size, workers=workers)
        probs, labels = C.predict(model, loader, device)
        acc = probs if acc is None else acc + probs
    return acc / len(views), labels


def main():
    p = C.common_args(
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    device, splits = C.setup(args)
    name = args.name or f"tta_{Path(args.checkpoint).stem}"
    model = C.load_checkpoint(Path(args.checkpoint), device)

    standard = C.evaluate_model(model, device, splits, args.batch_size, args.workers)
    out = {}
    cache = {}
    for split in ("dev", "test"):
        cache[split] = predict_tta(model, splits[split], device, args.batch_size, args.workers)
    cache["eval"] = (
        np.concatenate([cache["dev"][0], cache["test"][0]]),
        np.concatenate([cache["dev"][1], cache["test"][1]]),
    )
    for split, (probs, labels) in cache.items():
        out[split] = {
            "open": C.compute_metrics(labels, probs.argmax(1)),
            "restricted": C.compute_metrics(labels, C.restricted_argmax(probs, C.FOCUS_IDX)),
        }
    print(f"standard: {C.headline(standard)}\nTTA     : {C.headline(out)}   test only: {C.headline(out, 'test')}")
    C.save_json(
        {
            "name": name,
            "method": "tta",
            "checkpoint": args.checkpoint,
            "num_views": len(tta_views()),
            "views": list(tta_views()),
            "splits": C.describe_splits(splits),
            "standard": standard,
            "final": out,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        C.RESULTS_DIR / f"{name}.json",
    )


if __name__ == "__main__":
    main()
