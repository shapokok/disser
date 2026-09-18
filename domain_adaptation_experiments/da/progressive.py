#!/usr/bin/env python3
"""
Progressive (curriculum) domain adaptation: fine-tune on the "easy" classes
first (Squash powdery mildew, Tomato late blight), then on the "hard" corn
classes, then jointly on all four with class balancing.

    python -m da.progressive --seed 42
    python -m da.progressive --init results/checkpoints/self_training_seed42.pth --name progressive_from_st_seed42
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

from da import common as C

STAGES = [
    ("stage1_easy", C.EASY_CLASSES, 8, 5e-5),
    ("stage2_hard", C.HARD_CLASSES, 10, 3e-5),
    ("stage3_joint", C.FOCUS_CLASSES, 12, 1e-5),
]


def main():
    p = C.common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--init", default=None)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    device, splits = C.setup(args)
    name = args.name or f"progressive_seed{args.seed}"

    model = C.load_checkpoint(Path(args.init), device) if args.init else C.load_source_model(device)
    start = C.evaluate_model(model, device, splits, args.batch_size, args.workers)
    print(f"start: {C.headline(start)}")
    dev_fn = C.dev_accuracy_fn(model, device, splits, args.batch_size, args.workers)

    stages_out = {}
    best = {"dev_acc": -1.0, "stage": None, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    for stage_name, classes, epochs, lr in STAGES:
        allowed = set(classes.values())
        items = [it for it in splits["adapt"] if it[1] in allowed]
        loader = C.make_loader(items, C.heavy_augment(), args.batch_size, balanced=True, workers=args.workers)
        print(f"\n== {stage_name}: {len(items)} images, classes {[C.short_name(c) for c in allowed]}")

        def on_epoch_end(epoch, loss, acc, _stage=stage_name):
            row = dev_fn()
            if row["dev_acc"] > best["dev_acc"]:
                best.update(dev_acc=row["dev_acc"], stage=_stage, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            return row

        history = C.train_epochs(model, loader, device, epochs, lr, on_epoch_end=on_epoch_end, label_smoothing=0.05, log_prefix=f"[{name} {stage_name}] ")
        stages_out[stage_name] = {"classes": sorted(allowed), "epochs": epochs, "lr": lr, "history": history, "eval": C.evaluate_model(model, device, splits, args.batch_size, args.workers)}
        print(f"   after {stage_name}: {C.headline(stages_out[stage_name]['eval'])}")
        C.save_checkpoint(model, C.CHECKPOINT_DIR / f"{name}_{stage_name}.pth")

    last = C.evaluate_model(model, device, splits, args.batch_size, args.workers)
    model.load_state_dict(best["state"])
    final = C.evaluate_model(model, device, splits, args.batch_size, args.workers)
    print(f"\nlast stage: {C.headline(last)}\nbest by dev ({best['stage']} epoch {best['epoch']}): {C.headline(final)}   test only: {C.headline(final, 'test')}")
    ckpt = C.CHECKPOINT_DIR / f"{name}.pth"
    C.save_checkpoint(model, ckpt)
    C.save_json(
        {
            "name": name,
            "method": "progressive",
            "init": args.init or str(C.SOURCE_CHECKPOINT),
            "config": {k: v for k, v in vars(args).items() if k != "device"} | {"device": str(device)},
            "splits": C.describe_splits(splits),
            "stages": stages_out,
            "best": {"stage": best["stage"], "epoch": best["epoch"], "dev_acc": best["dev_acc"]},
            "start": start,
            "last_stage": last,
            "final": final,
            "checkpoint": str(ckpt),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        C.RESULTS_DIR / f"{name}.json",
    )


if __name__ == "__main__":
    main()
