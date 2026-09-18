#!/usr/bin/env python3
"""
Supervised joint fine-tuning on the PlantDoc adapt split (all 4 focus classes
together, class-balanced sampling, heavy augmentation).

    python -m da.joint_training --seed 42
    python -m da.joint_training --init results/checkpoints/self_training_seed42.pth --name joint_from_st_seed42

Model selection uses the dev split only; the test split is touched once, at the end.
Writes results/checkpoints/<name>.pth and results/<name>.json.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

from da import common as C


def main():
    p = C.common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--init", default=None, help="starting checkpoint (default: PlantVillage source model)")
    p.add_argument("--name", default=None)
    p.add_argument("--augment", default="heavy", choices=["heavy", "light"])
    args = p.parse_args()
    device, splits = C.setup(args)
    name = args.name or f"joint_seed{args.seed}"

    model = C.load_checkpoint(Path(args.init), device) if args.init else C.load_source_model(device)
    baseline = C.evaluate_model(model, device, splits, args.batch_size, args.workers)
    print(f"start: {C.headline(baseline)}")

    loader = C.make_loader(
        splits["adapt"], C.heavy_augment() if args.augment == "heavy" else C.light_augment(), args.batch_size, balanced=True, workers=args.workers
    )
    dev_fn = C.dev_accuracy_fn(model, device, splits, args.batch_size, args.workers)

    best = {"dev_acc": -1.0, "epoch": 0, "state": copy.deepcopy(model.state_dict())}

    def on_epoch_end(epoch, loss, acc):
        row = dev_fn()
        if row["dev_acc"] > best["dev_acc"]:
            best.update(dev_acc=row["dev_acc"], epoch=epoch, state=copy.deepcopy(model.state_dict()))
        return row

    history = C.train_epochs(model, loader, device, args.epochs, args.lr, args.weight_decay, on_epoch_end, label_smoothing=0.05, log_prefix=f"[{name}] ")

    model.load_state_dict(best["state"])
    final = C.evaluate_model(model, device, splits, args.batch_size, args.workers)
    print(f"best epoch {best['epoch']} (dev {best['dev_acc']*100:.1f}%)  ->  final: {C.headline(final)}   test only: {C.headline(final, 'test')}")

    ckpt = C.CHECKPOINT_DIR / f"{name}.pth"
    C.save_checkpoint(model, ckpt)
    C.save_json(
        {
            "name": name,
            "method": "joint_training",
            "init": args.init or str(C.SOURCE_CHECKPOINT),
            "config": {k: v for k, v in vars(args).items() if k != "device"} | {"device": str(device)},
            "splits": C.describe_splits(splits),
            "best_epoch": best["epoch"],
            "history": history,
            "start": baseline,
            "final": final,
            "checkpoint": str(ckpt),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        C.RESULTS_DIR / f"{name}.json",
    )


if __name__ == "__main__":
    main()
