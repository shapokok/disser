#!/usr/bin/env python3
"""
Self-training (pseudo-label) domain adaptation: unsupervised on the target.

    python -m da.self_training --seed 42

Each iteration: predict the adapt split (labels are NOT used), keep predictions
above a confidence threshold that decreases from 0.85 to 0.65, fine-tune on
those pseudo-labels, repeat. Pseudo-labels are restricted to the 4 focus
classes (closed-set adaptation). No model selection on labelled data: the
last iteration is the final model; dev accuracy is logged for transparency.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from da import common as C


def main():
    p = C.common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--epochs-per-iteration", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--conf-start", type=float, default=0.85)
    p.add_argument("--conf-end", type=float, default=0.65)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    device, splits = C.setup(args)
    name = args.name or f"self_training_seed{args.seed}"

    model = C.load_source_model(device)
    start = C.evaluate_model(model, device, splits, args.batch_size, args.workers)
    print(f"start: {C.headline(start)}")

    unlabeled_loader = C.make_loader(splits["adapt"], C.eval_transform(), args.batch_size, workers=args.workers)
    dev_fn = C.dev_accuracy_fn(model, device, splits, args.batch_size, args.workers)
    iterations = []

    for it in range(1, args.iterations + 1):
        progress = (it - 1) / max(1, args.iterations - 1)
        threshold = args.conf_start - (args.conf_start - args.conf_end) * progress

        probs, true_labels = C.predict(model, unlabeled_loader, device)
        focus_probs = probs[:, C.FOCUS_IDX]
        conf = focus_probs.max(1)  # raw 38-way probability of the best focus class: only truly confident samples pass
        pseudo = np.array(C.FOCUS_IDX)[focus_probs.argmax(1)]
        keep = conf >= threshold
        pseudo_labels = {splits["adapt"][i][0]: int(pseudo[i]) for i in np.where(keep)[0]}
        pseudo_acc = float((pseudo[keep] == true_labels[keep]).mean()) if keep.any() else 0.0  # diagnostic only
        print(f"[{name}] iteration {it}: threshold {threshold:.2f}  pseudo-labels {keep.sum()}/{len(keep)}  (pseudo-label accuracy {pseudo_acc*100:.1f}%, diagnostic)")
        if keep.sum() < args.batch_size:
            print("   too few confident samples, stopping")
            break

        items = [splits["adapt"][i] for i in np.where(keep)[0]]
        loader = C.make_loader(items, C.light_augment(), args.batch_size, balanced=True, workers=args.workers, pseudo_labels=pseudo_labels)
        history = C.train_epochs(model, loader, device, args.epochs_per_iteration, args.lr, on_epoch_end=dev_fn, log_prefix=f"[{name} it{it}] ")
        iterations.append(
            {"iteration": it, "threshold": threshold, "num_pseudo": int(keep.sum()), "pseudo_label_accuracy": pseudo_acc, "history": history}
        )

    final = C.evaluate_model(model, device, splits, args.batch_size, args.workers)
    print(f"final: {C.headline(final)}   test only: {C.headline(final, 'test')}")
    ckpt = C.CHECKPOINT_DIR / f"{name}.pth"
    C.save_checkpoint(model, ckpt)
    C.save_json(
        {
            "name": name,
            "method": "self_training",
            "config": {k: v for k, v in vars(args).items() if k != "device"} | {"device": str(device)},
            "splits": C.describe_splits(splits),
            "iterations": iterations,
            "start": start,
            "final": final,
            "checkpoint": str(ckpt),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        C.RESULTS_DIR / f"{name}.json",
    )


if __name__ == "__main__":
    main()
