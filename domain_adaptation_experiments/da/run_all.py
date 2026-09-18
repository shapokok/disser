#!/usr/bin/env python3
"""
Run the whole domain adaptation study and summarise it.

    python -m da.run_all --seeds 42 43 44 --device cpu --threads 4
    python -m da.run_all --skip-existing          # reuse finished result files

Steps
  1. zero-shot baseline (focus classes + all 27 overlapping classes)
  2. per seed: self-training, joint training (from the source model and from
     the self-trained model), progressive adaptation
  3. TTA on the joint model of every seed
  4. results/summary.json + summary.md (mean +/- std over seeds, 95 % CI)
  5. best model -> models/field_mobilenet_da.pth (+ field_model_info.json) for the web app
  6. figures/figure1-4.pdf
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime

from da import common as C

METHODS = {
    "self_training": "Self-training (pseudo-labels)",
    "joint": "Joint fine-tuning",
    "joint_from_st": "Self-training + joint fine-tuning",
    "progressive": "Progressive (easy -> hard -> joint)",
    "tta_joint": "Joint fine-tuning + TTA",
}


def run(module: str, *extra: str, skip_if=None, **kw):
    if skip_if and skip_if.exists():
        print(f"-- skip {module} {' '.join(extra)} ({skip_if.name} exists)")
        return
    cmd = [sys.executable, "-m", module, *extra]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=C.DA_DIR, check=True)


def collect(name: str) -> dict | None:
    path = C.RESULTS_DIR / f"{name}.json"
    return C.load_json(path) if path.exists() else None


def summarise(seeds: list[int]) -> dict:
    baseline = collect("eval_baseline_source")
    rows = {}

    def add(method, per_seed):
        vals = {}
        for split in ("dev", "test", "eval"):
            for mode in ("open", "restricted"):
                accs = [r[split][mode]["accuracy"] for r in per_seed]
                f1s = [r[split][mode]["macro_f1"] for r in per_seed]
                pooled_correct = sum(r[split][mode]["correct"] for r in per_seed)
                pooled_n = sum(r[split][mode]["n"] for r in per_seed)
                vals[f"{split}_{mode}"] = {
                    "accuracy_mean": statistics.mean(accs),
                    "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                    "macro_f1_mean": statistics.mean(f1s),
                    "macro_f1_std": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                    "ci95_pooled": list(C.wilson_ci(pooled_correct, pooled_n)),
                    "n": per_seed[0][split][mode]["n"],
                    "runs": len(per_seed),
                }
        rows[method] = {"label": METHODS.get(method, method), **vals}

    if baseline:
        add("baseline", [{k: baseline[k] for k in ("dev", "test", "eval")}])
    for method in ("self_training", "joint", "joint_from_st", "progressive", "tta_joint"):
        per_seed = [r["final"] for s in seeds if (r := collect(f"{method}_seed{s}"))]
        if per_seed:
            add(method, per_seed)

    # Per-class comparison for the headline method (joint) on the eval split, averaged over seeds
    per_class = {}
    joint_runs = [r for s in seeds if (r := collect(f"joint_seed{s}"))]
    if baseline and joint_runs:
        for c in C.FOCUS_IDX:
            per_class[C.short_name(c)] = {
                "baseline": baseline["eval"]["open"]["per_class"][str(c)]["accuracy"],
                "joint": statistics.mean(
                    r["final"]["eval"]["open"]["per_class"][str(c)]["accuracy"] for r in joint_runs
                ),
                "support": baseline["eval"]["open"]["per_class"][str(c)]["support"],
            }

    all27 = collect("eval_baseline_source_all27")
    zero_shot = {}
    for arch, focus_name, all_name in (
        ("baseline", "eval_baseline_baseline", "eval_baseline_baseline_all27"),
        ("efficientnet", "eval_baseline_efficientnet", "eval_baseline_efficientnet_all27"),
        ("mobilenet", "eval_baseline_source", "eval_baseline_source_all27"),
        ("hybrid", "eval_baseline_hybrid", "eval_baseline_hybrid_all27"),
    ):
        f, a = collect(focus_name), collect(all_name)
        if f:
            zero_shot[arch] = {
                "focus_eval_open": f["eval"]["open"]["accuracy"],
                "focus_eval_restricted": f["eval"]["restricted"]["accuracy"],
                "all27_test_open": a["test"]["open"]["accuracy"] if a else None,
                "all27_top5": a["test"]["top5_open"] if a else None,
            }
    return {
        "zero_shot_by_arch": zero_shot,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "protocol": {
            "source": "MobileNet-V2 trained on PlantVillage (38 classes)",
            "target": "PlantDoc, 4 overlapping classes with enough images",
            "splits": baseline["splits"] if baseline else None,
            "headline_metric": "open-set (38-way) accuracy on dev+test ('eval'); test split reported separately",
        },
        "methods": rows,
        "per_class_eval_open": per_class,
        "baseline_all27_test": all27["test"] if all27 else None,
    }


def write_markdown(summary: dict):
    lines = [
        "# Domain adaptation results (PlantVillage -> PlantDoc)",
        "",
        f"Generated {summary['generated_at']}, seeds {summary['seeds']}.",
        "",
    ]
    splits = summary["protocol"]["splits"] or {}
    if splits:
        lines += ["Splits: " + ", ".join(f"{k} = {v['n']}" for k, v in splits.items()), ""]
    lines += [
        "| Method | eval acc (open) | 95 % CI | macro-F1 | test acc (open) | eval acc (restricted) |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["methods"].values():
        e, t, rr = r["eval_open"], r["test_open"], r["eval_restricted"]
        pm = f" ± {e['accuracy_std'] * 100:.1f}" if e["runs"] > 1 else ""
        lines.append(
            f"| {r['label']} | {e['accuracy_mean'] * 100:.1f}{pm} | {e['ci95_pooled'][0] * 100:.0f}–{e['ci95_pooled'][1] * 100:.0f} | "
            f"{e['macro_f1_mean'] * 100:.1f} | {t['accuracy_mean'] * 100:.1f} | {rr['accuracy_mean'] * 100:.1f} |"
        )
    if summary["per_class_eval_open"]:
        lines += ["", "| Class | n | zero-shot | joint fine-tuning |", "|---|---|---|---|"]
        for cls, v in summary["per_class_eval_open"].items():
            lines.append(f"| {cls} | {v['support']} | {v['baseline'] * 100:.1f} | {v['joint'] * 100:.1f} |")
    if summary["baseline_all27_test"]:
        a = summary["baseline_all27_test"]
        lines += [
            "",
            f"Zero-shot on all 27 overlapping classes (PlantDoc test, n={a['open']['n']}): open {a['open']['accuracy'] * 100:.1f} %, restricted {a['restricted']['accuracy'] * 100:.1f} %, top-5 {a['top5_open'] * 100:.1f} %.",
        ]
    path = C.RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {path.relative_to(C.PROJECT_ROOT)}")


def export_best(seeds: list[int], summary: dict):
    """Pick the best adapted checkpoint (by dev accuracy, never test) and copy it for the web app."""
    candidates = []
    for method in ("joint", "joint_from_st", "progressive"):
        for s in seeds:
            r = collect(f"{method}_seed{s}")
            if r:
                candidates.append((r["final"]["dev"]["open"]["accuracy"], method, s, r))
    if not candidates:
        return
    dev_acc, method, seed, r = max(candidates, key=lambda c: c[0])
    src = C.PROJECT_ROOT / r["checkpoint"] if not r["checkpoint"].startswith("/") else r["checkpoint"]
    dst = C.MODELS_DIR / "field_mobilenet_da.pth"
    shutil.copyfile(src, dst)
    info = {
        "method": METHODS[method],
        "method_key": method,
        "seed": seed,
        "source_checkpoint": "mobilenet_model.pth",
        "covered_class_indices": C.FOCUS_IDX,
        "covered_classes": [C.CLASS_NAMES[i] for i in C.FOCUS_IDX],
        "accuracy_dev": dev_acc,
        "accuracy_test": r["final"]["test"]["open"]["accuracy"],
        "accuracy_eval": r["final"]["eval"]["open"]["accuracy"],
        "macro_f1_eval": r["final"]["eval"]["open"]["macro_f1"],
        "eval_n": r["final"]["eval"]["open"]["n"],
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(C.MODELS_DIR / "field_model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(
        f"exported {method} seed {seed} (dev {dev_acc * 100:.1f}%, test {info['accuracy_test'] * 100:.1f}%) -> models/field_mobilenet_da.pth"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=25, help="epochs for joint training")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=["baseline", "self_training", "joint", "joint_from_st", "progressive", "tta", "summary", "figures"],
    )
    args = p.parse_args()
    common = ["--device", args.device, "--workers", str(args.workers)] + (
        ["--threads", str(args.threads)] if args.threads else []
    )
    want = lambda step: args.only is None or step in args.only
    skip = (lambda name: C.RESULTS_DIR / f"{name}.json") if args.skip_existing else (lambda name: None)

    if want("baseline"):
        run("da.evaluate", *common, skip_if=skip("eval_baseline_source"))
        run("da.evaluate", "--classes", "all", *common, skip_if=skip("eval_baseline_source_all27"))
        # zero-shot generalisation of the other PlantVillage architectures (needs their weights in models/)
        for arch in ("baseline", "efficientnet", "hybrid"):
            if (C.MODELS_DIR / f"{arch}_model.pth").exists():
                run("da.evaluate", "--arch", arch, *common, skip_if=skip(f"baseline_{arch}"))
                run("da.evaluate", "--arch", arch, "--classes", "all", *common, skip_if=skip(f"baseline_{arch}_all27"))
    for seed in args.seeds:
        s = ["--seed", str(seed)]
        if want("self_training"):
            run("da.self_training", *s, *common, skip_if=skip(f"self_training_seed{seed}"))
        if want("joint"):
            run("da.joint_training", *s, "--epochs", str(args.epochs), *common, skip_if=skip(f"joint_seed{seed}"))
        if want("joint_from_st"):
            run(
                "da.joint_training",
                *s,
                "--epochs",
                str(args.epochs),
                "--lr",
                "5e-5",
                "--init",
                str(C.CHECKPOINT_DIR / f"self_training_seed{seed}.pth"),
                "--name",
                f"joint_from_st_seed{seed}",
                *common,
                skip_if=skip(f"joint_from_st_seed{seed}"),
            )
        if want("progressive"):
            run("da.progressive", *s, *common, skip_if=skip(f"progressive_seed{seed}"))
        if want("tta"):
            run(
                "da.tta",
                "--checkpoint",
                str(C.CHECKPOINT_DIR / f"joint_seed{seed}.pth"),
                "--name",
                f"tta_joint_seed{seed}",
                *s,
                *common,
                skip_if=skip(f"tta_joint_seed{seed}"),
            )

    if want("summary"):
        summary = summarise(args.seeds)
        C.save_json(summary, C.RESULTS_DIR / "summary.json")
        write_markdown(summary)
        export_best(args.seeds, summary)
    if want("figures"):
        run("da.figures")


if __name__ == "__main__":
    main()
