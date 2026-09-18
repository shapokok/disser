#!/usr/bin/env python3
"""
Aggregate the cross-validation results (results/cv/*_seed*.json) into the thesis tables.

    python -m da.cv_summary        # -> results/cv_summary.json, results/cv_summary.md

* mean ± std over seeds of the out-of-fold accuracy (38-way "open" and 4-way "restricted"), macro-F1
* 95 % Wilson interval for n = 500 at the mean accuracy
* "ensemble": average of the TTA outputs of every supervised fine-tuning variant (defined a priori,
  not selected by results), computed from the stored out-of-fold probabilities
* paired McNemar tests on the same 500 images for the key comparisons (seed-wise, first seed shown)
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime

import numpy as np

from da import common as C
from da.cv import CV_DIR, FOCUS, LABELS, UNSUPERVISED

ORDER = [
    "zero_shot",
    "adabn",
    "tent",
    "self_training",
    "self_training_v2",
    "sup25",
    "semi25",
    "joint",
    "joint_tta",
    "joint_long",
    "joint_long_tta",
    "joint_closed",
    "joint_closed_tta",
    "joint_320",
    "joint_320_tta",
    "joint_effnet",
    "joint_effnet_tta",
    "joint_v2_noadabn",
    "joint_v2_noadabn_tta",
    "joint_v2",
    "joint_v2_tta",
    "ensemble",
]
ENSEMBLE_MEMBERS = [
    "joint_tta",
    "joint_long_tta",
    "joint_closed_tta",
    "joint_320_tta",
    "joint_effnet_tta",
    "joint_v2_noadabn_tta",
    "joint_v2_tta",
]
LABELS = {**LABELS, "ensemble": "Ансамбль дообученных моделей + TTA"}
KEY_PAIRS = [
    ("zero_shot", "joint"),
    ("zero_shot", "self_training_v2"),
    ("self_training", "self_training_v2"),
    ("sup25", "semi25"),
    ("joint", "joint_tta"),
    ("joint", "joint_v2"),
    ("joint", "ensemble"),
]


def load_runs() -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for p in sorted(CV_DIR.glob("*_seed*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        runs[d["method"]][int(d["seed"])] = d
    return runs


def build_ensemble(runs):
    seeds = set.intersection(*[set(runs[m]) for m in ENSEMBLE_MEMBERS if m in runs]) if runs else set()
    members = [m for m in ENSEMBLE_MEMBERS if m in runs]
    for seed in sorted(seeds):
        docs = [runs[m][seed] for m in members if "probs" in runs[m][seed]["oof"]]
        if len(docs) < 2:
            continue
        paths = docs[0]["oof"]["paths"]
        if any(d["oof"]["paths"] != paths for d in docs):
            continue
        probs = np.mean([np.array(d["oof"]["probs"]) for d in docs], axis=0)
        labels = np.array(docs[0]["oof"]["labels"])
        runs["ensemble"][seed] = {
            "method": "ensemble",
            "seed": seed,
            "members": [d["method"] for d in docs],
            "open": C.compute_metrics(labels, probs.argmax(1), FOCUS),
            "restricted": C.compute_metrics(labels, C.restricted_argmax(probs, FOCUS), FOCUS),
            "oof": {"paths": paths, "labels": labels.tolist(), "pred_open": probs.argmax(1).tolist()},
            "protocol": docs[0]["protocol"],
        }


def mcnemar(a_pred, b_pred, labels):
    from scipy.stats import chi2

    a, b = np.array(a_pred) == labels, np.array(b_pred) == labels
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return {"a_only": n01, "b_only": n10, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"a_only": n01, "b_only": n10, "p_value": float(chi2.sf(stat, 1))}


def summarise() -> dict:
    runs = load_runs()
    build_ensemble(runs)
    methods = {}
    for m in ORDER + sorted(set(runs) - set(ORDER)):
        if m not in runs:
            continue
        per_seed = list(runs[m].values())
        n = per_seed[0]["protocol"]["n"]
        row = {"label": LABELS.get(m, m), "unsupervised": m in UNSUPERVISED, "seeds": sorted(runs[m]), "n": n}
        for mode in ("open", "restricted"):
            accs = [r[mode]["accuracy"] for r in per_seed]
            f1s = [r[mode]["macro_f1"] for r in per_seed]
            mean = statistics.mean(accs)
            row[mode] = {
                "accuracy_mean": mean,
                "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "macro_f1_mean": statistics.mean(f1s),
                "ci95": list(C.wilson_ci(round(mean * n), n)),
                "per_class": {
                    c: statistics.mean(r[mode]["per_class"][c]["accuracy"] for r in per_seed)
                    for c in per_seed[0][mode]["per_class"]
                },
            }
        if m == "ensemble":
            row["members"] = per_seed[0]["members"]
        methods[m] = row

    tests = []
    for a, b in KEY_PAIRS:
        if a in runs and b in runs:
            common = sorted(set(runs[a]) & set(runs[b]))
            if not common:
                continue
            seed = common[0]
            ra, rb = runs[a][seed], runs[b][seed]
            if ra["oof"]["paths"] != rb["oof"]["paths"]:
                continue
            labels = np.array(ra["oof"]["labels"])
            tests.append(
                {"a": a, "b": b, "seed": seed, **mcnemar(ra["oof"]["pred_open"], rb["oof"]["pred_open"], labels)}
            )

    sup = {k: v for k, v in methods.items() if not v["unsupervised"] and k not in ("sup25", "semi25")}
    best = max(sup, key=lambda k: sup[k]["open"]["accuracy_mean"]) if sup else None
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "type": "stratified 5-fold cross-validation",
            "pool": "PlantDoc train + test, 4 focus classes",
            "n": next(iter(methods.values()))["n"] if methods else 0,
            "inner_dev": 0.15,
            "note": "each image is predicted once as held-out; inner dev (15 % of the training folds) is used only for early stopping",
        },
        "class_names": {str(c): C.short_name(c) for c in FOCUS},
        "methods": methods,
        "mcnemar": tests,
        "best_supervised": best,
        "ensemble_members": ENSEMBLE_MEMBERS,
    }


def write_markdown(s: dict):
    lines = [
        "# Доменная адаптация: 5-кратная кросс-валидация",
        "",
        f"Сгенерировано {s['generated_at']}. Пул: {s['protocol']['pool']}, n = {s['protocol']['n']}; "
        "каждое изображение оценено ровно один раз как отложенное.",
        "",
        "| Метод | Разметка поля | Точность, 38 кл. | 95 % ДИ | Точность, 4 кл. | Macro-F1 | Сиды |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, r in s["methods"].items():
        o, rr = r["open"], r["restricted"]
        pm = f" ± {o['accuracy_std'] * 100:.1f}" if len(r["seeds"]) > 1 else ""
        pm4 = f" ± {rr['accuracy_std'] * 100:.1f}" if len(r["seeds"]) > 1 else ""
        lab = "нет" if r["unsupervised"] else ("25 %" if m in ("sup25", "semi25") else "да")
        lines.append(
            f"| {r['label']} | {lab} | {o['accuracy_mean'] * 100:.1f}{pm} | {o['ci95'][0] * 100:.0f}–{o['ci95'][1] * 100:.0f} | "
            f"{rr['accuracy_mean'] * 100:.1f}{pm4} | {o['macro_f1_mean'] * 100:.1f} | {len(r['seeds'])} |"
        )
    if s["mcnemar"]:
        lines += [
            "",
            "Тест МакНемара (38 классов, одни и те же 500 изображений):",
            "",
            "| A | B | A верно, B нет | B верно, A нет | p |",
            "|---|---|---|---|---|",
        ]
        for t in s["mcnemar"]:
            p = "< 0.001" if t["p_value"] < 0.001 else f"{t['p_value']:.3f}"
            lines.append(
                f"| {LABELS.get(t['a'], t['a'])} | {LABELS.get(t['b'], t['b'])} | {t['a_only']} | {t['b_only']} | {p} |"
            )
    path = C.RESULTS_DIR / "cv_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {path}")


def main():
    s = summarise()
    C.save_json(s, C.RESULTS_DIR / "cv_summary.json")
    write_markdown(s)
    for r in s["methods"].values():
        print(
            f"{r['label'][:44]:44s} open {r['open']['accuracy_mean'] * 100:5.1f}  restricted {r['restricted']['accuracy_mean'] * 100:5.1f}  seeds {r['seeds']}"
        )


if __name__ == "__main__":
    main()
