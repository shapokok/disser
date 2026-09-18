#!/usr/bin/env python3
"""
Thesis figures from results/summary.json and the per-run result files.

    python -m da.figures            # writes figures/figure1..4.pdf (+ .png)

figure1  training curves of joint fine-tuning (dev accuracy, train loss), mean over seeds
figure2  method comparison with 95 % confidence intervals
figure3  per-class accuracy, zero-shot vs adapted
figure4  confusion matrix of the exported field model on dev+test
"""

from __future__ import annotations

import numpy as np

from da import common as C

# Reference palette (validated): one hue for magnitude, gray for de-emphasis, slot-1 blue for the adapted model.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, GRAY = "#2a78d6", "#eb6834", "#c3c2b7"
SEQ = ["#f4f7fb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    return plt


def _save(fig, name):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(C.FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    print(f"saved figures/{name}.pdf")


def figure1(plt, summary):
    runs = []
    for seed in summary["seeds"]:
        path = C.RESULTS_DIR / f"joint_seed{seed}.json"
        if path.exists():
            runs.append(C.load_json(path))
    if not runs:
        return
    n = min(len(r["history"]) for r in runs)
    epochs = np.arange(1, n + 1)
    dev = np.array([[h["dev_acc"] for h in r["history"][:n]] for r in runs]) * 100
    loss = np.array([[h["train_loss"] for h in r["history"][:n]] for r in runs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a1.plot(epochs, loss.mean(0), color=S1, lw=2)
    a1.fill_between(epochs, loss.min(0), loss.max(0), color=S1, alpha=0.12, lw=0)
    a1.set(title="(a) Training loss (joint fine-tuning)", xlabel="Epoch", ylabel="Cross-entropy")
    a2.plot(epochs, dev.mean(0), color=S1, lw=2, label="dev accuracy")
    a2.fill_between(epochs, dev.min(0), dev.max(0), color=S1, alpha=0.12, lw=0)
    base = summary["methods"]["baseline"]["dev_open"]["accuracy_mean"] * 100
    a2.axhline(base, color=GRAY, lw=1.5, label="zero-shot")
    a2.text(n, base + 1, "zero-shot", ha="right", va="bottom", color=INK2, fontsize=8)
    a2.text(n, dev.mean(0)[-1] + 1, f"{dev.mean(0)[-1]:.1f} %", ha="right", va="bottom", color=INK2, fontsize=8)
    a2.set(title=f"(b) Dev accuracy, mean of {len(runs)} seeds", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
    fig.tight_layout()
    _save(fig, "figure1")


def figure2(plt, summary):
    rows = [(k, v) for k, v in summary["methods"].items()]
    labels = ["Zero-shot (source model)" if k == "baseline" else v["label"] for k, v in rows]
    acc = np.array([v["eval_open"]["accuracy_mean"] for _, v in rows]) * 100
    ci = np.array([v["eval_open"]["ci95_pooled"] for _, v in rows]) * 100
    colors = [GRAY if k == "baseline" else S1 for k, _ in rows]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, acc, color=colors, height=0.55)
    ax.errorbar(acc, y, xerr=[acc - ci[:, 0], ci[:, 1] - acc], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for yi, a, c in zip(y, acc, ci):
        ax.text(c[1] + 1.5, yi, f"{a:.1f} %", va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, labels)
    ax.set(xlim=(0, 100), xlabel="Accuracy on dev + test, open-set 38-way (%)")
    ax.grid(axis="y", visible=False)
    n = rows[0][1]["eval_open"]["n"]
    ax.set_title(f"Domain adaptation methods (PlantDoc, {n} held-out images, 95 % Wilson CI)", loc="left")
    fig.tight_layout()
    _save(fig, "figure2")


def figure3(plt, summary):
    pc = summary.get("per_class_eval_open") or {}
    if not pc:
        return
    names = list(pc)
    base = np.array([pc[n]["baseline"] for n in names]) * 100
    joint = np.array([pc[n]["joint"] for n in names]) * 100
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - w / 2 - 0.02, base, w, color=GRAY, label="Zero-shot")
    ax.bar(x + w / 2 + 0.02, joint, w, color=S1, label="Joint fine-tuning")
    for xi, b, j in zip(x, base, joint):
        ax.text(xi + w / 2 + 0.02, j + 1.5, f"{j:.0f}", ha="center", color=INK2, fontsize=8)
        ax.text(xi - w / 2 - 0.02, b + 1.5, f"{b:.0f}", ha="center", color=INK2, fontsize=8)
    ax.set_xticks(x, [f"{n}\n(n={pc[n]['support']})" for n in names])
    ax.set(ylim=(0, 100), ylabel="Accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Per-class accuracy on dev + test", loc="left")
    fig.tight_layout()
    _save(fig, "figure3")


def figure4(plt, summary):
    import json

    info_path = C.MODELS_DIR / "field_model_info.json"
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text())
    run = C.load_json(C.RESULTS_DIR / f"{info['method_key']}_seed{info['seed']}.json")
    m = run["final"]["eval"]["open"]
    cm = np.array(m["confusion_matrix"], dtype=float)
    labels = m["confusion_labels"]
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels) - 1), labels[:-1])
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]*100:.0f} %", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.55 else INK)
    ax.set(xlabel="Predicted", ylabel="True")
    ax.set_title(f"{info['method']} (seed {info['seed']}), dev + test, acc {m['accuracy']*100:.1f} %", loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Share of the true class")
    fig.tight_layout()
    _save(fig, "figure4")


def main():
    plt = _style()
    summary_path = C.RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        raise SystemExit("results/summary.json not found - run `python -m da.run_all` first")
    summary = C.load_json(summary_path)
    figure1(plt, summary)
    figure2(plt, summary)
    figure3(plt, summary)
    figure4(plt, summary)


if __name__ == "__main__":
    main()
