#!/usr/bin/env python3
"""
Thesis figures from results/cv_summary.json (cross-validation) and the per-run result files.

    python -m da.figures            # writes figures/figure1..4.pdf (+ .png)

figure1  training curves of joint fine-tuning on the fixed split (dev accuracy, train loss)
figure2  cross-validated method comparison with 95 % confidence intervals, coloured by label usage
figure3  per-class accuracy, zero-shot vs the best supervised method (cross-validation)
figure4  confusion matrix of the best method (out-of-fold predictions)
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


S3, S2_ORANGE = "#1baf7a", "#eb6834"
GROUP_COLOR = {"none0": GRAY, "none": S3, "partial": S2_ORANGE, "full": S1}
GROUP_LABEL = {"none0": "без адаптации", "none": "без разметки поля", "partial": "25 % разметки", "full": "с разметкой"}


def _group(key, m):
    if m["unsupervised"]:
        return "none0" if key == "zero_shot" else "none"
    return "partial" if key in ("sup25", "semi25") else "full"


def figure1(plt, cv):
    """Training curves of the joint fine-tuning run on the fixed split (illustrates the dynamics)."""
    runs = [C.load_json(p) for p in sorted(C.RESULTS_DIR.glob("joint_seed*.json"))]
    runs = [r for r in runs if r.get("history")]
    if not runs:
        return
    n = min(len(r["history"]) for r in runs)
    epochs = np.arange(1, n + 1)
    dev = np.array([[h["dev_acc"] for h in r["history"][:n]] for r in runs]) * 100
    loss = np.array([[h["train_loss"] for h in r["history"][:n]] for r in runs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a1.plot(epochs, loss.mean(0), color=S1, lw=2)
    if len(runs) > 1:
        a1.fill_between(epochs, loss.min(0), loss.max(0), color=S1, alpha=0.12, lw=0)
    a1.set(title="(a) Функция потерь (joint fine-tuning)", xlabel="Эпоха", ylabel="Кросс-энтропия")
    a2.plot(epochs, dev.mean(0), color=S1, lw=2)
    if len(runs) > 1:
        a2.fill_between(epochs, dev.min(0), dev.max(0), color=S1, alpha=0.12, lw=0)
    zs = cv["methods"].get("zero_shot") if cv else None
    if zs:
        base = zs["open"]["accuracy_mean"] * 100
        a2.axhline(base, color=GRAY, lw=1.5)
        a2.text(n, base + 1, "без адаптации", ha="right", va="bottom", color=INK2, fontsize=8)
    a2.set(title="(b) Точность на dev", xlabel="Эпоха", ylabel="Точность, %", ylim=(0, 100))
    fig.tight_layout()
    _save(fig, "figure1")


def figure2(plt, cv):
    rows = list(cv["methods"].items())
    labels = [m["label"] for _, m in rows]
    acc = np.array([m["open"]["accuracy_mean"] for _, m in rows]) * 100
    ci = np.array([m["open"]["ci95"] for _, m in rows]) * 100
    groups = [_group(k, m) for k, m in rows]
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, acc, color=[GROUP_COLOR[g] for g in groups], height=0.6)
    ax.errorbar(acc, y, xerr=[acc - ci[:, 0], ci[:, 1] - acc], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
    for yi, a, c in zip(y, acc, ci):
        ax.text(c[1] + 1.2, yi, f"{a:.1f}", va="center", color=INK2, fontsize=7.5)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set(xlim=(0, 100), xlabel="Точность по 38 классам на отложенных фолдах, %")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    seen = [g for g in GROUP_LABEL if g in groups]
    ax.legend(
        handles=[Patch(color=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in seen], loc="lower right", fontsize=7.5
    )
    ax.set_title(
        f"Методы доменной адаптации: 5-кратная кросс-валидация, n = {cv['protocol']['n']}, 95 % ДИ", loc="left"
    )
    fig.tight_layout()
    _save(fig, "figure2")


def figure3(plt, cv):
    best = cv.get("best_supervised")
    zs = cv["methods"].get("zero_shot")
    if not best or not zs:
        return
    bm = cv["methods"][best]
    keys = list(zs["open"]["per_class"])
    names = [cv["class_names"].get(k, k) for k in keys]
    base = np.array([zs["open"]["per_class"][k] for k in keys]) * 100
    adapted = np.array([bm["open"]["per_class"][k] for k in keys]) * 100
    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - w / 2 - 0.02, base, w, color=GRAY, label="Без адаптации")
    ax.bar(x + w / 2 + 0.02, adapted, w, color=S1, label=bm["label"])
    for xi, b, j in zip(x, base, adapted):
        ax.text(xi + w / 2 + 0.02, j + 1.5, f"{j:.0f}", ha="center", color=INK2, fontsize=8)
        ax.text(xi - w / 2 - 0.02, b + 1.5, f"{b:.0f}", ha="center", color=INK2, fontsize=8)
    ax.set_xticks(x, names, fontsize=8)
    ax.set(ylim=(0, 100), ylabel="Точность, %")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title("Точность по классам (кросс-валидация)", loc="left")
    fig.tight_layout()
    _save(fig, "figure3")


def figure4(plt, cv):
    best = cv.get("best_supervised")
    if not best:
        return
    files = sorted((C.RESULTS_DIR / "cv").glob(f"{best}_seed*.json"))
    if not files:
        return
    run = C.load_json(files[0])
    m = run["open"]
    cm = np.array(m["confusion_matrix"], dtype=float)
    labels = m["confusion_labels"]
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(
        range(len(labels)), [lbl if lbl != "other" else "другие 34" for lbl in labels], rotation=30, ha="right"
    )
    ax.set_yticks(range(len(labels) - 1), labels[:-1])
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    f"{int(cm[i, j])}\n{norm[i, j] * 100:.0f} %",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if norm[i, j] > 0.55 else INK,
                )
    ax.set(xlabel="Предсказано", ylabel="Истинный класс")
    title = cv["methods"][best]["label"]
    ax.set_title(f"{title}: матрица ошибок (сид {run['seed']}, n = {m['n']})", loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Доля строки")
    fig.tight_layout()
    _save(fig, "figure4")


def main():
    plt = _style()
    cv_path = C.RESULTS_DIR / "cv_summary.json"
    if not cv_path.exists():
        raise SystemExit(
            "results/cv_summary.json not found - run `python -m da.cv` and `python -m da.cv_summary` first"
        )
    cv = C.load_json(cv_path)
    figure1(plt, cv)
    figure2(plt, cv)
    figure3(plt, cv)
    figure4(plt, cv)


if __name__ == "__main__":
    main()
