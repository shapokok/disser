#!/usr/bin/env python3
"""
Generate thesis-ready tables (LaTeX + Markdown) from the JSON files produced by
scripts/evaluate_models.py and domain_adaptation_experiments/da/run_all.py,
so the text of the thesis never drifts from the code.

    python scripts/make_thesis_tables.py          # -> docs/thesis/*.tex, docs/thesis/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from model import MODEL_TYPES

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DA_RESULTS = PROJECT_ROOT / "domain_adaptation_experiments" / "results"
OUT = PROJECT_ROOT / "docs" / "thesis"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def pct(x, d=2):
    return "—" if x is None else f"{100 * x:.{d}f}"


def tex_table(caption: str, label: str, header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(esc(h) for h in header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    return (
        "\n".join(
            ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        )
        + "\n"
    )


def label(name: str) -> str:
    return MODEL_TYPES[name]["label"] if name in MODEL_TYPES else ("Ансамбль" if name == "ensemble" else name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = [
        "# Таблицы для диссертации",
        "",
        "Сгенерировано `scripts/make_thesis_tables.py`; источники — JSON-файлы с метриками.",
        "",
    ]

    # --- 1. PlantVillage validation metrics
    metrics = load(MODELS_DIR / "model_metrics.json") or {}
    if metrics:
        header = [
            "Модель",
            "Accuracy, %",
            "Top-5, %",
            "Precision, %",
            "Recall, %",
            "Macro-F1, %",
            "ECE",
            "Инференс, мс",
            "Параметры",
            "Размер, МБ",
        ]
        rows = []
        for name in [*MODEL_TYPES, "ensemble"]:
            m = metrics.get(name)
            if not m:
                continue
            rows.append(
                [
                    label(name) + ("" if m.get("trained", True) else " (не обучена)"),
                    pct(m.get("accuracy")),
                    pct(m.get("top5_accuracy")),
                    pct(m.get("precision")),
                    pct(m.get("recall")),
                    pct(m.get("f1_score")),
                    f"{m['ece']:.3f}" if m.get("ece") is not None else "—",
                    f"{m['inference_time_ms']:.0f}" if m.get("inference_time_ms") else "—",
                    m.get("parameters") or "—",
                    f"{m['size_mb']:.1f}" if m.get("size_mb") else "—",
                ]
            )
        n = next((m.get("validation_samples") for m in metrics.values() if m.get("validation_samples")), "?")
        (OUT / "table_models.tex").write_text(
            tex_table(
                f"Метрики моделей на валидационной выборке PlantVillage ({n} изображений)", "tab:models", header, rows
            ),
            encoding="utf-8",
        )
        md += [f"## Модели на PlantVillage/valid (n = {n})", "", md_table(header, rows)]

        # calibration
        rows_c = [
            [
                label(n),
                f"{m['ece']:.4f}",
                f"{m['ece_after_temperature']:.4f}",
                f"{m['temperature']:.2f}",
                f"{m['nll']:.3f}",
            ]
            for n, m in metrics.items()
            if m.get("ece_after_temperature") is not None
        ]
        if rows_c:
            header_c = ["Модель", "ECE", "ECE после temperature scaling", "T", "NLL"]
            (OUT / "table_calibration.tex").write_text(
                tex_table(
                    "Калибровка уверенности (15 бинов; T подобрана на половине valid, ECE — на другой половине)",
                    "tab:calibration",
                    header_c,
                    rows_c,
                ),
                encoding="utf-8",
            )
            md += ["## Калибровка", "", md_table(header_c, rows_c)]

    # --- 2. McNemar + de-duplication
    comp = load(METRICS_DIR / "model_comparison.json")
    if comp and comp.get("mcnemar"):
        header_m = ["Модель A", "Модель B", "A верно, B нет", "B верно, A нет", "χ²", "p"]
        rows_m = [
            [label(r["a"]), label(r["b"]), r["a_only"], r["b_only"], f"{r['statistic']:.2f}", f"{r['p_value']:.3g}"]
            for r in comp["mcnemar"]
        ]
        (OUT / "table_mcnemar.tex").write_text(
            tex_table(
                "Попарное сравнение моделей: тест МакНемара на валидационной выборке",
                "tab:mcnemar",
                header_m,
                rows_m,
                "llrrrr",
            ),
            encoding="utf-8",
        )
        md += ["## Тест МакНемара", "", md_table(header_m, rows_m)]
    if comp and comp.get("deduplicated"):
        header_d = ["Модель", "Accuracy (все), %", "Accuracy (без дубликатов), %", "Удалено изображений"]
        rows_d = [
            [label(n), pct(v["accuracy_all"]), pct(v["accuracy_dedup"]), v["removed"]]
            for n, v in comp["deduplicated"].items()
        ]
        (OUT / "table_dedup.tex").write_text(
            tex_table(
                "Точность на valid после удаления изображений, имеющих почти-дубликат в train",
                "tab:dedup",
                header_d,
                rows_d,
            ),
            encoding="utf-8",
        )
        md += ["## Утечка train/valid", "", md_table(header_d, rows_d)]

    # --- 3. Domain adaptation
    summ = load(DA_RESULTS / "summary.json")
    if summ:
        header_da = [
            "Метод",
            "Accuracy dev+test, %",
            "95 % ДИ",
            "Macro-F1, %",
            "Accuracy test, %",
            "Accuracy (4 класса), %",
        ]
        rows_da = []
        for key, m in summ["methods"].items():
            e, t, r = m["eval_open"], m["test_open"], m["eval_restricted"]
            pm = f" ± {100 * e['accuracy_std']:.1f}" if e["runs"] > 1 else ""
            rows_da.append(
                [
                    "Без адаптации" if key == "baseline" else m["label"],
                    pct(e["accuracy_mean"], 1) + pm,
                    f"{100 * e['ci95_pooled'][0]:.0f}–{100 * e['ci95_pooled'][1]:.0f}",
                    pct(e["macro_f1_mean"], 1),
                    pct(t["accuracy_mean"], 1),
                    pct(r["accuracy_mean"], 1),
                ]
            )
        splits = summ["protocol"].get("splits") or {}
        cap = (
            "Доменная адаптация PlantVillage → PlantDoc: "
            + ", ".join(f"{k} = {v['n']}" for k, v in splits.items())
            + f"; сиды {summ['seeds']}"
        )
        (OUT / "table_da.tex").write_text(tex_table(cap, "tab:da", header_da, rows_da), encoding="utf-8")
        md += ["## Доменная адаптация", "", md_table(header_da, rows_da)]
        if summ.get("per_class_eval_open"):
            header_pc = ["Класс", "n", "Без адаптации, %", "Joint fine-tuning, %"]
            rows_pc = [
                [c, v["support"], pct(v["baseline"], 1), pct(v["joint"], 1)]
                for c, v in summ["per_class_eval_open"].items()
            ]
            (OUT / "table_da_classes.tex").write_text(
                tex_table("Точность по классам на dev+test до и после адаптации", "tab:da-classes", header_pc, rows_pc),
                encoding="utf-8",
            )
            md += ["### По классам", "", md_table(header_pc, rows_pc)]
        if summ.get("zero_shot_by_arch"):
            header_z = ["Архитектура", "4 класса, dev+test, %", "27 классов, test, %", "Top-5 (27), %"]
            rows_z = [
                [label(a), pct(v["focus_eval_open"], 1), pct(v["all27_test_open"], 1), pct(v["all27_top5"], 1)]
                for a, v in summ["zero_shot_by_arch"].items()
            ]
            (OUT / "table_zero_shot.tex").write_text(
                tex_table(
                    "Zero-shot обобщение моделей PlantVillage на полевые снимки PlantDoc",
                    "tab:zero-shot",
                    header_z,
                    rows_z,
                ),
                encoding="utf-8",
            )
            md += ["### Zero-shot по архитектурам", "", md_table(header_z, rows_z)]

    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT}/tables.md and {len(list(OUT.glob('*.tex')))} .tex tables")


if __name__ == "__main__":
    main()
