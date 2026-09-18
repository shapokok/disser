# Таблицы для диссертации

Сгенерировано `scripts/make_thesis_tables.py`; источники — JSON-файлы с метриками.

## Модели на PlantVillage/valid (n = 17572)

| Модель | Accuracy, % | Top-5, % | Precision, % | Recall, % | Macro-F1, % | ECE | Инференс, мс | Параметры | Размер, МБ |
|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 98.44 | 99.97 | 98.46 | 98.43 | 98.43 | 0.006 | 12 | 0.5M | 2.2 |
| EfficientNet-B0 | 98.94 | 99.99 | 98.93 | 98.95 | 98.93 | 0.004 | 379 | 4.1M | 16.5 |
| MobileNet-V2 | 98.54 | 100.00 | 98.53 | 98.52 | 98.52 | 0.002 | 159 | 2.3M | 9.3 |
| Hybrid CNN-Transformer (не обучена) | 4.60 | 21.19 | 0.40 | 4.33 | 0.52 | 0.021 | 70 | 74.0M | 296.2 |
| Ансамбль | 99.63 | — | 99.63 | 99.63 | 99.63 | 0.016 | 550 | 6.9M | 28.0 |

## Калибровка

| Модель | ECE | ECE после temperature scaling | T | NLL |
|---|---|---|---|---|
| Baseline CNN | 0.0059 | 0.0032 | 0.85 | 0.050 |
| EfficientNet-B0 | 0.0044 | 0.0028 | 1.28 | 0.034 |
| MobileNet-V2 | 0.0022 | 0.0019 | 1.09 | 0.044 |
| Hybrid CNN-Transformer | 0.0209 | 0.0118 | 1.28 | 3.496 |

## Тест МакНемара

| Модель A | Модель B | A верно, B нет | B верно, A нет | χ² | p |
|---|---|---|---|---|---|
| Baseline CNN | EfficientNet-B0 | 158 | 246 | 18.74 | 1.5e-05 |
| Baseline CNN | MobileNet-V2 | 200 | 219 | 0.77 | 0.379 |
| Baseline CNN | Hybrid CNN-Transformer | 16493 | 5 | 16476.01 | 0 |
| Baseline CNN | Ансамбль | 22 | 232 | 171.97 | 2.74e-39 |
| EfficientNet-B0 | MobileNet-V2 | 218 | 149 | 12.60 | 0.000386 |
| EfficientNet-B0 | Hybrid CNN-Transformer | 16584 | 8 | 16558.02 | 0 |
| EfficientNet-B0 | Ансамбль | 23 | 145 | 87.15 | 1.01e-20 |
| MobileNet-V2 | Hybrid CNN-Transformer | 16507 | 0 | 16505.00 | 0 |
| MobileNet-V2 | Ансамбль | 14 | 205 | 164.84 | 9.91e-38 |
| Hybrid CNN-Transformer | Ансамбль | 0 | 16698 | 16696.00 | 0 |

## Утечка train/valid

| Модель | Accuracy (все), % | Accuracy (без дубликатов), % | Удалено изображений |
|---|---|---|---|
| Baseline CNN | 98.44 | 98.42 | 385 |
| EfficientNet-B0 | 98.94 | 98.94 | 385 |
| MobileNet-V2 | 98.54 | 98.54 | 385 |
| Hybrid CNN-Transformer | 4.60 | 4.67 | 385 |
| Ансамбль | 99.63 | 99.63 | 385 |
