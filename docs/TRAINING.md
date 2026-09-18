# Обучение и оценка моделей

## 1. Датасет PlantVillage

```bash
python scripts/download_dataset.py        # Kaggle: vipoooool/new-plant-diseases-dataset (нужен kaggle.json)
python scripts/split_train_valid.py       # если скачали исходный PlantVillage без сплитов
```

Ожидаемая структура:

```
data/PlantVillage/
├── train/<38 классов>/*.jpg    70 295 изображений
└── valid/<38 классов>/*.jpg    17 572 изображения
```

## 2. Обучение

```bash
python scripts/train_models.py                              # все четыре модели, 5 эпох
python scripts/train_models.py --models hybrid --epochs 5   # только Hybrid CNN-Transformer
python scripts/train_models.py --models hybrid --device mps --batch-size 32 --workers 4
python scripts/train_models.py --models baseline --limit 512 --epochs 1   # быстрая проверка
python scripts/train_models.py --models hybrid --resume     # продолжить с последнего чекпоинта
```

Устройство выбирается автоматически (`cuda` → `mps` → `cpu`). Параметры: AdamW, warm-up + косинусное
затухание, label smoothing 0.05, аугментации RandomResizedCrop / flip / rotation / color jitter.
Learning rate по умолчанию: 1e-3 для Baseline (с нуля), 1e-4 для предобученных сетей.

Результаты:

* `models/<name>_model.pth` — лучшие веса по валидации;
* `results/checkpoints/<name>_epoch_N.pth` — два последних чекпоинта (для `--resume`);
* `results/metrics/<name>_metrics.{json,png}` — кривые обучения (показываются на странице «Статистика → Обучение»).

Ориентировочное время одной эпохи (70 k изображений, batch 32): RTX 3070 — 4–6 мин для
EfficientNet/MobileNet, ~10 мин для Hybrid; Apple M-серии (MPS) — примерно вдвое дольше.

### Почему Hybrid требует переобучения

Исходный файл `hybrid_model.pth` (282 МБ) давал 3,5 % точности при 38 классах — уровень случайного
угадывания. Обучение с Adam и lr 1e-3 для ResNet-50 + Transformer расходилось. Новый скрипт
использует lr 1e-4, warm-up, clip-grad и AdamW.

## 3. Оценка (обязательно после обучения)

```bash
python scripts/evaluate_models.py                 # все модели, авто-устройство
python scripts/evaluate_models.py --device cpu --threads 4 --batch-size 64
python scripts/evaluate_models.py --models hybrid # одна модель
```

Скрипт пишет `models/model_metrics.json` (accuracy, top-5, macro precision/recall/F1, ECE до и после
temperature scaling, время инференса, размер), `results/metrics/<name>_validation.json` (метрики по классам,
матрица ошибок 38×38, путаемые пары, бины диаграммы надёжности), `results/metrics/<name>_probs.npz`
(вероятности для пост-анализа) и `results/metrics/model_comparison.json` (попарный тест МакНемара,
точность на valid без почти-дубликатов из train — см. `scripts/check_duplicates.py`). Ансамбль всех
обученных моделей считается там же. Именно эти файлы показывает веб-приложение — в коде нет захардкоженных
метрик. `--analysis-only` пересчитывает калибровку/статистику из сохранённых вероятностей без инференса.

## 3a. Утечка train/valid и таблицы для диссертации

```bash
python scripts/check_duplicates.py      # results/metrics/duplicates.{json,md}: valid-изображения с почти-дубликатом в train
python scripts/make_thesis_tables.py    # docs/thesis/*.tex + tables.md из JSON-метрик
```

## 4. Доменная адаптация

См. `domain_adaptation_experiments/README.md`.
