# Доменная адаптация PlantVillage → PlantDoc

Исследовательская часть диссертации. Модель, обученная на лабораторных снимках
PlantVillage (лист на однородном фоне), проверяется и адаптируется к полевым
фотографиям PlantDoc.

## Данные

```
datasets/plantdoc/
├── train/<27 классов>/*.jpg   2 316 изображений
└── test/<27 классов>/*.jpg      236 изображений
```

Скачать: <https://github.com/pratikkayal/PlantDoc-Dataset> (папки `train/` и `test/`
положить в `datasets/plantdoc/`; путь можно переопределить переменной `PLANTDOC_DIR`).

27 классов PlantDoc сопоставлены индексам PlantVillage (`da/common.py: PLANTDOC_TO_PV`).
Основные эксперименты ведутся на четырёх классах, где в PlantDoc достаточно изображений:

| PlantDoc | PlantVillage (индекс) | train | test |
|---|---|---|---|
| Corn Gray leaf spot | 7 | 64 | 4 |
| Corn leaf blight | 9 | 180 | 12 |
| Squash Powdery mildew leaf | 25 | 123 | 6 |
| Tomato leaf late blight | 30 | 101 | 10 |

## Протокол (честный)

Ранние скрипты (`legacy/`) обучали и оценивали модель **на одних и тех же** изображениях
train-части PlantDoc, поэтому их 76–79 % не являются оценкой обобщающей способности.
Новый протокол:

1. train-часть 4 классов (468 изобр.) делится стратифицированно с фиксированным сидом:
   **adapt** (374) — для адаптации, **dev** (94) — для выбора модели/ранней остановки;
2. официальный **test** (32) используется один раз в конце;
3. заголовочная метрика — точность на **dev + test** (126 изобр.) в режиме *open-set*
   (argmax среди всех 38 классов); дополнительно даётся *restricted* (argmax среди 4 классов),
   95 % интервалы Уилсона и среднее ± σ по трём сидам;
4. self-training не использует метки adapt-выборки вообще (псевдометки только из
   уверенных предсказаний, порог 0.85 → 0.65).

Разбиение кэшируется в `results/splits_*.json`, так что все методы видят одни и те же изображения.

## Методы (`da/`)

| Модуль | Метод |
|---|---|
| `evaluate.py` | zero-shot оценка исходной модели (4 класса и все 27) или любого чекпоинта |
| `self_training.py` | self-training с псевдометками, 5 итераций × 3 эпохи |
| `joint_training.py` | совместное дообучение всех 4 классов с балансировкой и тяжёлой аугментацией (25 эпох) |
| `progressive.py` | прогрессивная адаптация: лёгкие классы → сложные → совместно |
| `tta.py` | test-time augmentation (10 детерминированных видов) |
| `run_all.py` | весь пайплайн, `results/summary.{json,md}`, экспорт лучшей модели в `models/field_mobilenet_da.pth` |
| `figures.py` | figure1–4 (кривые, сравнение методов с ДИ, по классам, матрица ошибок) |
| `flowchart.py` | схема методологии `figures/figure_methodology.pdf` |

## Запуск

```bash
cd domain_adaptation_experiments
python -m da.run_all --seeds 42 43 44            # авто-выбор устройства (cuda / mps / cpu)
python -m da.run_all --device cpu --threads 4     # ~3 ч на CPU Apple M-серии
python -m da.run_all --skip-existing --only summary figures   # только пересобрать таблицу и рисунки
```

Отдельные шаги: `python -m da.joint_training --seed 42`, `python -m da.evaluate --checkpoint results/checkpoints/joint_seed42.pth`.

Результаты: `results/summary.md` (таблица методов), `results/<метод>_seed<k>.json` (история,
метрики по классам, матрицы), `figures/*.pdf|png`. Веб-приложение читает `results/summary.json`
на вкладке «Доменная адаптация» и использует экспортированную модель в полевом режиме.

## Legacy

`legacy/` — исходные скрипты и результаты с абсолютными Windows-путями и оценкой на
обучающих данных. Сохранены для истории и не запускаются.
