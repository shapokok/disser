# models/

Веса моделей не хранятся в git (`*.pth` в `.gitignore`). В репозитории лежат только:

| Файл | Что это | Кто создаёт |
|---|---|---|
| `class_names.json` | 38 классов PlantVillage в порядке индексов выходного слоя | `scripts/train_models.py` |
| `model_metrics.json` | реальные метрики каждой модели и ансамбля на `data/PlantVillage/valid` | `scripts/evaluate_models.py` |
| `field_model_info.json` | описание адаптированной полевой модели (метод, сид, точность на PlantDoc) | `domain_adaptation_experiments/da/run_all.py` |

Ожидаемые веса:

| Файл | Архитектура | Как получить |
|---|---|---|
| `baseline_model.pth` | Baseline CNN | `python scripts/train_models.py --models baseline` |
| `efficientnet_model.pth` | EfficientNet-B0 | `python scripts/train_models.py --models efficientnet` |
| `mobilenet_model.pth` | MobileNet-V2 | `python scripts/train_models.py --models mobilenet` |
| `hybrid_model.pth` | Hybrid CNN-Transformer (ResNet-50 + Transformer) | `python scripts/train_models.py --models hybrid` |
| `field_mobilenet_da.pth` | MobileNet-V2, адаптированная к PlantDoc (4 класса) | `cd domain_adaptation_experiments && python -m da.run_all` |

После обучения обязательно пересчитайте метрики, иначе приложение будет показывать прочерки:

```bash
python scripts/evaluate_models.py
```

Приложение помечает модель как «требует обучения», если её точность на валидации ниже порога
`CROP_TRAINED_THRESHOLD` (по умолчанию 0.5), и исключает её из ансамбля.
