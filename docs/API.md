# HTTP API

Базовый адрес: `http://localhost:5001` (порт — `CROP_PORT`). Все ответы — JSON, ошибки имеют вид
`{"success": false, "error": "..."}` с соответствующим HTTP-статусом. Параметр `lang=ru|en`
(query или поле JSON) выбирает язык названий классов в рекомендациях и экспорте.

## Служебные

| Метод | Путь | Описание |
|---|---|---|
| GET | `/api/health` | статус, устройство, загруженные и обученные модели |
| GET | `/api/models` | описание архитектур, метрики, лучшая модель, полевая модель |
| GET | `/api/classes` | 38 классов с названиями ru/en, группировка по культурам |
| GET | `/api/dataset_info` | размеры сплитов PlantVillage |
| GET | `/api/stats` | `models/model_metrics.json`, кривые обучения, датасет |
| GET | `/api/validation/all`, `/api/validation/<model>` | per-class метрики, матрица ошибок, путаемые пары |
| GET | `/api/confusion_matrix/<model>` | только матрица и подписи |
| GET | `/api/research` | сводка доменной адаптации (`domain_adaptation_experiments/results/summary.json`) |
| GET | `/api/treatment/<class_raw>?lang=` | рекомендации по классу |
| GET | `/api/uploads/<name>` | загруженное изображение |

## Инференс

### `POST /api/upload` (multipart, поле `files`, можно несколько)

```json
{"success": true, "count": 1, "uploaded": [{"original_name": "leaf.jpg", "saved_name": "20260918_120000_123_leaf.jpg", "url": "/api/uploads/..."}], "errors": []}
```

### `POST /api/predict`

```json
{"image_path": "20260918_120000_123_leaf.jpg", "model": "efficientnet", "explanation": "gradcam|lime|none", "dataset_type": "controlled|field", "lang": "ru"}
```

Ответ:

```json
{
  "success": true, "image_name": "...", "model": "efficientnet", "model_label": "EfficientNet-B0",
  "explanation": "gradcam", "dataset_type": "controlled",
  "prediction": {"class_raw": "Tomato___Late_blight", "class": "Tomato — Late blight", "class_ru": "Томат — Фитофтороз",
                 "plant": "Tomato", "plant_ru": "Томат", "disease": "Late blight", "disease_ru": "Фитофтороз",
                 "healthy": false, "confidence": 0.93, "confidence_percent": "93.00%"},
  "top_predictions": [ ...5 элементов той же формы... ],
  "images": {"original": "<base64 jpeg>", "heatmap": "...", "overlay": "..."},
  "field_mode": null, "ensemble": null,
  "treatment": {"disease_name": "Фитофтороз", "severity": "критическая", "symptoms": "...", "treatments": [...], "prevention": [...], "organic_options": [...]},
  "inference_time_ms": 120.4, "timestamp": "2026-09-18T12:00:00"
}
```

В режиме `dataset_type=field` при наличии `models/field_mobilenet_da.pth` используется адаптированная
модель, предсказание ограничено её классами, а `field_mode.covered_classes` перечисляет их.

### `POST /api/ensemble`

Те же поля без `model`, плюс `ensemble_method: weighted|average|voting`. В ответе `ensemble` содержит
`models_used`, веса, индивидуальные предсказания, `agreement_rate` и `uncertainty_metrics`
(энтропия, дисперсия, уровень `low|medium|high`). Ансамбль включает только обученные модели.

### `POST /api/compare` — `{"image_path": "...", "models": ["baseline", ...]}`

Предсказания каждой модели, время инференса, `agreement` (сколько моделей согласны).

### `POST /api/batch` — `{"image_paths": [...], "model": "...", "explanation": "..."}`

## Журнал анализов

Каждый успешный `predict` / `ensemble` / `batch` записывается в SQLite (`results/history.sqlite`, отключается
`CROP_HISTORY=false`). Хранится миниатюра 160 px и результат без больших изображений.

| Метод | Путь | Описание |
|---|---|---|
| GET | `/api/history?limit=&offset=&model=&plant=&healthy=0\|1&q=` | записи (новые первыми), `total` |
| GET | `/api/history/stats` | всего, доля здоровых, топ классов, по культурам, по моделям, по дням |
| GET | `/api/history/<id>` | полная запись с результатом |
| DELETE | `/api/history/<id>`, `/api/history` | удалить запись / очистить журнал |

## Экспорт

| Путь | Тело | Результат |
|---|---|---|
| `POST /api/export/csv\|json\|excel\|pdf?lang=` | `{"results": [ответы predict]}` | файл |
| `POST /api/export/comparison/csv\|excel\|pdf?lang=` | ответ `/api/compare` | файл |
| `POST /api/generate_report` | как выше | PDF (совместимость) |
