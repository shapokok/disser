# Запуск и развёртывание

## Локально (рекомендуется)

```bash
uv sync --extra dev                 # создаёт .venv с Python ≥ 3.10 и всеми зависимостями
uv run python backend/app.py        # http://localhost:5001
```

или без uv:

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python backend/app.py
```

`scripts/run.sh` / `scripts/run.bat` делают то же самое одной командой.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `CROP_PORT` | `5001` | порт (5000 на macOS занят AirPlay Receiver) |
| `CROP_HOST` | `0.0.0.0` | интерфейс |
| `CROP_DEBUG` | `false` | режим отладки Flask |
| `CROP_DEVICE` | `auto` | `auto` → cuda / mps / cpu |
| `CROP_MODELS` | `baseline,efficientnet,mobilenet,hybrid` | какие модели загружать |
| `CROP_MODELS_DIR`, `CROP_DATA_DIR`, `CROP_UPLOAD_DIR`, `CROP_RESULTS_DIR` | `models/`, `data/`, … | пути |
| `CROP_FIELD_MODEL` | `models/field_mobilenet_da.pth` | адаптированная полевая модель |
| `CROP_TRAINED_THRESHOLD` | `0.5` | ниже этой точности модель считается необученной |
| `CROP_LIME_SAMPLES` | `300` | число возмущений LIME (меньше — быстрее) |
| `CROP_MAX_FILE_MB` | `16` | лимит загрузки |
| `CROP_CORS_ORIGINS` | `*` | CORS для `/api/*` |
| `CROP_HISTORY` | `true` | вести журнал анализов |
| `CROP_HISTORY_DB` | `results/history.sqlite` | путь к базе журнала |

## Docker

```bash
docker compose up --build           # http://localhost:5001
```

Образ содержит зависимости и метрики; код (`backend/`, `frontend/`) монтируется из репозитория, поэтому после
правок достаточно `docker compose restart`. Веса моделей монтируются из `./models`, результаты
DA-экспериментов — из `./domain_adaptation_experiments/results`. Пересборка образа нужна только при смене
зависимостей: `docker compose up --build -d`. Внутри контейнера работает
gunicorn (1 воркер × 4 потока, таймаут 300 с для LIME). Используется CPU-сборка torch.

## Продакшен без Docker

```bash
uv run gunicorn --chdir backend --bind 0.0.0.0:5001 --workers 1 --threads 4 --timeout 300 app:app
```

Фронтенд отдаётся самим Flask (`/`, `/analyze`, `/stats`), отдельный веб-сервер не нужен.
Для внешнего доступа поставьте перед ним nginx/Caddy с HTTPS и лимитом размера тела запроса ≥ 16 МБ.
