# План работ по диссертационному проекту

Обновлено: 18 сентября 2026 (вечер). Легенда: ✅ сделано · 🔄 выполняется · ⬜ предстоит · 💡 рекомендация.

## 1. Инфраструктура и репозиторий

| Статус | Задача |
|---|---|
| ✅ | Чистый git-репозиторий без Windows-venv, датасета и весов (`.git` 4,4 ГБ → 1,6 МБ); мусор в `~/disser_backup_2026-09-18` (можно удалить после проверки) |
| ✅ | Единое окружение: `pyproject.toml` + `uv sync`, экспорт `requirements.txt`, Python 3.12, torch 2.8 с поддержкой Apple MPS |
| ✅ | Нормализация переносов строк (`.gitattributes`), `.gitignore` для данных/весов/артефактов |
| ✅ | Docker: `python:3.12-slim`, CPU-сборка torch, gunicorn, `docker compose up` (веса монтируются снаружи) |
| ✅ | CI (GitHub Actions): ruff, pytest без весов, Playwright chromium |
| ✅ | Ветка `clean-main` запушена в https://github.com/shapokok/disser и сделана основной |
| ✅ | CI на GitHub зелёный (ruff + pytest, Playwright) |
| ⬜ | Опубликовать веса как GitHub Release после завершения DA (с полевой моделью): `gh release create v2.0.0 models/*.pth models/model_metrics.json models/field_model_info.json`; `scripts/download_weights.py` уже готов |

## 2. Бэкенд

| Статус | Задача |
|---|---|
| ✅ | Переписан `backend/app.py`: конфигурация через `CROP_*`, порт 5001 (5000 занят AirPlay на macOS — это и была причина «HTTP 502» при загрузке фото) |
| ✅ | Реальные метрики из `models/model_metrics.json` и `results/metrics/*_validation.json`; удалена генерация фиктивных матриц ошибок |
| ✅ | Необученные модели помечаются «требует обучения» и не входят в ансамбль; веса ансамбля — реальные точности |
| ✅ | Grad-CAM / LIME без matplotlib, компактные JPEG-превью; MPS/CUDA/CPU автоматически |
| ✅ | Полевой режим: адаптер `field_model.py` подключает `models/field_mobilenet_da.pth`, ограничивает классы |
| ✅ | База рекомендаций на 38 классов (ru/en), названия классов ru/en, PDF с кириллицей, экспорт CSV/JSON/Excel |
| ✅ | Отдача фронтенда самим Flask; JSON-ошибки; защита от path traversal; проверка, что загруженный файл — изображение |
| ✅ | Журнал анализов (SQLite) и страница «Журнал»: динамика по дням, частые диагнозы, фильтры, поиск |
| ✅ | Калибровка: ECE до/после temperature scaling, диаграмма надёжности во вкладке «Калибровка» (`evaluate_models.py`) |
| ✅ | `scripts/export_onnx.py` — ONNX-экспорт любой модели с проверкой через onnxruntime (запускать по необходимости) |

## 3. Модели и оценка

| Статус | Задача |
|---|---|
| ✅ | `scripts/train_models.py`: выбор моделей, AdamW + warm-up/cosine, MPS, resume, smoke-режим `--limit` |
| ✅ | `scripts/evaluate_models.py`: честные метрики на 17 572 изображениях valid, per-class, матрицы, ансамбль |
| ✅ | Оценка на valid: Baseline 98,44 %, EfficientNet 98,94 %, MobileNet 98,54 %, Hybrid 4,60 % (не обучена), ансамбль трёх 99,63 % |
| ⬜ | **Обучить Hybrid CNN-Transformer** (исходные веса не обучены, 3,5 %): `python scripts/train_models.py --models hybrid --device mps`, затем `python scripts/evaluate_models.py --models hybrid` |
| 💡 | Дообучить остальные модели дольше 5 эпох (кривые ещё растут) и сравнить при равном бюджете |
| ✅ | `scripts/check_duplicates.py`: 385 из 17 572 valid-изображений (2,2 %) имеют почти-дубликат в train, 42 точных, 25 — с другой меткой; точность без дубликатов считает `evaluate_models.py` |
| ✅ | Тест МакНемара для всех пар моделей — `results/metrics/model_comparison.json`, вкладка «Калибровка» |
| ✅ | Zero-shot на PlantDoc по архитектурам (4 класса, dev+test): Baseline 19,0 %, EfficientNet 31,0 %, MobileNet 45,2 %, Hybrid 0 % — MobileNet выбран для адаптации обоснованно |

## 4. Доменная адаптация (исследовательская часть)

Основной протокол — **стратифицированная 5-кратная кросс-валидация** на всех 500 снимках PlantDoc
четырёх классов (`da/cv.py`, сводка `da/cv_summary.py` → `results/cv_summary.{json,md}`).

| Статус | Задача |
|---|---|
| ✅ | Пакет `da/` с честным протоколом; старые скрипты в `legacy/`; фиксированное разбиение dev/test оставлено как дополнительное |
| ✅ | Кросс-валидация (3 сида): без адаптации 42,0 %; AdaBN/TENT 18 %; self-training 16,9 %; self-training v2 (CBST + EMA) 36,7 %; joint 25 эп. 62,5 %; **joint 60 эп. 72,9 % (+TTA 74,7 %)**; **EfficientNet 60 эп. 72,3 % (+TTA 75,5 %)**; по 4 классам до 80,3 % |
| ✅ | Тест МакНемара: joint 60 эп. лучше 25 эп. (p < 0,001) и исходной модели (p < 0,001); MobileNet и EfficientNet статистически неразличимы (p = 0,92); TTA даёт +1,5–3 п.п., но незначимо |
| ✅ | Рисунки `figures/figure1-4` и таблицы `docs/thesis/table_da_cv*.tex` построены по кросс-валидации |
| ✅ | TTA в полевом режиме приложения (`CROP_FIELD_TTA`) |
| ⏸ | **Пауза.** Не досчитаны для 60-эпохового бюджета: `joint_320`, `joint_closed`, `joint_v2`, `joint_v2_noadabn`, `semi25` (все сиды) и `joint_long` для сида 42 уже есть. Продолжить: `cd domain_adaptation_experiments && for s in 42 43 44; do python -m da.cv --seeds $s --device mps --skip-existing & done; wait; python -m da.cv_summary && python -m da.figures` |
| ⬜ | После завершения: обучить развёрнутую полевую модель лучшим MobileNet-рецептом `python -m da.cv --final joint_long --device mps`, обновить скриншоты, опубликовать веса |
| ⬜ | Переписать раздел результатов диссертации по `results/cv_summary.md` (старые 76–79 % не воспроизводимы) |
| ⬜ | Адаптация на все 27 классов: `python -m da.joint_training --classes all` (по желанию) |
| ⬜ | `scripts/robustness_eval.py` (размытие, шум, JPEG, освещение) — по желанию |
| 💡 | Количественная оценка объяснимости (deletion/insertion для Grad-CAM vs LIME) |

## 5. Фронтенд

| Статус | Задача |
|---|---|
| ✅ | Новый интерфейс: три страницы, ru/en, светлая/тёмная тема, шрифты Manrope + Inter (self-hosted, кириллица), градиентные акценты, кольца уверенности, анимации появления |
| ✅ | Анализ: drag-and-drop, несколько файлов, ансамбль с неопределённостью, вкладки Grad-CAM/карта/оригинал, рекомендации, экспорт по одному и всех результатов |
| ✅ | Статистика: обзор, кривые обучения, 38 классов с поиском/сортировкой, интерактивная матрица ошибок, вкладка доменной адаптации с ДИ |
| ✅ | Chart.js локально (работает без интернета на защите), mobile-friendly, без inline-стилей и `alert()` |
| ⬜ | Финальные скриншоты в `docs/screenshots/` после завершения DA (вкладка исследования, полевой режим) |
| ✅ | Кнопка «Сфотографировать» на телефоне (показывается на сенсорных устройствах) |
| 💡 | PWA-манифест и офлайн-кэш статики |

## 6. Тесты и качество

| Статус | Задача |
|---|---|
| ✅ | pytest: 34 теста (юнит + API + журнал + инференс при наличии весов), ruff чистый |
| ✅ | Playwright: 13 сценариев × desktop/mobile против реального бэкенда; работает и без весов (CI) |
| ✅ | Docker-образ собран и проверен (2,1 ГБ, CPU-torch); код монтируется в контейнер, правки видны после `docker compose restart` |
| 💡 | Нагрузочный тест API (locust) и таблица времени инференса CPU/MPS/GPU для раздела «производительность» |

## 7. Документация и текст диссертации

| Статус | Задача |
|---|---|
| ✅ | README (ru), `docs/API.md`, `docs/TRAINING.md`, `docs/DEPLOYMENT.md`, `docs/TESTING.md`, `models/README.md`, README исследования; удалены устаревшие и противоречивые документы |
| ⬜ | Заполнить автора/университет в README и цитировании |
| ⬜ | Обновить главы диссертации: честные метрики Hybrid и DA, описание протокола (adapt/dev/test, ДИ, сиды), таблицы из `summary.md` |
| ✅ | `scripts/make_thesis_tables.py` → `docs/thesis/*.tex` + `tables.md` (модели, калибровка, МакНемар, дубликаты, DA, zero-shot) |
| ✅ | Схема архитектуры: `docs/figures/architecture.svg` / `.png` |

## Ближайшие шаги по порядку

1. Досчитать кросс-валидацию (команда в разделе 4, ~1,5 ч на GPU) → сводка, рисунки, таблицы, итоговая полевая модель, скриншоты, релиз весов.
2. Обучить Hybrid, когда освободится GPU (1–2 ч на MPS): `python scripts/train_models.py --models hybrid --device mps && python scripts/evaluate_models.py`.
3. Запустить подготовленные эксперименты: `scripts/robustness_eval.py`, `da.joint_training --classes all`, консервативный self-training.
4. Обновить текст диссертации по новым цифрам (таблицы из `docs/thesis/`), заполнить автора/университет в README.
