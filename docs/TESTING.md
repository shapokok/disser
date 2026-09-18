# Тестирование

## Python (pytest)

```bash
uv run pytest                    # всё: юнит-тесты, API через Flask test client, инференс (если есть веса)
uv run pytest -m "not models"    # без тестов, требующих models/*.pth (так работает CI)
uv run ruff check backend scripts tests domain_adaptation_experiments/da
```

* `tests/test_units.py` — названия классов и база рекомендаций покрывают все 38 классов,
  forward всех архитектур, маскирование классов полевой модели, Grad-CAM, метрики и разбиение DA;
* `tests/test_api.py` — health/models/classes/treatment, загрузка (валидация расширения и содержимого),
  экспорт CSV/JSON/Excel/PDF на двух языках, отчёты валидации, а при наличии весов —
  `predict`/`compare`/`ensemble` с реальными моделями.

## End-to-end (Playwright)

```bash
npm ci && npx playwright install chromium
npx playwright test                       # поднимает backend/app.py сам (порт 5001)
npx playwright test --project=chromium --headed
npx playwright show-report
```

`tests/e2e/pages.spec.js` проверяет три страницы на десктопе и мобильном viewport: загрузку данных из API,
переключение языка и темы с сохранением, навигацию, drag-and-drop/валидацию файлов, полный цикл
анализа с экспортом CSV, сравнение моделей, вкладки статистики, таблицу по классам и матрицу ошибок.
Тесты, требующие обученных моделей, пропускаются, если бэкенд запущен без весов
(`CROP_MODELS=""`), поэтому набор проходит и в CI.

## CI

`.github/workflows/ci.yml`: ruff + pytest (без весов) и Playwright (chromium) на каждом push/PR.
