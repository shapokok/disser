# 🎓 Model Training Guide

Полное руководство по обучению всех 4 моделей для диссертационной работы.

---

## 📋 Требования

### Оборудование
- **GPU**: RTX 3070 (8GB VRAM) ✅
- **RAM**: 16GB+ рекомендуется
- **Диск**: ~10GB свободного места

### Программное обеспечение
- Python 3.8+
- CUDA 11.8+ (для GPU)
- PyTorch 2.0+

---

## 🚀 Быстрый старт (3 шага)

### Шаг 1: Скачать датасет

```bash
python scripts/download_dataset.py
```

**Если автоматическая загрузка не работает:**
1. Скачайте датасет вручную:
   - Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
   - Или: https://data.mendeley.com/datasets/tywbtsjrjv
2. Сохраните ZIP файл в `data/plantvillage.zip`
3. Запустите скрипт снова - он автоматически распакует

---

### Шаг 2: Установить зависимости

```bash
# Активируйте виртуальное окружение (если есть)
.\venv\Scripts\Activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Установите зависимости
cd backend
pip install -r requirements.txt

# Дополнительно для обучения
pip install tqdm matplotlib
```

---

### Шаг 3: Запустить обучение

```bash
# Вернитесь в корень проекта
cd ..

# Запустите обучение всех моделей
python scripts/train_models.py
```

---

## ⏱️ Ожидаемое время (RTX 3070)

| Модель | Время обучения | Примерный размер |
|--------|----------------|------------------|
| Baseline CNN | 15-20 минут | 4.8 MB |
| MobileNet-V2 | 25-30 минут | 9.1 MB |
| EfficientNet-B0 | 35-40 минут | 16.2 MB |
| Hybrid CNN-Transformer | 1-1.5 часа | 102.4 MB |
| **ВСЕГО** | **~2.5-3 часа** | **~130 MB** |

*Включает 30-секундные паузы между моделями для охлаждения GPU*

---

## 🎮 Особенности обучения

### ✅ Auto-Checkpoint
- Сохранение каждые 5 эпох
- Можно прервать и продолжить позже
- Хранятся последние 3 checkpoint'а

### ✅ Resume Training
Если прервали обучение:
```bash
# Просто запустите снова
python scripts/train_models.py

# Скрипт автоматически продолжит с последнего checkpoint
```

### ✅ GPU Cooling
- 30 секунд паузы между моделями
- Предотвращает перегрев RTX 3070

### ✅ Metrics Export
Автоматически сохраняются:
- Графики loss и accuracy
- JSON с метриками
- Лучшие веса моделей

---

## 📁 Структура после обучения

```
models/
├── baseline_model.pth          ← Обученные веса
├── mobilenet_model.pth
├── efficientnet_model.pth
├── hybrid_model.pth
└── class_names.json

results/
├── checkpoints/                ← Промежуточные сохранения
│   ├── baseline_checkpoint_epoch_25.pth
│   └── ...
└── metrics/                    ← Метрики и графики
    ├── baseline_metrics.png
    ├── baseline_metrics.json
    ├── efficientnet_metrics.png
    ├── efficientnet_metrics.json
    └── ...
```

---

## 🎯 После обучения

### 1. Перезапустить backend

```bash
# Остановить Docker
docker-compose down

# Перезапустить (подхватит новые модели)
docker-compose up -d --build
```

### 2. Проверить результаты

Откройте: http://localhost:8000/stats.html

Теперь увидите **реальные** метрики!

### 3. Тестировать предсказания

Откройте: http://localhost:8000/analyze.html

Загрузите изображение из `data/sample_images/` - теперь будут **правильные** результаты!

---

## 🛠️ Настройка параметров

Откройте `scripts/train_models.py` и измените:

```python
# Меньше эпох = быстрее (но хуже точность)
NUM_EPOCHS = 20  # По умолчанию: 30

# Меньше batch size = меньше нагрузка на GPU
BATCH_SIZE = 8  # По умолчанию: 16

# Скорость обучения
LEARNING_RATE = 0.0005  # По умолчанию: 0.001
```

---

## ⚠️ Решение проблем

### Проблема: Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```

**Решение:**
1. Уменьшить `BATCH_SIZE` с 16 до 8
2. Закрыть другие программы
3. Обучать по одной модели за раз

---

### Проблема: Датасет не найден
```
❌ Dataset not found at data/PlantVillage
```

**Решение:**
```bash
python scripts/download_dataset.py
```

---

### Проблема: GPU не определяется
```
⚠️  No GPU found, using CPU
```

**Решение:**
1. Установите CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Переустановите PyTorch с CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Ожидаемая точность

После обучения на PlantVillage dataset:

| Модель | Ожидаемая accuracy |
|--------|-------------------|
| Baseline CNN | 88-92% |
| MobileNet-V2 | 92-95% |
| EfficientNet-B0 | 95-97% |
| Hybrid CNN-Transformer | 96-98% |

*Результаты могут варьироваться ±2%*

---

## 🎓 Для диссертации

### Что включить:
1. ✅ Графики обучения из `results/metrics/`
2. ✅ Сравнение моделей (будет на stats.html)
3. ✅ Confusion matrices (автоматически в приложении)
4. ✅ Метрики JSON из `results/metrics/*.json`

### Скриншоты для защиты:
- Statistics Dashboard с реальными графиками
- Примеры предсказаний (Analyze page)
- Confusion Matrix для лучшей модели
- Comparison всех 4 моделей

---

## 💡 Советы

### Оптимизация времени:
- Начните обучение вечером перед сном (~3 часа)
- Или обучайте по 1-2 модели в день

### Сохранение батареи:
- Подключите ноутбук к розетке
- GPU потребляет много энергии

### Контроль температуры:
- Используйте охлаждающую подставку
- Скрипт автоматически делает паузы

---

## ✅ Checklist

- [ ] Скачан датасет PlantVillage
- [ ] Установлены зависимости
- [ ] Запущено обучение всех 4 моделей
- [ ] Проверены метрики в `results/metrics/`
- [ ] Модели сохранены в `models/*.pth`
- [ ] Перезапущен Docker backend
- [ ] Проверены реальные предсказания
- [ ] Сделаны скриншоты для диссертации

---

**Готовы к обучению? 🚀**

```bash
python scripts/train_models.py
```
