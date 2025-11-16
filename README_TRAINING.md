# 🚀 Model Training Guide

Руководство по обучению моделей для классификации болезней растений.

## 📋 Подготовка

### 1. Разделение датасета

Сначала разделите данные на train/valid:

```bash
python scripts/split_train_valid.py
```

Проверьте результат:

```bash
python scripts/check_dataset.py
```

### 2. Проверка требований

Убедитесь что установлены все зависимости:

```bash
pip install torch torchvision torchaudio
pip install pillow numpy
```

## 🎯 Обучение моделей

### Вариант 1: Обучить все модели

```bash
python train.py
```

Этот скрипт обучит все 4 модели:
- **BaselineCNN** - простая CNN, быстрая
- **EfficientNet** - высокая точность, средняя скорость
- **MobileNet** - легковесная, для мобильных устройств
- **Hybrid CNN-Transformer** - максимальная точность, медленная

### Вариант 2: Обучить одну модель

```bash
# Baseline (быстро, для экспериментов)
python scripts/train_single_model.py --model baseline --epochs 30

# EfficientNet (рекомендуется для продакшена)
python scripts/train_single_model.py --model efficientnet --epochs 50

# MobileNet (для мобильных приложений)
python scripts/train_single_model.py --model mobilenet --epochs 40

# Hybrid (максимальная точность)
python scripts/train_single_model.py --model hybrid --epochs 50
```

### Вариант 3: Быстрое тестирование

Для быстрой проверки (10 эпох, маленький batch):

```bash
python scripts/train_single_model.py --model baseline --quick
```

## ⚙️ Параметры обучения

### Основные параметры

- `--model` - архитектура модели (baseline, efficientnet, mobilenet, hybrid)
- `--epochs` - количество эпох (по умолчанию: 50)
- `--batch-size` - размер батча (по умолчанию: 32)
- `--lr` - learning rate (по умолчанию: зависит от модели)
- `--quick` - быстрое обучение (10 эпох)

### Примеры

```bash
# Обучение с кастомными параметрами
python scripts/train_single_model.py --model efficientnet --epochs 100 --batch-size 64 --lr 0.0001

# Быстрое тестирование
python scripts/train_single_model.py --model baseline --quick
```

## 📊 Мониторинг обучения

### Во время обучения

Скрипт выводит:
- Loss и Accuracy для каждой эпохи
- Прогресс батчей
- Время обучения
- Автоматическое сохранение лучшей модели

Пример вывода:
```
Epoch [10/50] LR: 0.000100
  Batch [50/281] Loss: 0.2341 Acc: 92.50%
  Batch [100/281] Loss: 0.1987 Acc: 93.75%
  ...
  Train Loss: 0.2156 | Train Acc: 93.21%
  Valid Loss: 0.1843 | Valid Acc: 94.67%
  Epoch Time: 45.23s
  ✓ Saved best model (Acc: 94.67%)
```

### После обучения

Результаты сохраняются в:
- `models/` - обученные модели (.pth файлы)
- `logs/` - история обучения (JSON файлы)

## 📁 Структура выходных файлов

```
disser/
├── models/
│   ├── baseline_model.pth
│   ├── efficientnet_model.pth
│   ├── mobilenet_model.pth
│   ├── hybrid_model.pth
│   └── class_names.json
└── logs/
    ├── baseline_model_history.json
    ├── efficientnet_model_history.json
    ├── mobilenet_model_history.json
    └── hybrid_model_history.json
```

## 🔧 Настройка конфигурации

Редактируйте `train_config.json` для изменения параметров:

```json
{
  "training": {
    "batch_size": 32,
    "num_epochs": 50,
    "early_stopping_patience": 10
  },
  "models": {
    "baseline": {
      "learning_rate": 0.001
    },
    "efficientnet": {
      "learning_rate": 0.0001
    }
  }
}
```

## 💡 Рекомендации

### Для быстрых экспериментов
```bash
python scripts/train_single_model.py --model baseline --quick
```
⏱️ ~10-15 минут на CPU

### Для лучшей точности
```bash
python scripts/train_single_model.py --model efficientnet --epochs 50
```
⏱️ ~2-4 часа на GPU

### Для мобильных устройств
```bash
python scripts/train_single_model.py --model mobilenet --epochs 40
```
⏱️ ~1-2 часа на GPU

## 🎯 Ожидаемые результаты

На PlantVillage датасете:

| Модель         | Точность | Скорость | Размер |
|----------------|----------|----------|--------|
| Baseline       | ~89%     | Быстрая  | 4.8 MB |
| EfficientNet   | ~95%     | Средняя  | 16 MB  |
| MobileNet      | ~92%     | Быстрая  | 9 MB   |
| Hybrid         | ~97%     | Медленная| 102 MB |

## ⚠️ Troubleshooting

### Ошибка: CUDA out of memory
Уменьшите batch size:
```bash
python scripts/train_single_model.py --model efficientnet --batch-size 16
```

### Ошибка: FileNotFoundError (train/valid)
Запустите сначала:
```bash
python scripts/split_train_valid.py
```

### Медленное обучение
- Используйте GPU если доступен
- Уменьшите `num_workers` в конфиге
- Используйте меньшую модель (baseline или mobilenet)

## 🚀 Запуск веб-приложения

После обучения запустите приложение:

```bash
cd backend
python app.py
```

Откройте: http://localhost:5000

## 📝 Дополнительная информация

- Модели используют ImageNet pretrained weights (кроме Baseline)
- Применяется data augmentation для train set
- Early stopping для предотвращения переобучения
- Learning rate scheduling для лучшей сходимости
