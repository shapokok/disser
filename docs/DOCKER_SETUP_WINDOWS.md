# Docker Setup для Windows

## Шаг 1: Установка Docker Desktop

### 1.1 Скачайте Docker Desktop
1. Перейдите на https://www.docker.com/products/docker-desktop/
2. Скачайте **Docker Desktop for Windows**
3. Запустите установщик `Docker Desktop Installer.exe`

### 1.2 Системные требования
- Windows 10 64-bit: Pro, Enterprise, или Education (Build 19041 или выше)
- WSL 2 (Windows Subsystem for Linux 2)
- Минимум 4GB RAM (рекомендуется 8GB+)

### 1.3 Включите WSL 2 (если не включен)
Откройте PowerShell как администратор и выполните:
```powershell
wsl --install
```
Перезагрузите компьютер после установки.

### 1.4 Запустите Docker Desktop
1. Найдите Docker Desktop в меню Пуск и запустите
2. Дождитесь, пока Docker запустится (значок Docker в трее станет зелёным)
3. В настройках Docker Desktop убедитесь, что включен WSL 2:
   - Settings → General → Use the WSL 2 based engine ✓

## Шаг 2: Проверка установки

Откройте PowerShell и выполните:
```powershell
docker --version
docker-compose --version
```

Должно вывести версии Docker и Docker Compose.

## Шаг 3: Запуск проекта

### Вариант A: С помощью PowerShell скрипта (рекомендуется)

```powershell
# Перейдите в папку проекта
cd C:\Users\Нурислам\Desktop\disser

# Запустите скрипт
.\docker-start.ps1
```

### Вариант B: Вручную через docker-compose

```powershell
# 1. Создайте необходимые папки
mkdir -p data/uploads, data/test_images, models, results/heatmaps

# 2. Соберите образ
docker-compose build

# 3. Запустите контейнеры
docker-compose up -d

# 4. Проверьте статус
docker-compose ps
```

## Шаг 4: Проверка работы

### Проверьте backend API:
Откройте в браузере: http://localhost:5000

Должна загрузиться страница с API информацией.

### Проверьте frontend:
Откройте файл в браузере:
```
C:\Users\Нурислам\Desktop\disser\crop_monitoring_app\frontend\index.html
```

## Полезные команды

### Просмотр логов
```powershell
# Все логи
docker-compose logs -f

# Только backend
docker-compose logs -f backend
```

### Остановка
```powershell
# Остановить, но не удалять контейнеры
docker-compose stop

# Остановить и удалить контейнеры
docker-compose down
```

### Перезапуск
```powershell
docker-compose restart
```

### Удаление всего (включая volumes)
```powershell
docker-compose down -v
```

### Пересборка после изменений кода
```powershell
docker-compose build --no-cache
docker-compose up -d
```

## Устранение проблем

### Docker Desktop не запускается
1. Убедитесь, что WSL 2 установлен: `wsl --status`
2. Обновите WSL 2: `wsl --update`
3. Перезапустите Docker Desktop

### Порт 5000 занят
```powershell
# Найдите процесс на порту 5000
netstat -ano | findstr :5000

# Убейте процесс (замените PID на номер из предыдущей команды)
taskkill /PID <PID> /F
```

### Не хватает памяти
В Docker Desktop → Settings → Resources:
- Увеличьте Memory до минимум 6GB
- Увеличьте CPUs до 4

### Контейнер падает при старте
```powershell
# Проверьте логи ошибок
docker-compose logs backend

# Проверьте статус
docker-compose ps
```

## Режимы запуска

### Development mode (по умолчанию)
```powershell
docker-compose up -d
```
- Backend: http://localhost:5000
- Frontend: Открыть HTML файл вручную

### Production mode (с Nginx)
```powershell
docker-compose --profile production up -d
```
- Backend: http://localhost:5000
- Frontend: http://localhost (через Nginx)

## Экспорт функциональность

После запуска Docker контейнера:
1. Откройте `crop_monitoring_app/frontend/analyze.html`
2. Загрузите изображения
3. Нажмите "Analyze"
4. После анализа появятся кнопки экспорта:
   - 📄 Export as CSV
   - 📋 Export as JSON
   - 📊 Export as Excel

Файлы будут автоматически скачаны в папку Downloads.

## Дополнительная информация

### Структура Docker volumes
- `./data/uploads` - Загруженные изображения
- `./models` - ML модели
- `./results` - Результаты анализа и heatmaps

### Обновление проекта из Git
```powershell
git pull
docker-compose build --no-cache
docker-compose up -d
```
