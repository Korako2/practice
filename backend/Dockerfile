# Используем официальный Python образ (например, Python 3.9 slim)
FROM python:3.9-slim

# Обновляем систему и устанавливаем ffmpeg (требуется для конвертации аудио)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения в рабочую директорию
COPY . .

# Открываем порт 5000 (Flask по умолчанию)
EXPOSE 5000

# Определяем переменную окружения для Flask
ENV FLASK_APP=main
ENV FLASK_RUN_HOST=0.0.0.0

# Запускаем приложение
CMD ["flask", "run"]
