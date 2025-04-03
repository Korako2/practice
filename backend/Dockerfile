FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Создаем директорию для кеша моделей
RUN mkdir -p /app/cache

# Устанавливаем переменные окружения для кеша
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV PYANNOTE_CACHE=/app/cache

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
ENV FLASK_APP=main
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]