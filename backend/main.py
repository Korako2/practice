from flask import Flask, request, jsonify
import os
import uuid
import time
import torch
import whisperx
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Определяем устройство: GPU (если доступно) или CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель Whisper (выбирайте размер модели: tiny, base, small, medium, large)
model = whisperx.load_model("medium", device, compute_type="float32")

# Папка для временного хранения загруженных файлов
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def transcribe_audio(file_path, with_timestamps=False, with_diarization=False):
    """
    Выполняет транскрипцию аудиофайла с использованием модели WhisperX.
    """
    start_time = time.time()
    # Расшифровка аудио с указанием языка "ru"
    result = model.transcribe(file_path, language="ru")
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Объединяем текст из всех сегментов
    full_text = " ".join(segment["text"] for segment in result["segments"])
    transcription = f"Полный текст:\n{full_text}\n\nВремя транскрипции: {elapsed_time:.2f} секунд"

    # Если запрошены временные метки, добавляем их
    if with_timestamps:
        timestamps_text = ""
        for segment in result["segments"]:
            timestamps_text += f"[{segment['start']:.2f}] {segment['text']}\n"
        transcription += "\nВременные метки:\n" + timestamps_text

    # Если запрошена диаризация, выводим сообщение о том, что функция не реализована
    if with_diarization:
        transcription += "\n-- Функция диаризации не реализована в данном примере."

    return transcription

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    # Сохраняем файл во временную папку
    filename = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Получаем параметры из запроса (чекбоксы приходят в виде строки "true" или "false")
    with_timestamps = request.form.get("timestamps", "false").lower() == "true"
    with_diarization = request.form.get("diarization", "false").lower() == "true"

    # Выполняем транскрипцию
    transcription = transcribe_audio(file_path, with_timestamps, with_diarization)

    # (Опционально) Удаляем временный файл после обработки
    os.remove(file_path)

    return jsonify({"result": transcription})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
