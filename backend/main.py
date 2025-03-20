from flask import Flask, request, jsonify
import os
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Папка для временного хранения загруженных файлов
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def transcribe_audio(file_path, with_timestamps=False, with_diarization=False):
    """
    Здесь должна быть реализация транскрипции.
    Например, можно использовать модель Whisper, Google Speech API или другое решение.
    В данном примере возвращается тестовый результат.
    """
    # Пример: базовая транскрипция
    result_text = "Пример распознанного текста без дополнительных данных."

    # Если запрошены временные метки
    if with_timestamps:
        result_text += "\n[00:00:01] Пример фразы 1.\n[00:00:05] Пример фразы 2."

    # Если запрошена диаризация
    if with_diarization:
        result_text += "\n-- Говорящий 1: Пример фразы.\n-- Говорящий 2: Другой пример."

    return result_text

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
