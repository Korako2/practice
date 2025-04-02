from flask import Flask, request, jsonify
import os
import uuid
import time
import subprocess
import torch
import whisperx
from flask_cors import CORS
from pyannote.audio import Pipeline

app = Flask(__name__)
CORS(app)

# Определяем устройство: GPU (если доступно) или CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель WhisperX (варианты: tiny, base, small, medium, large)
model = whisperx.load_model("medium", device, compute_type="float32")

# Папка для временного хранения загруженных файлов
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def convert_audio_to_wav(input_path):
    """
    Преобразует аудио в формат WAV с частотой 16000 Гц и одним каналом (моно)
    с помощью ffmpeg.
    """
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    command = [
        "ffmpeg", "-y",         # -y перезаписывает файл, если он существует
        "-i", input_path,        # входной файл
        "-ar", "16000",          # установка частоты дискретизации в 16000 Гц
        "-ac", "1",              # установка одного аудио канала (моно)
        output_path
    ]
    # Выполняем команду и игнорируем вывод
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def split_segment_by_diarization(seg, diarization_intervals):
    """
    Разбивает транскрибированный сегмент по говорящим, основываясь на пересечении с диаризацией.
    Назначает сегменту спикера с наибольшим пересечением (чтобы избежать дублирования).
    """
    seg_start = seg.get("start", 0)
    seg_end = seg.get("end", seg_start + 1.0)

    best_match = None
    max_overlap = 0

    for d in diarization_intervals:
        # Вычисляем пересечение сегмента и интервала спикера
        overlap_start = max(seg_start, d["start"])
        overlap_end = min(seg_end, d["end"])
        overlap = max(0, overlap_end - overlap_start)

        if overlap > max_overlap:
            max_overlap = overlap
            best_match = d["speaker"]

    # Если совпадений нет, оставляем "UNKNOWN"
    speaker = best_match if best_match else "UNKNOWN"

    return [{
        "start": seg_start,
        "end": seg_end,
        "speaker": speaker,
        "text": seg["text"]
    }]


def merge_intervals(segments, gap_threshold=0.5):
    """
    Объединяет подряд идущие сегменты с одним спикером, исключая дублирование текста
    и корректно объединяя близко расположенные сегменты.
    """
    if not segments:
        return []

    merged = [segments[0]]

    for seg in segments[1:]:
        last = merged[-1]

        if seg["speaker"] == last["speaker"] and (seg["start"] - last["end"] <= gap_threshold):
            # Избегаем дублирования текста при слиянии
            if seg["text"] not in last["text"]:
                if last["text"] and not last["text"].endswith(" "):
                    last["text"] += " "
                last["text"] += seg["text"]
            last["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


def transcribe_audio(file_path, with_timestamps=False, with_diarization=False):
    """
    Выполняет транскрипцию аудиофайла с использованием модели WhisperX.
    Если запрошена идентификация говорящих, то для каждого сегмента производится разбиение по
    пересечениям с диаризационными интервалами, а затем объединяются соседние сегменты одного спикера.
    """
    start_time = time.time()
    # Транскрибируем аудио, устанавливая параметр chunk_size для получения более коротких сегментов
    result = model.transcribe(file_path, language="ru", chunk_size=10)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Собираем общий текст транскрипции
    full_text = " ".join(segment["text"] for segment in result["segments"])
    transcription = f"Полный текст:\n{full_text}\n\nВремя транскрипции: {elapsed_time:.2f} секунд"

    # Добавляем временные метки, если требуется
    if with_timestamps:
        timestamps_text = ""
        for segment in result["segments"]:
            timestamps_text += f"[{segment['start']:.2f}] {segment['text']}\n"
        transcription += "\nВременные метки:\n" + timestamps_text

    # Если запрошена идентификация говорящих, выполняем диаризацию и более точное разделение сегментов
    if with_diarization:
        try:
            hf_token = "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"  # Ваш токен доступа Hugging Face
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
            diarization = pipeline(file_path)

            # Преобразуем результат диаризации в список интервалов
            diarization_intervals = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarization_intervals.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            # Разбиваем каждый транскрибированный сегмент по пересечениям с диаризационными интервалами
            split_segments = []
            for seg in result["segments"]:
                split_segments.extend(split_segment_by_diarization(seg, diarization_intervals))

            # Объединяем подряд идущие сегменты с одинаковым спикером
            grouped_segments = merge_intervals(sorted(split_segments, key=lambda s: s["start"]), gap_threshold=0.5)

            # Формируем итоговый текст с указанием временных интервалов и спикера
            speaker_text = ""
            for seg in grouped_segments:
                speaker_text += f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}: {seg['text']}\n"

            transcription += "\nИдентификация говорящих с текстом:\n" + speaker_text
        except Exception as e:
            transcription += "\n-- Идентификация говорящих не выполнена. Ошибка: " + str(e)

    return transcription


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    # Сохраняем исходный файл во временную папку
    filename = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Преобразуем аудио в формат WAV с 16000 Гц и моно каналом
    converted_path = convert_audio_to_wav(file_path)

    # Получаем параметры из запроса (чекбоксы приходят в виде строки "true" или "false")
    with_timestamps = request.form.get("timestamps", "false").lower() == "true"
    with_diarization = request.form.get("diarization", "false").lower() == "true"

    # Выполняем транскрипцию, используя преобразованный файл
    transcription = transcribe_audio(converted_path, with_timestamps, with_diarization)

    # Удаляем временные файлы (исходный и преобразованный)
    os.remove(file_path)
    os.remove(converted_path)

    return jsonify({"result": transcription})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
