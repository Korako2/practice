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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель WhisperX (варианты: tiny, base, small, medium, large)
MODEL = whisperx.load_model("tiny", DEVICE, compute_type="float32")
print(f"Используемая модель WhisperX: {MODEL}")


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
        "-ar", "16000",          # частота дискретизации 16000 Гц
        "-ac", "1",              # один аудио канал (моно)
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def split_segment_by_diarization(segment, diarization_intervals):
    """
    Разбивает транскрибированный сегмент по говорящим, основываясь на пересечении с диаризацией.
    Назначает сегменту спикера с наибольшим пересечением.
    """
    seg_start = segment.get("start", 0)
    seg_end = segment.get("end", seg_start + 1.0)
    best_match = None
    max_overlap = 0

    for d in diarization_intervals:
        overlap_start = max(seg_start, d["start"])
        overlap_end = min(seg_end, d["end"])
        overlap = max(0, overlap_end - overlap_start)

        if overlap > max_overlap:
            max_overlap = overlap
            best_match = d["speaker"]

    speaker = best_match if best_match else "UNKNOWN"
    return [{
        "start": seg_start,
        "end": seg_end,
        "speaker": speaker,
        "text": segment["text"]
    }]


def merge_intervals(segments, gap_threshold=0.5):
    """
    Объединяет подряд идущие сегменты с одним спикером, корректно объединяя текст и временные метки.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and (seg["start"] - last["end"] <= gap_threshold):
            if seg["text"] not in last["text"]:
                if last["text"] and not last["text"].endswith(" "):
                    last["text"] += " "
                last["text"] += seg["text"]
            last["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def perform_diarization(file_path):
    """
    Выполняет диаризацию аудио с использованием модели pyannote.
    Возвращает список интервалов с указанием говорящего.
    """
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Токен HUGGINGFACE_TOKEN не установлен в переменных окружения")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization = pipeline(file_path)
    diarization_intervals = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_intervals.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return diarization_intervals


def build_transcription_text(result, elapsed_time, with_timestamps):
    """
    Формирует итоговый текст транскрипции из результатов WhisperX.
    """
    full_text = " ".join(segment["text"] for segment in result["segments"])
    transcription = f"Полный текст:\n{full_text}\n\nВремя транскрипции: {elapsed_time:.2f} секунд"

    if with_timestamps:
        timestamps_text = ""
        for segment in result["segments"]:
            timestamps_text += f"[{segment['start']:.2f}] {segment['text']}\n"
        transcription += "\nВременные метки:\n" + timestamps_text

    return transcription


def append_diarization(transcription, result, file_path):
    """
    Добавляет в текст транскрипции информацию о диаризации.
    """
    try:
        diarization_intervals = perform_diarization(file_path)
        split_segments = []
        for seg in result["segments"]:
            split_segments.extend(split_segment_by_diarization(seg, diarization_intervals))
        grouped_segments = merge_intervals(sorted(split_segments, key=lambda s: s["start"]), gap_threshold=0.5)

        speaker_text = ""
        for seg in grouped_segments:
            speaker_text += f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}: {seg['text']}\n"
        transcription += "\nИдентификация говорящих с текстом:\n" + speaker_text
    except Exception as e:
        transcription += "\n-- Идентификация говорящих не выполнена. Ошибка: " + str(e)
    return transcription


def transcribe_audio(file_path, with_timestamps=False, with_diarization=False):
    """
    Выполняет транскрипцию аудиофайла с использованием модели WhisperX.
    При необходимости добавляет временные метки и диаризацию.
    """
    start_time = time.time()
    result = MODEL.transcribe(file_path, language="ru", chunk_size=10)
    elapsed_time = time.time() - start_time

    transcription = build_transcription_text(result, elapsed_time, with_timestamps)

    if with_diarization:
        transcription = append_diarization(transcription, result, file_path)

    return transcription


def save_uploaded_file(file):
    """
    Сохраняет загруженный файл во временную папку и возвращает его путь.
    """
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    return file_path


def cleanup_files(*file_paths):
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    # Проверка наличия файла в запросе
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    original_file_path = save_uploaded_file(file)
    converted_file_path = convert_audio_to_wav(original_file_path)

    with_timestamps = request.form.get("timestamps", "false").lower() == "true"
    with_diarization = request.form.get("diarization", "false").lower() == "true"

    transcription = transcribe_audio(converted_file_path, with_timestamps, with_diarization)

    cleanup_files(original_file_path, converted_file_path)

    return jsonify({"result": transcription})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
