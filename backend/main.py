from flask import Flask, request, jsonify
import os
import uuid
import time
import torch
import whisperx
from flask_cors import CORS
from pyannote.audio import Pipeline

app = Flask(__name__)
CORS(app)

# Определяем устройство: GPU (если доступно) или CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель Whisper (tiny, base, small, medium, large)
model = whisperx.load_model("medium", device, compute_type="float32")

# Папка для временного хранения загруженных файлов
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def merge_intervals(segments, gap_threshold=1.0):
    """
    Объединяет соседние интервалы, если они принадлежат одному спикеру
    и разрыв между ними меньше gap_threshold (в секундах).
    Каждый сегмент — словарь с ключами: 'start', 'end', 'speaker', 'text'.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        # Если тот же спикер и разрыв между последним окончанием и текущим началом меньше порога
        if seg["speaker"] == last["speaker"] and (seg["start"] - last["end"] <= gap_threshold):
            # Обновляем конец последнего сегмента
            last["end"] = seg["end"]
            # Если в текущем сегменте есть текст, добавляем его через пробел
            if seg["text"]:
                # Добавляем разделитель, если последний сегмент уже не заканчивается на пробел
                if last["text"] and not last["text"].endswith(" "):
                    last["text"] += " "
                last["text"] += seg["text"]
        else:
            merged.append(seg)
    return merged


def transcribe_audio(file_path, with_timestamps=False, with_diarization=False):
    """
    Выполняет транскрипцию аудиофайла с использованием модели WhisperX.
    Если запрошена идентификация говорящих, для каждого транскрибированного сегмента определяется
    спикер, исходя из того, в какой интервал из pyannote попадает середина сегмента.
    Затем соседние сегменты с одинаковым спикером объединяются.
    """
    start_time = time.time()
    # Расшифровка аудио (язык "ru")
    result = model.transcribe(file_path, language="ru")
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

    # Если запрошена идентификация говорящих, назначаем спикера каждому сегменту
    if with_diarization:
        try:
            hf_token = "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"  # Ваш токен доступа Hugging Face
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            diarization = pipeline(file_path)

            # Преобразуем результат диаризации в список интервалов
            diarization_intervals = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarization_intervals.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            # Функция для поиска спикера по моменту времени (используем середину сегмента)
            def find_speaker(mid_time):
                for interval in diarization_intervals:
                    if interval["start"] <= mid_time < interval["end"]:
                        return interval["speaker"]
                return "UNKNOWN"

            # Назначаем каждому транскрибированному сегменту спикера.
            # Если для сегмента нет точного конца, определим его как начало следующего или +1 сек.
            assigned_segments = []
            segments = result["segments"]
            for i, seg in enumerate(segments):
                seg_start = seg.get("start", 0)
                if "end" in seg:
                    seg_end = seg["end"]
                else:
                    seg_end = segments[i + 1]["start"] if i + 1 < len(segments) else seg_start + 1.0
                mid_time = (seg_start + seg_end) / 2
                speaker = find_speaker(mid_time)
                assigned_segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": speaker,
                    "text": seg["text"].strip()
                })

            # Группируем подряд идущие сегменты с одинаковым спикером
            grouped_segments = []
            if assigned_segments:
                current = assigned_segments[0].copy()
                for seg in assigned_segments[1:]:
                    # Если спикер тот же и разрыв между сегментами небольшой (например, <= 0.5 сек), объединяем
                    if seg["speaker"] == current["speaker"] and seg["start"] - current["end"] <= 0.5:
                        current["end"] = seg["end"]
                        current["text"] += " " + seg["text"]
                    else:
                        grouped_segments.append(current)
                        current = seg.copy()
                grouped_segments.append(current)

            # Формируем итоговый текст с указанием интервалов и соответствующего текста
            speaker_text = ""
            for seg in grouped_segments:
                speaker_text += f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']} {seg['text']}\n"

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
