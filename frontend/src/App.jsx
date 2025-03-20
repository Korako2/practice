// App.js
import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [withTimestamps, setWithTimestamps] = useState(false);
  const [withDiarization, setWithDiarization] = useState(false);
  const [result, setResult] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("timestamps", withTimestamps);
    formData.append("diarization", withDiarization);

    try {
      const response = await fetch("/api/transcribe", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      console.error("Ошибка:", error);
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h2>Загрузка видео/аудиофайла для транскрипции</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Выберите файл: </label>
          <input
            type="file"
            accept="audio/*,video/*"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </div>
        <div>
          <label>
            <input
              type="checkbox"
              checked={withTimestamps}
              onChange={(e) => setWithTimestamps(e.target.checked)}
            />
            Добавить временные метки
          </label>
        </div>
        <div>
          <label>
            <input
              type="checkbox"
              checked={withDiarization}
              onChange={(e) => setWithDiarization(e.target.checked)}
            />
            Применить диаризацию (идентификация говорящих)
          </label>
        </div>
        <button type="submit">Запустить транскрипцию</button>
      </form>
      {result && (
        <div style={{ marginTop: "2rem" }}>
          <h3>Результат транскрипции:</h3>
          <pre>{result}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
