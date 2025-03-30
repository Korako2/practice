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
      const response = await fetch("http://127.0.0.1:5000/api/transcribe", {
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
    <div className="main-container">
      <h1 className="title">Аудио Транскрипция</h1>
      <div className="form-wrapper">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-group">
            <label className="file-label">
              <input
                type="file"
                accept="audio/*,video/*"
                onChange={(e) => setFile(e.target.files[0])}
                className="file-input"
              />
              <span className="custom-file-upload">
                Выберите аудио/видео файл
              </span>
            </label>
            {file && (
              <div className="file-info">
                Файл выбран: <strong>{file.name}</strong>
              </div>
            )}
          </div>

          <div className="checkbox-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={withTimestamps}
                onChange={(e) => setWithTimestamps(e.target.checked)}
                className="checkbox-input"
              />
              <span className="checkmark"></span>
              Временные метки
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={withDiarization}
                onChange={(e) => setWithDiarization(e.target.checked)}
                className="checkbox-input"
              />
              <span className="checkmark"></span>
              Идентификация говорящих
            </label>
          </div>

          <button type="submit" className="submit-btn">
            Начать транскрипцию
          </button>
        </form>
      </div>

      {result && (
        <div className="result-container">
          <h2 className="result-title">Результат транскрипции:</h2>
          <pre className="result-content">{result}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
