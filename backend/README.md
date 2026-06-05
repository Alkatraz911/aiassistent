# Backend — Протокол-ассистент

FastAPI + faster-whisper. Канальная диаризация (спикер = микрофон).

## Запуск

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Переменные окружения

| Переменная        | По умолчанию     | Назначение |
|-------------------|------------------|------------|
| `ASR_PROVIDER`    | `faster_whisper` | `stub` — прогон UI без модели |
| `WHISPER_MODEL`   | `small`          | `small`/`medium`/`large-v3` |
| `WHISPER_DEVICE`  | `cpu`            | `cuda` на GPU-сервере |
| `WHISPER_COMPUTE` | `int8`           | `float16` на GPU |
| `WHISPER_LANGUAGE`| `ru`             | язык распознавания |
| `VAD_SILENCE_MS`  | `600`            | пауза, закрывающая фразу |
| `VAD_MAX_CHUNK_MS`| `4000`           | макс. чанк (контроль задержки ≤5 c) |

Быстрый прогон UI без модели:
```powershell
$env:ASR_PROVIDER="stub"; uvicorn app.main:app --port 8000
```

## Smoke-тест пайплайна (без сети/UI)

```powershell
$env:ASR_PROVIDER="stub"; py -3.11 -m app.smoke
```

## API

- `WS /ws/stream/{session}` — стриминг PCM (кадр: int32 канал + int16 сэмплы 16кГц)
- `POST /api/transcribe` — разовая транскрипция (raw int16) для ответов анкеты
- `POST /api/assistant/{start,answer,next}` — сценарий анкеты
- `POST /api/speaker` — маркировка спикера канала
- `POST /api/segment/edit` — правка текста сегмента (с аудитом)
- `POST /api/save` — сохранить протокол (JSON)
- `GET  /api/protocol/{session}` / `GET /api/audio/{session}` — протокол / фонограмма
