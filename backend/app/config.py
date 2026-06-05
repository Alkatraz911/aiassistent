"""Конфигурация backend через переменные окружения."""
from __future__ import annotations

import os
from pathlib import Path

# Протокол Xet у Hugging Face часто блокируется корпоративными сетями и вешает
# загрузку модели. Принудительно используем классический HTTPS-путь.
# Должно быть выставлено ДО первого импорта huggingface_hub / faster_whisper.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage" / "sessions"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Частота дискретизации, в которой работает весь пайплайн (Whisper ждёт 16 кГц).
SAMPLE_RATE = 16_000

# Выбор движка ASR: "faster_whisper" (боевой) или "stub" (быстрый прогон UI).
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "faster_whisper")

# Параметры faster-whisper.
# medium заметно лучше small по русскому; на GPU — large-v3.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")         # small/medium/large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")          # cpu / cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")       # int8 / float16
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))  # 5 = качество, 1 = скорость
# Подсказка домена смещает распознавание (можно дополнить терминами вашей предметной области).
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "Протокол опроса. Интервьюер и опрашиваемый. Разговорная русская речь.",
)

# VAD-чанкинг: при паузе длиннее порога текущий накопленный фрагмент уходит в ASR.
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "700"))     # тишина для закрытия фразы
# Принудительный сброс делаем реже, чтобы не рвать фразу посреди слова.
VAD_MAX_CHUNK_MS = int(os.getenv("VAD_MAX_CHUNK_MS", "9000"))
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.008"))  # RMS-порог речи
