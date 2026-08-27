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
TEMPLATES_DIR = BASE_DIR / "storage" / "templates"      # шаблоны анкеты (Блок 5)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Частота дискретизации, в которой работает весь пайплайн (Whisper ждёт 16 кГц).
SAMPLE_RATE = 16_000

# Выбор движка ASR: "faster_whisper" (боевой) или "stub" (быстрый прогон UI).
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "faster_whisper")

# Параметры faster-whisper.
# small — компромисс, реально пригодный для live-стриминга на CPU (см. ASR_OVERLOAD_LAG_MS
# и backend/README.md «Производительность»): на измеренном CPU medium/turbo/large-v3 декодируют
# МЕДЛЕННЕЕ реального времени (0.7x/0.5x), из-за чего задержка финала растёт без ограничения при
# двух одновременных каналах. medium/turbo/large-v3 стоит использовать на GPU или для офлайн
# `/api/finalize` (уже есть выбор модели из UI, Блок 4 — можно переключиться на сессию вручную).
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")          # small/medium/turbo/large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")          # cpu / cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")       # int8 / float16
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))  # 5 = качество, 1 = скорость
# Лестница температур для fallback при зацикливании/низкой уверенности (compression_ratio /
# logprob threshold в faster-whisper) — ОБЯЗАТЕЛЬНО список, не одно число: одно фиксированное
# число отключает встроенный защитный механизм от повторов (реальный баг, см. faster_whisper_provider.py).
WHISPER_TEMPERATURE_FALLBACK = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Порог тишины (сек) перед сегментом, после которого декодирование подавляется как вероятная
# галлюцинация (типично — «Редактор субтитров...», повторы) — реальный баг, вылезающий при
# длинных паузах в речи между репликами.
WHISPER_HALLUCINATION_SILENCE_S = float(os.getenv("WHISPER_HALLUCINATION_SILENCE_S", "2.0"))

# Офлайн forced-alignment (без токенов): русский wav2vec2 для точных тайм-кодов слов.
ALIGN_MODEL = os.getenv("ALIGN_MODEL", "jonatasgrosman/wav2vec2-large-xlsr-53-russian")

# Офлайн-диаризация для одного общего микрофона (без токенов): ECAPA + кластеризация.
# Порог косинусного расстояния для авто-режима (когда число голосов не задано).
DIARIZE_THRESHOLD = float(os.getenv("DIARIZE_THRESHOLD", "0.55"))
# Подсказка домена смещает распознавание (можно дополнить терминами вашей предметной области).
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "Протокол опроса. Интервьюер и опрашиваемый. Разговорная русская речь.",
)

# Кросс-канальный гейтинг (основной режим, микрофон-на-участника).
# В одном помещении микрофоны слышат друг друга; «протёкший» голос подавляем,
# отдавая речь каналу с максимальной энергией. Это делает «канал = спикер» надёжным.
CROSSTALK_ENABLED = os.getenv("CROSSTALK_ENABLED", "1") == "1"
CROSSTALK_WINDOW_MS = int(os.getenv("CROSSTALK_WINDOW_MS", "250"))   # окно сравнения каналов
CROSSTALK_MARGIN_DB = float(os.getenv("CROSSTALK_MARGIN_DB", "5.0"))  # насколько громче должен быть «хозяин»
GATE_NOISE_FLOOR = float(os.getenv("GATE_NOISE_FLOOR", "0.004"))     # ниже — считаем тишиной/шумом
# Подавление активируется только после того, как другой канал непрерывно громче это время —
# не с первого кадра. Реальный тест показал: при близко расположенных микрофонах разница между
# каналами часто «мигает» в пределах пары дБ кадр от кадра — takeover сглаживает этот дребезг и
# не режет короткие «да»/«угу» в начале.
CROSSTALK_TAKEOVER_MS = int(os.getenv("CROSSTALK_TAKEOVER_MS", "180"))
# Порог bleed_score (0..1), после которого кадр считается протечкой для маршрутизации в ASR
# (НЕ для физической записи — см. Блок 0.4, raw всегда пишется).
CROSSTALK_SCORE_THRESHOLD = float(os.getenv("CROSSTALK_SCORE_THRESHOLD", "0.5"))

# Пост-ASR дедупликация протечки (Блок 3.7) — сравнение уже распознанного текста, а не сырой
# энергии, поэтому надёжнее при близко расположенных микрофонах.
CROSSTALK_DEDUPE_MIN_SIMILARITY = float(os.getenv("CROSSTALK_DEDUPE_MIN_SIMILARITY", "0.6"))
CROSSTALK_DEDUPE_SHORT_WORDS = int(os.getenv("CROSSTALK_DEDUPE_SHORT_WORDS", "2"))
CROSSTALK_DEDUPE_SHORT_SNR_GAP_DB = float(os.getenv("CROSSTALK_DEDUPE_SHORT_SNR_GAP_DB", "8.0"))

# Эндпоинтинг (Endpointer, backend/app/audio/endpointer.py): решает, когда реплика
# закончилась. Не путать с частотой ASR-обновлений (ASR_UPDATE_MS ниже) — это два
# независимых механизма, раньше слитых в один VAD_SILENCE_MS/VAD_MAX_CHUNK_MS.
ENDPOINT_SILENCE_MS = int(os.getenv("ENDPOINT_SILENCE_MS", "600"))   # тишина закрывает реплику
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "180"))               # защита от случайных всплесков
# Верхний санитарный предел на длительность одной необорванной реплики. Это НЕ механизм
# удержания задержки (задержку держат partial-апдейты, ASR_UPDATE_MS) — только защита от
# бесконечно открытой реплики, если пауз в речи не было вообще очень долго.
MAX_UTTERANCE_MS = int(os.getenv("MAX_UTTERANCE_MS", "60000"))
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.008"))  # минимальный пол шума
# Насколько кадр должен быть громче адаптивного пола шума (NoiseFloorTracker), чтобы считаться
# речью. Ниже — используем более жёсткий Silero VAD (Этап 2 плана, опционально, см. README).
VAD_SPEECH_MARGIN_DB = float(os.getenv("VAD_SPEECH_MARGIN_DB", "6.0"))

# Real-time streaming ASR: частота partial-обновлений текста, независимая от эндпоинтинга.
ASR_UPDATE_MS = int(os.getenv("ASR_UPDATE_MS", "900"))         # целевой каданс (не жёсткий таймер)
ASR_WINDOW_MS = int(os.getenv("ASR_WINDOW_MS", "12000"))       # окно RollingBuffer для partial-decode
ASR_LOOKBACK_MS = int(os.getenv("ASR_LOOKBACK_MS", "2000"))    # контекст до committed_boundary
WHISPER_BEAM_SIZE_PARTIAL = int(os.getenv("WHISPER_BEAM_SIZE_PARTIAL", "1"))  # партиалы — быстрее

# Кэш ASR-моделей (backend/app/asr/model_manager.py) — сколько моделей одновременно держим
# в памяти при переключении из UI (Блок 4 плана).
ASR_MAX_CACHED_MODELS = int(os.getenv("ASR_MAX_CACHED_MODELS", "2"))

# Сколько потоков планировщика (backend/app/asr/scheduler.py) параллельно вызывают
# ASRProvider.transcribe(). >1 позволяет каналам декодироваться по-настоящему одновременно —
# реальный тест с 2 каналами показал, что 1 поток не успевает за темпом partial-заданий и
# задержка финала растёт без ограничения (см. историю правок session.py/scheduler.py).
ASR_WORKER_THREADS = int(os.getenv("ASR_WORKER_THREADS", "2"))

# Сколько CPU-потоков ctranslate2 использует НА ОДИН вызов transcribe() (только для CPU-режима).
# 0 — не ограничивать явно (отдать на откуп ctranslate2, обычно почти все ядра). При
# ASR_WORKER_THREADS>1 несколько неограниченных вызовов конкурируют за одни и те же ядра и
# суммарно работают медленнее, чем по очереди — реальный тест это подтвердил. Дефолт делит ядра
# поровну между воркерами планировщика.
WHISPER_CPU_THREADS = int(os.getenv(
    "WHISPER_CPU_THREADS", str(max(1, (os.cpu_count() or 4) // max(1, ASR_WORKER_THREADS)))))

# Если среднее время partial-раунда (submit->result) превышает это значение — модель на этом
# железе фундаментально не успевает за реальным временем (наблюдалось на CPU: decode ~10с на
# 5с аудио на turbo). В этом состоянии partial-задания перестают ставиться совсем — они бы
# только отнимали ёмкость воркеров у финалов и устаревали раньше, чем дошли бы до клиента.
ASR_OVERLOAD_LAG_MS = float(os.getenv("ASR_OVERLOAD_LAG_MS", "4000"))
