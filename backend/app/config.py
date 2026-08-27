"""Конфигурация backend через переменные окружения."""
from __future__ import annotations

import os
from pathlib import Path

from . import device

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

# Устройство. "auto" (дефолт) — cuda, если CTranslate2 её видит, иначе cpu. Явные "cuda"/"cpu"
# берутся как есть, без проверки: догадываться за пользователя тут нечего. Если GPU при этом не
# поднимется, преполёт (`main.py::_preflight`) откатит на CPU — но с причиной в `/api/health`,
# а не молча.
WHISPER_DEVICE_REQUESTED = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_DEVICE = device.resolve_device(WHISPER_DEVICE_REQUESTED)
IS_GPU = WHISPER_DEVICE.startswith("cuda")
# Какие карты использовать: "0" или "0,1" при нескольких GPU (CTranslate2 device_index).
WHISPER_DEVICE_INDEX = [int(x) for x in os.getenv("WHISPER_DEVICE_INDEX", "0").split(",") if x.strip()]

# Параметры faster-whisper.
# Дефолт модели зависит от устройства — это разные режимы работы, а не одна настройка:
#   • CPU: small — компромисс, реально пригодный для live-стриминга (см. ASR_OVERLOAD_LAG_MS и
#     backend/README.md «Производительность»): на измеренном CPU medium/turbo/large-v3 декодируют
#     МЕДЛЕННЕЕ реального времени (0.7x/0.5x), из-за чего задержка финала растёт без ограничения
#     при двух одновременных каналах.
#   • GPU: large-v3 — на GPU тяжёлая модель идёт кратно быстрее реального времени, и держать
#     small там незачем: это была уступка слабому железу, а не выбор по качеству.
# Память: large-v3 в float16 занимает ~3 ГБ VRAM; при ASR_MAX_CACHED_MODELS=2 и переключении
# моделей из UI держатся две сразу — на картах с 8 ГБ и меньше ставьте ASR_MAX_CACHED_MODELS=1.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3" if IS_GPU else "small")
# На какую модель откатываться, если GPU просили, а он не поднялся (см. main.py::_preflight):
# large-v3 на CPU не тянет live вовсе, поэтому откат — это откат и по устройству, и по модели.
WHISPER_MODEL_CPU_FALLBACK = os.getenv("WHISPER_MODEL_CPU_FALLBACK", "small")
# int8 — для CPU (там это главный ускоритель). На GPU float16 и быстрее, и точнее int8: тензорные
# ядра считают fp16 нативно, а int8 на GPU требует квантования с потерей качества без выигрыша.
# int8_float16 — компромисс для карт с малым VRAM (модель весит вдвое меньше, скорость близка).
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16" if IS_GPU else "int8")
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
# Устройство для офлайн-финализации (alignment + диаризация). Это torch, а не CTranslate2, —
# отдельная переменная: CUDA-сборка torch ставится отдельно от CUDA-библиотек faster-whisper,
# и вполне рабочая конфигурация — ASR на GPU, а finalize на CPU (torch остался CPU-сборкой).
# "auto" — cuda, если torch её видит, иначе cpu (проверяется лениво, при первой загрузке модели).
FINALIZE_DEVICE = os.getenv("FINALIZE_DEVICE", "auto")
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
# Это НИЖНЯЯ ГРАНИЦА интервала, а не обещание: session.py берёт max(ASR_UPDATE_MS, lag_ema*1.3),
# поэтому реальный каданс всё равно диктует скорость декодирования. Замер на GPU (RTX 3080,
# два канала, окно 12с) показывает, что решает именно ВЫБОР МОДЕЛИ, а не эта переменная:
#     large-v3 -> раунд ~1.2с (400 мс недостижимы, каданс растянется до ~1.6с)
#     turbo    -> раунд ~0.5с (вот здесь 400 мс уже работают)
# Отсюда и значение: на GPU держим границу низкой, чтобы лёгкая модель могла обновлять текст так
# часто, как реально успевает, а тяжёлая просто упёрлась в своё время декодирования. На CPU
# низкая граница смысла не имеет — там не успевает ни одна модель.
ASR_UPDATE_MS = int(os.getenv("ASR_UPDATE_MS", "400" if IS_GPU else "900"))
ASR_WINDOW_MS = int(os.getenv("ASR_WINDOW_MS", "12000"))       # окно RollingBuffer для partial-decode
ASR_LOOKBACK_MS = int(os.getenv("ASR_LOOKBACK_MS", "2000"))    # контекст до committed_boundary
# Партиалы считаем beam=1 (жадно) — и на CPU, и на GPU. Соблазн поднять beam на GPU («там же
# быстро») проверен и отвергнут: на large-v3, окно 12с, beam 1/2/3/5 дал 0.63/0.77/0.78/0.80с
# при ОДИНАКОВОМ распознанном тексте. То есть +22% к задержке партиала за неподтверждённую
# надежду на более стабильную гипотезу. Партиал по определению черновик — его уточняет финал
# (WHISPER_BEAM_SIZE=5), а до тех пор дешевле обновить его ещё раз, чем считать точнее.
WHISPER_BEAM_SIZE_PARTIAL = int(os.getenv("WHISPER_BEAM_SIZE_PARTIAL", "1"))

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

# Число параллельных исполнителей ВНУТРИ модели (CTranslate2 num_workers). Напрашивающийся
# GPU-аналог деления ядер: пусть два канала декодируются одновременно, а не встают в очередь
# CTranslate2 гуськом. Замер это опроверг — на RTX 3080 / large-v3, два канала по окну 12с:
#     num_workers=1 -> wall 1.67с   num_workers=2 -> 2.04с   num_workers=3 -> 2.15с
# То есть чем больше «параллелизма», тем МЕДЛЕННЕЕ. Причина: одна большая модель уже насыщает
# карту, свободных SM под второй поток нет, и настоящая одновременность даёт только накладные
# расходы на переключение и вытеснение кэша — тогда как последовательное исполнение той же
# работы идеально эффективно. Это ровно тот же вывод, что и на CPU (WHISPER_CPU_THREADS выше),
# просто по другой причине, и он НЕ переносится автоматически на лёгкую модель + большую карту:
# там карта может быть недогружена, и 2 способны выиграть. Проверять — `py -3.11 -m app.test_gpu`.
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))

# Батчинг длинного аудио (faster-whisper BatchedInferencePipeline). Длинный кусок нарезается по
# VAD на сегменты, и они декодируются ОДНИМ батчем. Замер на GPU (large-v3, RTX 3080) — выигрыш
# растёт с длиной куска, потому что растёт число сегментов, которые есть чем заполнить батч:
#     10с -> 1.2x     30с -> 1.4-1.9x     77с -> 2.3x
# Отсюда и порог: короче ASR_BATCH_MIN_S батчить нечего (один-два сегмента), а VAD-нарезку
# оплачивать всё равно придётся. На CPU выигрыша нет вовсе — ядра и так загружены.
# Партиалы (окно 12с, каждые ASR_UPDATE_MS) под порог не попадают по построению: у них приоритет —
# задержка одного прохода, а не пропускная способность.
ASR_BATCH_ENABLED = os.getenv("ASR_BATCH_ENABLED", "1" if IS_GPU else "0") == "1"
ASR_BATCH_MIN_S = float(os.getenv("ASR_BATCH_MIN_S", "20.0"))   # короче — обычный проход
# 8 — с запасом: на 77с замер дал 2.08/2.02/2.01с для batch 4/8/16, то есть выигрыш даёт сам
# факт батчинга, а не размер. Больше batch_size — только больше VRAM без ускорения.
ASR_BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE", "8"))

# Если среднее время partial-раунда (submit->result) превышает это значение — модель на этом
# железе фундаментально не успевает за реальным временем (наблюдалось на CPU: decode ~10с на
# 5с аудио на turbo). В этом состоянии partial-задания перестают ставиться совсем — они бы
# только отнимали ёмкость воркеров у финалов и устаревали раньше, чем дошли бы до клиента.
# На GPU порог ниже — но НЕ настолько, насколько хочется. Замер: у дефолтной large-v3 раунд
# partial при двух одновременно говорящих каналах занимает ~1.2с, то есть заманчивые «1.5с»
# отключали бы партиалы ровно в момент активного диалога — там, где живой текст нужнее всего.
# 2.5с оставляет запас над нормальным раундом тяжёлой модели и всё равно реагирует заметно
# раньше CPU-порога. Если ставите модель полегче (turbo), порог можно опустить следом.
ASR_OVERLOAD_LAG_MS = float(os.getenv("ASR_OVERLOAD_LAG_MS", "2500" if IS_GPU else "4000"))
