"""ASR на faster-whisper (CTranslate2). Без torch, работает на CPU и GPU.

GPU-специфика собрана в трёх местах и везде это разные механизмы, а не одна «галочка cuda»:
  • `__init__` — `num_workers`/`device_index` (параллелизм внутри модели, см. config).
  • `_decode` — выбор между обычным и батчевым проходом для длинного аудио.
  • `app/device.py` — чтобы CTranslate2 вообще нашёл cuBLAS/cuDNN на Windows.
"""
from __future__ import annotations

import numpy as np

from .. import config
from .base import ASRProvider, ASRResult, ASRWord, has_repeating_ngram


class FasterWhisperASR(ASRProvider):
    def __init__(self, model: str | None = None, device: str | None = None,
                 compute: str | None = None) -> None:
        # ДО импорта faster_whisper: подмешать в пути поиска DLL каталоги pip-пакетов
        # nvidia-cublas-cu12 / nvidia-cudnn-cu12, иначе на Windows CTranslate2 не найдёт
        # cublas64_12.dll при полностью исправной установке (см. app/device.py).
        from .. import device as device_mod
        device_mod.ensure_cuda_dlls()
        from faster_whisper import WhisperModel

        self.model_name = model or config.WHISPER_MODEL
        self.device = device or config.WHISPER_DEVICE
        self.compute = compute or config.WHISPER_COMPUTE
        self.is_gpu = self.device.startswith("cuda")
        self._batched = None      # BatchedInferencePipeline, лениво (см. _batched_pipeline)
        kwargs = {}
        if self.is_gpu:
            # Какие карты (при нескольких GPU) и сколько параллельных исполнителей внутри модели.
            # Без num_workers>1 конкурентные вызовы transcribe() из разных потоков планировщика
            # сериализуются очередью CTranslate2 — второй канал ждёт первый, хотя карта свободна.
            kwargs["device_index"] = (config.WHISPER_DEVICE_INDEX
                                      if len(config.WHISPER_DEVICE_INDEX) > 1
                                      else (config.WHISPER_DEVICE_INDEX or [0])[0])
            kwargs["num_workers"] = max(1, config.WHISPER_NUM_WORKERS)
        elif self.device == "cpu" and config.WHISPER_CPU_THREADS > 0:
            # По умолчанию ctranslate2 сам разбирает почти все ядра под ОДИН вызов transcribe().
            # При нескольких worker-потоках планировщика (ASR_WORKER_THREADS > 1) это приводит к
            # переподписке: два конкурентных вызова начинают драться за одни и те же ядра и
            # суммарно работают МЕДЛЕННЕЕ, чем по очереди с полным доступом к CPU каждый — это
            # подтвердилось на реальном прогоне (задержка выросла, а не упала, после включения
            # второго воркера). Явно делим ядра между воркерами.
            kwargs["cpu_threads"] = config.WHISPER_CPU_THREADS
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute,
            **kwargs,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
    ) -> ASRResult:
        # faster-whisper принимает float32 моно 16 кГц.
        # Нормализуем громкость: тихий микрофон сильно ухудшает распознавание.
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 1e-4:
                audio = (audio * (0.95 / peak)).astype(np.float32)

        segments, info = self._decode(
            audio,
            language=config.WHISPER_LANGUAGE,
            word_timestamps=True,
            # Встроенный Silero-VAD дочищает тишину по краям чанка -> меньше галлюцинаций.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,   # внутри чанка контекст не нужен
            beam_size=beam_size if beam_size is not None else config.WHISPER_BEAM_SIZE,
            # temperature=0.0 ФИКСИРОВАННЫМ числом отключал встроенный fallback faster-whisper:
            # библиотека сама умеет заметить зацикливание (compression_ratio_threshold — если
            # результат слишком «сжимаемый», т.е. повторяющийся) и повторить попытку на более
            # высокой температуре, ломая петлю. Реальный баг: без списка модель на temp=0
            # зацикливалась («Ветка. Ветка. Ветка...» десятки раз) без единой попытки исправиться.
            temperature=config.WHISPER_TEMPERATURE_FALLBACK,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            # Специально для галлюцинаций после долгих пауз (YouTube-субтитровые концовки вроде
            # «Редактор субтитров...») — подавляет декодирование сегментов, идущих сразу за
            # тишиной длиннее порога, вместо того чтобы пытаться что-то там расслышать.
            hallucination_silence_threshold=config.WHISPER_HALLUCINATION_SILENCE_S,
            initial_prompt=(initial_prompt if initial_prompt is not None
                             else (config.WHISPER_PROMPT or None)),
        )

        words: list[ASRWord] = []
        parts: list[str] = []
        for seg in segments:
            parts.append(seg.text)
            for w in (seg.words or []):
                words.append(
                    ASRWord(text=w.word, start=float(w.start), end=float(w.end),
                            prob=float(w.probability or 1.0))
                )
        text = "".join(parts).strip()

        # Последний рубеж защиты от галлюцинаций-повторов (Блок 3, продолжение): temperature-
        # fallback и hallucination_silence_threshold выше снижают частоту, но реальные тесты
        # показали, что на коротких/шумных клипах модель всё равно иногда зацикливается — причём
        # уверенно (высокий logprob на каждом повторе), так что пороги уверенности это не ловят.
        # Здесь — фактическая проверка результата, а не догадка на входе: если текст выглядит как
        # зацикленный повтор, считаем это тем же, что и «речь не распознана» (пустой результат),
        # а не пропускаем в протокол.
        if has_repeating_ngram(text):
            return ASRResult(text="", words=[], language=info.language)

        return ASRResult(text=text, words=words, language=info.language)

    def _decode(self, audio: np.ndarray, **kwargs):
        """Один проход декодирования: обычный или батчевый.

        Батчевый (`BatchedInferencePipeline`) режет аудио по VAD и считает получившиеся куски
        ОДНИМ батчем. Выигрыш на GPU растёт с длиной куска (замер: 1.2x на 10с, до 2.3x на 77с) —
        чем больше сегментов, тем плотнее заполнен батч. На коротком куске батчить нечего,
        поэтому порог `ASR_BATCH_MIN_S`; на CPU выигрыша нет вовсе — ядра и так загружены.

        Набор параметров у обоих путей одинаковый (в faster-whisper 1.1.0 батчевый
        `transcribe()` отличается только дополнительным `batch_size`), поэтому защиты от
        галлюцинаций — temperature-лестница, `hallucination_silence_threshold`, VAD — действуют
        в обоих режимах одинаково.
        """
        duration_s = audio.size / float(config.SAMPLE_RATE)
        if (config.ASR_BATCH_ENABLED and self.is_gpu
                and duration_s >= config.ASR_BATCH_MIN_S):
            return self._batched_pipeline().transcribe(
                audio, batch_size=config.ASR_BATCH_SIZE, **kwargs)
        return self.model.transcribe(audio, **kwargs)

    def _batched_pipeline(self):
        """Ленивое создание батчевого конвейера — он оборачивает уже загруженную модель и
        отдельных весов не держит, но создавать его на CPU-провайдере незачем.

        Без лока: гонка двух worker-потоков планировщика здесь безобидна (в худшем случае
        создадутся две обёртки над одной моделью, лишняя тут же станет мусором) — в отличие от
        `ModelManager.acquire`, где параллельная загрузка ОДНОЙ модели реально ломалась."""
        if self._batched is None:
            from faster_whisper import BatchedInferencePipeline
            self._batched = BatchedInferencePipeline(model=self.model)
        return self._batched

    def warmup(self) -> None:
        """Прогрев: провести через модель реальный проход декодирования, а не просто загрузить веса.

        Прогрев тишиной (как было) на GPU почти бесполезен: `vad_filter=True` вырезает тишину
        целиком, декодер не запускается ни разу — и первую же настоящую реплику пользователь
        ждёт вместе с ленивой инициализацией CUDA-модулей и автотюном cuDNN. Поэтому гоним шум
        с ЯВНО выключенным VAD: качество результата здесь не важно (он выбрасывается), важно,
        что путь encoder->decoder реально исполнится.

        Ошибки НЕ глотаем (раньше глотали). Типовой отказ GPU — отсутствующая cuBLAS/cuDNN —
        проявляется не при создании `WhisperModel` (оно проходит успешно), а при первом
        декодировании, то есть ровно здесь. Проглоченное исключение означало бы «модель
        загружена и прогрета» в логе старта и падение КАЖДОЙ реальной реплики потом; вместо
        этого пусть преполёт (`main.py::_preflight`) увидит отказ и откатится на CPU.
        """
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(config.SAMPLE_RATE) * 0.05).astype(np.float32)
        segments, _ = self.model.transcribe(
            noise, language=config.WHISPER_LANGUAGE, vad_filter=False,
            beam_size=1, without_timestamps=True,
        )
        list(segments)              # transcribe ленив: без обхода генератора decode не случится
