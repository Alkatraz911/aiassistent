"""ASR на faster-whisper (CTranslate2). Без torch, работает на CPU и GPU."""
from __future__ import annotations

import numpy as np

from .. import config
from .base import ASRProvider, ASRResult, ASRWord, has_repeating_ngram


class FasterWhisperASR(ASRProvider):
    def __init__(self, model: str | None = None, device: str | None = None,
                 compute: str | None = None) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model or config.WHISPER_MODEL
        self.device = device or config.WHISPER_DEVICE
        self.compute = compute or config.WHISPER_COMPUTE
        kwargs = {}
        if self.device == "cpu" and config.WHISPER_CPU_THREADS > 0:
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

        segments, info = self.model.transcribe(
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

    def warmup(self) -> None:
        silence = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
        try:
            self.transcribe(silence, config.SAMPLE_RATE)
        except Exception:
            pass
