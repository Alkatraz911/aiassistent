"""ASR на faster-whisper (CTranslate2). Без torch, работает на CPU и GPU."""
from __future__ import annotations

import numpy as np

from .. import config
from .base import ASRProvider, ASRResult, ASRWord


class FasterWhisperASR(ASRProvider):
    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
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
            beam_size=config.WHISPER_BEAM_SIZE,
            temperature=0.0,
            no_speech_threshold=0.6,
            initial_prompt=config.WHISPER_PROMPT or None,
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
        return ASRResult(text="".join(parts).strip(), words=words, language=info.language)

    def warmup(self) -> None:
        silence = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
        try:
            self.transcribe(silence, config.SAMPLE_RATE)
        except Exception:
            pass
