"""Накопитель аудио по каналу с простым энергетическим VAD.

Логика псевдо-стриминга: копим сэмплы; как только встретили достаточно длинную
паузу (тишину) ПОСЛЕ речи — отдаём накопленную фразу на ASR. Чтобы задержка не
превышала ~5 c, есть принудительный сброс по максимальной длине чанка.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config


@dataclass
class ChunkBuffer:
    sample_rate: int = config.SAMPLE_RATE
    _samples: list[np.ndarray] = field(default_factory=list)
    _n: int = 0                      # накоплено сэмплов
    _silence: int = 0               # подряд тишины (в сэмплах)
    _had_speech: bool = False
    chunk_start_sample: int = 0      # абсолютная позиция начала чанка в фонограмме
    _total_samples: int = 0         # всего получено по каналу (для тайм-кодов)

    def _rms(self, frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def add(self, pcm: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Добавляет сэмплы, возвращает список готовых чанков.

        Каждый чанк = (audio float32, offset_seconds от начала фонограммы).
        """
        ready: list[tuple[np.ndarray, float]] = []
        self._samples.append(pcm)
        self._n += len(pcm)
        self._total_samples += len(pcm)

        # оценка речь/тишина по этому фрейму
        if self._rms(pcm) >= config.VAD_ENERGY_THRESHOLD:
            self._had_speech = True
            self._silence = 0
        else:
            self._silence += len(pcm)

        silence_limit = int(config.VAD_SILENCE_MS * self.sample_rate / 1000)
        max_limit = int(config.VAD_MAX_CHUNK_MS * self.sample_rate / 1000)

        closed_by_silence = self._had_speech and self._silence >= silence_limit
        closed_by_size = self._n >= max_limit

        if closed_by_silence or closed_by_size:
            chunk = self._flush()
            if chunk is not None:
                ready.append(chunk)
        return ready

    def _flush(self) -> tuple[np.ndarray, float] | None:
        if self._n == 0 or not self._had_speech:
            self._reset_chunk()
            return None
        audio = np.concatenate(self._samples).astype(np.float32)
        offset = self.chunk_start_sample / self.sample_rate
        self._reset_chunk()
        return audio, offset

    def _reset_chunk(self) -> None:
        self.chunk_start_sample = self._total_samples
        self._samples = []
        self._n = 0
        self._silence = 0
        self._had_speech = False

    def finalize(self) -> tuple[np.ndarray, float] | None:
        """Сброс остатка при остановке записи."""
        return self._flush()
