"""Непрерывный скользящий буфер аудио по каналу (Блок 2.1 плана).

В отличие от старого `ChunkBuffer` (см. `endpointer.py`, куда перешла его роль эндпоинтинга),
этот буфер НЕ решает, когда реплика закончилась — он просто копит raw PCM непрерывно, независимо
от границ реплик, и по запросу отдаёт «окно с учётом уже закоммиченной границы» для очередного
partial/final ASR-прохода. Разделение ролей: `Endpointer` решает «когда», `RollingBuffer` даёт
«что именно распознавать».
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config


@dataclass
class RollingBuffer:
    sample_rate: int = config.SAMPLE_RATE
    window_ms: int = config.ASR_WINDOW_MS
    lookback_ms: int = config.ASR_LOOKBACK_MS
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _start_sample: int = 0     # абсолютный индекс сэмпла, соответствующего _buf[0]
    _total_samples: int = 0    # абсолютный индекс следующего входящего сэмпла

    def add(self, pcm: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, pcm]) if self._buf.size else np.asarray(pcm, dtype=np.float32).copy()
        self._total_samples += len(pcm)
        self._trim()

    def _trim(self, keep_extra_ms: int = 2000) -> None:
        max_samples = (self.window_ms + keep_extra_ms) * self.sample_rate // 1000
        if len(self._buf) > max_samples:
            drop = len(self._buf) - max_samples
            self._buf = self._buf[drop:]
            self._start_sample += drop

    def window_since(self, committed_sample: int) -> tuple[np.ndarray, int]:
        """Аудио от `max(буфер_старт, committed_sample - lookback)` до текущего конца буфера.
        Возвращает (audio, window_start_sample) — начало окна в абсолютных сэмплах, чтобы
        вызывающий мог восстановить абсолютные тайм-коды слов из ASR-результата."""
        lookback = self.lookback_ms * self.sample_rate // 1000
        start = max(self._start_sample, committed_sample - lookback)
        start_idx = start - self._start_sample
        return self._buf[start_idx:], start

    def new_audio_ms_since(self, last_sample: int) -> float:
        return max(0, self._total_samples - last_sample) * 1000.0 / self.sample_rate

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def reset(self) -> None:
        """Сбросить буфер (например, после закрытия реплики — следующая начинается с чистого
        `committed_boundary`, старое аудио больше не нужно для контекста decode)."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._start_sample = self._total_samples
