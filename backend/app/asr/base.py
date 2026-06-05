"""Базовый интерфейс ASR — провайдер-агностичный слой.

Меняя реализацию (faster-whisper, NeMo/Riva на боевом сервере, облако),
остальной код не трогаем.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ASRWord:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class ASRResult:
    text: str
    words: list[ASRWord] = field(default_factory=list)
    language: str = "ru"


class ASRProvider:
    """Транскрибирует кусок аудио (float32 PCM, 16 кГц, моно)."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:  # noqa: D401
        raise NotImplementedError

    def warmup(self) -> None:
        """Прогрев модели (опционально)."""
