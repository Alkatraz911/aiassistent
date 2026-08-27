"""Заглушка ASR — для проверки UI/пайплайна без модели и без GPU.

Возвращает псевдотекст с правдоподобными word-тайм-кодами по длине аудио,
чтобы можно было прокликать маркировку спикеров и привязку аудио↔текст.
"""
from __future__ import annotations

import random

import numpy as np

from .base import ASRProvider, ASRResult, ASRWord

_PHRASES = [
    "да я понимаю",
    "поясните пожалуйста ещё раз",
    "это было примерно в восемь часов вечера",
    "нет я там не был",
    "хорошо записывайте",
    "мне нужно подумать над ответом",
]


class StubASR(ASRProvider):
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
    ) -> ASRResult:
        dur = max(0.4, len(audio) / sample_rate)
        # тихий фрагмент считаем тишиной
        if float(np.sqrt(np.mean(audio ** 2)) if len(audio) else 0.0) < 1e-4:
            return ASRResult(text="", words=[])

        phrase = random.choice(_PHRASES)
        tokens = phrase.split()
        step = dur / len(tokens)
        words = [
            ASRWord(text=(" " + t), start=round(i * step, 2),
                    end=round((i + 1) * step, 2), prob=0.9)
            for i, t in enumerate(tokens)
        ]
        return ASRResult(text=phrase, words=words)
