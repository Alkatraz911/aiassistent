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


def _normalize_word(w: str) -> str:
    return w.strip(" .,!?;:—-\"'«»").lower()


def has_repeating_ngram(text: str, max_n: int = 4, min_repeats: int = 3) -> bool:
    """True, если в тексте одно и то же слово/короткая фраза (1-4 слова) повторяется подряд
    `min_repeats` раз и более — надёжный признак типичной галлюцинации Whisper на коротких/
    шумных клипах («Ветка. Ветка. Ветка...», «я не знаю, я не знаю, я не знаю...»). Проверено на
    реальных примерах из живых записей — обычная речь так не повторяется, а зацикленное
    декодирование почти всегда именно так и выглядит, даже при включённых
    temperature-fallback/compression_ratio_threshold (см. faster_whisper_provider.py — та
    настройка снижает частоту, но не устраняет полностью на очень коротких клипах)."""
    words = [_normalize_word(w) for w in text.split()]
    words = [w for w in words if w]
    n_words = len(words)
    for n in range(1, max_n + 1):
        if n_words < n * min_repeats:
            continue
        i = 0
        while i + n * 2 <= n_words:
            chunk = words[i:i + n]
            repeats = 1
            j = i + n
            while j + n <= n_words and words[j:j + n] == chunk:
                repeats += 1
                j += n
            if repeats >= min_repeats:
                return True
            i += 1
    return False


class ASRProvider:
    """Транскрибирует кусок аудио (float32 PCM, 16 кГц, моно)."""

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
    ) -> ASRResult:  # noqa: D401
        """`beam_size`/`initial_prompt` — необязательные оверрайды на конкретный вызов
        (нужны ASR-планировщику: у partial- и final-заданий разные настройки качества/скорости
        и разный контекст). `None` — использовать дефолт провайдера."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Прогрев модели (опционально)."""
