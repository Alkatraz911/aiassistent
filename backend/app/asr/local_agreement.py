"""LocalAgreement-2: стабилизация partial-текста между последовательными ASR-раундами
(Блок 2.3 плана).

Идея (whisper_streaming / SimulStreaming): сравнить хвост текущего decode с хвостом предыдущего
decode того же `utterance_id` (word-level, после нормализации регистра/пунктуации), взять
наибольший общий префикс — это «stable» часть, которую больше не должно переписать (кроме
последних `REVISION_HOLDBACK_WORDS` слов — они сознательно ещё не коммитятся, т.к. чаще всего
именно конец хвоста ревизуется на следующем decode).

Работает только со словами, уже смещёнными в АБСОЛЮТНЫЕ тайм-коды (секунды от начала фонограммы
канала) — вызывающий код (Session) отвечает за то, чтобы отфильтровать слова раньше
`committed_boundary_sample` перед вызовом `apply_partial` (см. RollingBuffer.window_since —
окно может содержать lookback уже закоммиченного аудио для контекста decode, но эти слова не
должны повторно участвовать в сравнении/попадать в текст дважды).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import ASRWord

REVISION_HOLDBACK_WORDS = 2


def _norm(word: str) -> str:
    return word.strip().lower().strip(".,!?;:—-")


def common_prefix_len(prev: list[ASRWord], new: list[ASRWord]) -> int:
    n = 0
    for a, b in zip(prev, new):
        if _norm(a.text) != _norm(b.text):
            break
        n += 1
    return n


@dataclass
class UtteranceHypothesis:
    """Состояние LocalAgreement-2 для одной открытой реплики. Один инстанс на `utterance_id`."""
    utterance_id: str
    channel: int
    committed_words: list[ASRWord] = field(default_factory=list)
    committed_boundary_sample: int = 0
    _previous_tail: list[ASRWord] = field(default_factory=list)
    # Накопление own-SNR (Блок 3.7) для пост-ASR дедупликации cross-talk — хранится именно на
    # гипотезе (не на разделяемом состоянии канала), чтобы не словить ту же гонку, что и с
    # committed_words: колбэк финала захватывает гипотезу напрямую, а не текущее состояние канала.
    snr_sum: float = 0.0
    snr_count: int = 0

    def add_snr_sample(self, snr_db: float) -> None:
        self.snr_sum += snr_db
        self.snr_count += 1

    @property
    def avg_snr_db(self) -> float:
        return self.snr_sum / self.snr_count if self.snr_count else 0.0

    def apply_partial(self, tail_words: list[ASRWord], sample_rate: int) -> dict:
        """`tail_words` — слова текущего decode-раунда, абсолютные тайм-коды, уже отфильтрованные
        от всего, что раньше `committed_boundary_sample`. Возвращает данные для `asr_update`."""
        lcp = common_prefix_len(self._previous_tail, tail_words)
        stable_count = max(0, lcp - REVISION_HOLDBACK_WORDS)
        newly_stable = tail_words[:stable_count]
        remaining_tail = tail_words[stable_count:]

        if newly_stable:
            self.committed_words.extend(newly_stable)
            self.committed_boundary_sample = int(round(newly_stable[-1].end * sample_rate))

        self._previous_tail = tail_words
        all_words = self.committed_words + remaining_tail
        text = " ".join(w.text.strip() for w in all_words).strip()
        return {
            "text": text,
            "stable_word_count": len(self.committed_words),
            "newly_stable": newly_stable,
            "committed_boundary_sample": self.committed_boundary_sample,
        }

    @staticmethod
    def finalize_text(words: list[ASRWord]) -> str:
        return " ".join(w.text.strip() for w in words).strip()
