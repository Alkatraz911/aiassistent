"""Эндпоинтинг: решает, когда реплика началась/закончилась (Блок 2.1 плана).

Раньше одна и та же пауза одновременно решала «когда закрыть фразу» и «когда впервые показать
текст» (см. старый `ChunkBuffer` в `buffer.py`). Здесь — только первое: классификация речь/тишина
и открытие/закрытие «реплики» (`utterance`). Частота ASR-обновлений — независимый механизм
(`RollingBuffer` + `AsrScheduler`, каданс `ASR_UPDATE_MS`).

Классификация речи — адаптивный порог шума (`NoiseFloorTracker`, Блок 3.6), а не фиксированная
константа: реальный шум помещения/чувствительность микрофона со временем «переучивают» порог,
вместо того чтобы навсегда застревать на дефолтном значении. Полноценный потоковый Silero VAD
(Этап 2 плана) — опциональное дальнейшее улучшение, не влючено по умолчанию (см. README).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .. import config
from .noise import NoiseFloorTracker


def _new_utterance_id() -> str:
    return uuid.uuid4().hex[:12]


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


@dataclass
class EndpointEvent:
    kind: Literal["utterance_start", "utterance_end"]
    utterance_id: str
    generation: int
    start_sample: int = 0   # только для utterance_start
    forced: bool = False    # utterance_end по MAX_UTTERANCE_MS, а не по паузе


@dataclass
class Endpointer:
    """Один инстанс на канал."""
    sample_rate: int = config.SAMPLE_RATE
    _floor: NoiseFloorTracker = field(
        default_factory=lambda: NoiseFloorTracker(config.VAD_ENERGY_THRESHOLD))
    _silence_run: int = 0             # сэмплов подряд тишины
    _speech_run: int = 0              # сэмплов подряд речи (для MIN_SPEECH_MS)
    _utterance_open: bool = False
    _utterance_id: str = ""
    _utterance_start_sample: int = 0
    _total_samples: int = 0
    _gen_counter: int = 0
    generations: dict[str, int] = field(default_factory=dict)   # utterance_id -> поколение
    closed: dict[str, bool] = field(default_factory=dict)       # utterance_id -> уже завершена?

    def add(self, pcm: np.ndarray) -> list[EndpointEvent]:
        events: list[EndpointEvent] = []
        n = len(pcm)
        is_speech = self._floor.classify(_rms(pcm), margin_db=config.VAD_SPEECH_MARGIN_DB)

        if is_speech:
            self._speech_run += n
            self._silence_run = 0
            min_speech_samples = config.MIN_SPEECH_MS * self.sample_rate // 1000
            if not self._utterance_open and self._speech_run >= min_speech_samples:
                self._utterance_open = True
                self._utterance_id = _new_utterance_id()
                self._utterance_start_sample = self._total_samples - self._speech_run + n
                self._gen_counter += 1
                self.generations[self._utterance_id] = self._gen_counter
                self.closed[self._utterance_id] = False
                events.append(EndpointEvent(
                    kind="utterance_start", utterance_id=self._utterance_id,
                    generation=self._gen_counter, start_sample=self._utterance_start_sample,
                ))
        else:
            self._silence_run += n
            self._speech_run = 0

        self._total_samples += n

        if self._utterance_open:
            silence_ms = self._silence_run * 1000.0 / self.sample_rate
            open_ms = (self._total_samples - self._utterance_start_sample) * 1000.0 / self.sample_rate
            ended_by_silence = silence_ms >= config.ENDPOINT_SILENCE_MS
            ended_by_maxlen = open_ms >= config.MAX_UTTERANCE_MS
            if ended_by_silence or ended_by_maxlen:
                events.append(self._close_utterance(forced=ended_by_maxlen and not ended_by_silence))
        return events

    def _close_utterance(self, forced: bool) -> EndpointEvent:
        uid = self._utterance_id
        self.closed[uid] = True
        ev = EndpointEvent(
            kind="utterance_end", utterance_id=uid,
            generation=self.generations[uid], forced=forced,
        )
        self._utterance_open = False
        self._silence_run = 0
        self._speech_run = 0
        return ev

    def force_end(self) -> EndpointEvent | None:
        """Принудительно закрыть открытую реплику (например, по `{"type":"stop"}`)."""
        if not self._utterance_open:
            return None
        return self._close_utterance(forced=True)

    def generation_of(self, utterance_id: str) -> int:
        return self.generations.get(utterance_id, -1)

    def is_closed(self, utterance_id: str) -> bool:
        return self.closed.get(utterance_id, True)

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def open_utterance_id(self) -> str | None:
        return self._utterance_id if self._utterance_open else None
