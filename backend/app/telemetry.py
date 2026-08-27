"""Телеметрия задержки распознавания (Блок 1 плана).

Без измерений нельзя доказать, что стриминговый пайплайн (Блок 2) реально уложился в целевые
p95/p99 задержки, а не просто «кажется, работает быстрее». На каждую реплику копится набор
временных меток; из них считаются перцентили по двум ключевым метрикам:

  - «first partial after speech start»  — насколько быстро появляется первый черновой текст;
  - «final after speech end»            — насколько быстро после реальной паузы приходит финал.

Все latency-метки — `time.monotonic_ns()` (не подвержены переводу системных часов/NTP-коррекциям).
`time.time()` в проекте используется только для календарных/аудиторских меток (см. `Edit.ts` в
models.py) — здесь не нужен.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UtteranceTiming:
    utterance_id: str
    channel: int
    speech_start_ts: int | None = None
    speech_end_ts: int | None = None
    asr_enqueued_ts: int | None = None
    asr_started_ts: int | None = None
    asr_finished_ts: int | None = None
    first_partial_ts: int | None = None
    final_ts: int | None = None
    ws_sent_ts: int | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _ms(delta_ns: int) -> float:
    return delta_ns / 1_000_000.0


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


class SessionTelemetry:
    """Копит тайминги реплик одной сессии; опционально пишет JSONL на диск."""

    def __init__(self, session_id: str, jsonl_path: Path | None = None, max_events: int = 2000) -> None:
        self.session_id = session_id
        self._jsonl_path = jsonl_path
        self._lock = threading.Lock()
        self._events: deque[UtteranceTiming] = deque(maxlen=max_events)
        self._by_id: dict[str, UtteranceTiming] = {}

    def _get_or_create(self, utterance_id: str, channel: int) -> UtteranceTiming:
        t = self._by_id.get(utterance_id)
        if t is None:
            t = UtteranceTiming(utterance_id=utterance_id, channel=channel)
            self._by_id[utterance_id] = t
            self._events.append(t)
        return t

    def mark(self, utterance_id: str, channel: int, **fields: int) -> None:
        with self._lock:
            t = self._get_or_create(utterance_id, channel)
            for k, v in fields.items():
                if hasattr(t, k):
                    setattr(t, k, v)
            if t.final_ts is not None:
                self._append_jsonl(t)

    def mark_first_partial(self, utterance_id: str, channel: int) -> None:
        """Как `mark(first_partial_ts=...)`, но только если ещё не выставлено — нам нужен именно
        ПЕРВЫЙ partial после начала речи для метрики SLA, не последний."""
        with self._lock:
            t = self._get_or_create(utterance_id, channel)
            if t.first_partial_ts is None:
                t.first_partial_ts = now_ns()

    def _append_jsonl(self, t: UtteranceTiming) -> None:
        if not self._jsonl_path:
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(t.as_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def summary(self) -> dict:
        with self._lock:
            events = list(self._events)

        first_partial_ms: list[float] = []
        final_ms: list[float] = []
        for t in events:
            if t.speech_start_ts is not None and t.first_partial_ts is not None:
                first_partial_ms.append(_ms(t.first_partial_ts - t.speech_start_ts))
            if t.speech_end_ts is not None and t.final_ts is not None:
                final_ms.append(_ms(t.final_ts - t.speech_end_ts))

        def stats(values: list[float]) -> dict:
            return {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
            }

        return {
            "session_id": self.session_id,
            "utterances": len(events),
            "first_partial_after_speech_start": stats(first_partial_ms),
            "final_after_speech_end": stats(final_ms),
        }

    def raw_events(self) -> list[dict]:
        with self._lock:
            return [t.as_dict() for t in self._events]


class TelemetryRegistry:
    """Реестр телеметрии по сессиям (аналог паттерна SessionManager)."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionTelemetry] = {}
        self._lock = threading.Lock()

    def for_session(self, session_id: str, jsonl_path: Path | None = None) -> SessionTelemetry:
        with self._lock:
            t = self._sessions.get(session_id)
            if t is None:
                t = SessionTelemetry(session_id, jsonl_path)
                self._sessions[session_id] = t
            return t

    def get(self, session_id: str) -> SessionTelemetry | None:
        with self._lock:
            return self._sessions.get(session_id)


registry = TelemetryRegistry()
now_ns = time.monotonic_ns
