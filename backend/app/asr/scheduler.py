"""Единый ASR-планировщик (Блок 0.1 / 2.2 плана).

Единственная точка входа во все вызовы `ASRProvider.transcribe()` в процессе — live-стриминг
(partial/final), `/api/transcribe` (анкета), `/api/finalize`, `warmup()` при переключении модели.
Раньше `Session.ingest()` вызывала `self.asr.transcribe()` синхронно прямо внутри `async def
ws_stream` — это блокировало event loop на время инференса (канал 1 не мог быть прочитан из
сокета, пока транскрibировался канал 0). Здесь все вызовы сериализованы через один worker-поток с
очередью приоритетов, что заодно устраняет и гонку конкурентных вызовов на одном инстансе модели.

Приоритеты: меньшее число выполняется раньше (обычная семантика `heapq`/`PriorityQueue`).
`PRIORITY_FINAL` меньше `PRIORITY_PARTIAL`, поэтому финалы всегда обгоняют ожидающие партиалы.

Coalescing: на канал одновременно может быть только одно НЕ начатое partial-задание в очереди
(новое заменяет старое) и не ставится новое, пока предыдущее partial-задание этого канала уже
выполняется — иначе при отставании ASR от реального времени очередь только растёт.
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time as _time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .base import ASRProvider, ASRResult
from .model_manager import ModelKey, ModelManager
from .. import config

PRIORITY_FINAL = 0
PRIORITY_PARTIAL = 10
PRIORITY_ONESHOT = 5   # /api/transcribe, /api/finalize — между partial и final по важности


@dataclass(order=True)
class AsrJob:
    priority: int
    seq: int
    session_id: str = field(compare=False)
    channel: int = field(compare=False)
    kind: str = field(compare=False)            # "partial" | "final" | "oneshot"
    utterance_id: str = field(compare=False, default="")
    generation: int = field(compare=False, default=0)
    epoch: int = field(compare=False, default=0)   # «заход» записи (Session.stream_epoch)
    audio: np.ndarray = field(compare=False, default=None)
    initial_prompt: str = field(compare=False, default="")
    provider_key: ModelKey = field(compare=False, default=None)
    on_result: Callable[["AsrJob", ASRResult], None] | None = field(compare=False, default=None)
    on_start: Callable[["AsrJob"], None] | None = field(compare=False, default=None)
    is_current: Callable[[], bool] = field(compare=False, default=lambda: True)


class AsrScheduler:
    def __init__(self, model_manager: ModelManager, num_workers: int | None = None) -> None:
        """`num_workers` > 1 позволяет каналам декодироваться по-настоящему параллельно, а не
        строго по очереди на одном потоке. faster-whisper/CTranslate2 поддерживает конкурентные
        вызовы `transcribe()` на одном инстансе модели (они лишь делят CPU/GPU-ресурсы, но не
        гонятся за внутренним состоянием) — сериализация была нужна только чтобы не блокировать
        event loop (Блок 0.1), а не потому что параллельный инференс небезопасен. Реальный
        живой тест (2 канала, длинные реплики) показал, что 1 воркер на оба канала не успевает за
        темпом поступления partial-заданий — очередь и задержка финала растут без ограничения."""
        self._models = model_manager
        self._queue: list[AsrJob] = []
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending_partial: dict[tuple[str, int], AsrJob] = {}
        self._running_partial: set[tuple[str, int]] = set()
        self._in_flight = 0     # задания, которые сейчас реально выполняются (для wait_idle в тестах)
        self._stopped = False
        n = num_workers or config.ASR_WORKER_THREADS
        self._threads = [
            threading.Thread(target=self._run, name=f"asr-scheduler-{i}", daemon=True)
            for i in range(max(1, n))
        ]
        for t in self._threads:
            t.start()

    def next_seq(self) -> int:
        return next(self._seq)

    def submit_partial(self, job: AsrJob) -> bool:
        """True — задание поставлено; False — пропущено (этот канал уже что-то распознаёт
        прямо сейчас, дожидаемся его результата, а не копим очередь)."""
        key = (job.session_id, job.channel)
        with self._cv:
            if key in self._running_partial:
                return False
            self._pending_partial[key] = job   # старое (если было) станет "неактуальным" по key-check в _run
            heapq.heappush(self._queue, job)
            self._cv.notify()
            return True

    def submit_final(self, job: AsrJob) -> None:
        key = (job.session_id, job.channel)
        with self._cv:
            # Финал обгоняет любой ещё не выполненный partial этого же канала.
            self._pending_partial.pop(key, None)
            heapq.heappush(self._queue, job)
            self._cv.notify()

    def submit_oneshot(self, job: AsrJob) -> None:
        with self._cv:
            heapq.heappush(self._queue, job)
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stopped:
                    self._cv.wait()
                if self._stopped and not self._queue:
                    return
                job = heapq.heappop(self._queue)
                key = (job.session_id, job.channel)
                if job.kind == "partial":
                    if self._pending_partial.get(key) is not job:
                        continue   # заменено более новым partial-заданием — пропускаем
                    del self._pending_partial[key]
                    self._running_partial.add(key)
                self._in_flight += 1
            try:
                self._process_job(job)
            except Exception:
                # Одно задание не должно ронять весь worker-поток насовсем — живой прогон
                # поймал реальный случай (гонка при параллельной загрузке модели, см.
                # `ModelManager.acquire`), когда необработанное исключение тихо убивало поток
                # без восстановления, и планировщик оставался с меньшей ёмкостью незаметно для
                # всего остального. Логируем и продолжаем со следующим заданием.
                import traceback
                traceback.print_exc()
            finally:
                with self._cv:
                    if job.kind == "partial":
                        self._running_partial.discard(key)
                    self._in_flight -= 1
                    self._cv.notify_all()

    def _process_job(self, job: AsrJob) -> None:
        if not job.is_current():
            return
        if job.on_start is not None:
            job.on_start(job)
        provider = self._models.acquire(job.provider_key)
        try:
            beam = (config.WHISPER_BEAM_SIZE_PARTIAL if job.kind == "partial"
                    else config.WHISPER_BEAM_SIZE)
            result = provider.transcribe(
                job.audio, config.SAMPLE_RATE,
                beam_size=beam, initial_prompt=job.initial_prompt or None,
            )
        finally:
            self._models.release(job.provider_key)
        if job.is_current() and job.on_result is not None:
            job.on_result(job, result)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Блокирует, пока очередь не опустеет и не завершатся все выполняющиеся задания.
        Только для тестов/скриптов (smoke.py) — синхронная альтернатива подписке на события."""
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            with self._cv:
                idle = not self._queue and self._in_flight == 0
                if idle:
                    return True
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=min(0.05, remaining))
        with self._cv:
            return not self._queue and self._in_flight == 0

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
