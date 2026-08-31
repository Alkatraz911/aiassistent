"""Владение экземплярами ASR-моделей с подсчётом ссылок (Блок 4 плана).

Модели-провайдеры (`FasterWhisperASR` и т.п.) тяжёлые (сотни МБ – единицы ГБ), поэтому их
переключение из UI кэшируется. Раньше `Session` держала бы прямую ссылку на инстанс — тогда
эвикция из LRU не освобождает память, если сессия ещё держит ссылку (LRU думает, что моделей 2,
а RAM/VRAM реально держит 3–4). Поэтому `Session` хранит только `key` (см. `session.py`), а
владение и подсчёт ссылок — здесь; эвикция разрешена только когда `refcount == 0`.
"""
from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict

from . import get_asr
from .base import ASRProvider
from .. import config

ModelKey = tuple[str, str, str]   # (model, device, compute)


class ModelManager:
    # Сколько помнить, что создание/прогрев ключа упали. Без этой памяти отказ не кешируется
    # вообще (провайдер записывается в `_providers` только после успешного `warmup()`), и
    # сессия, привязанная к непригодному ключу, пересоздавала бы многогигабайтную модель на
    # КАЖДОЙ задаче — то есть ~2 раза в секунду при партиалах, каждый раз с полным трейсбеком
    # в лог. Не навсегда: отказ бывает и внешним (VRAM занял соседний процесс, карта
    # освободилась), поэтому через минуту разрешаем попытку заново.
    _FAIL_TTL_S = 60.0

    def __init__(self, max_cached: int | None = None) -> None:
        self._max_cached = max_cached or config.ASR_MAX_CACHED_MODELS
        self._providers: dict[ModelKey, ASRProvider] = {}
        self._refcount: dict[ModelKey, int] = {}
        self._lru: "OrderedDict[ModelKey, None]" = OrderedDict()
        self._lock = threading.Lock()
        # Персональный лок на КАЖДЫЙ ключ, пока он создаётся — без этого два worker-потока
        # планировщика (ASR_WORKER_THREADS > 1), одновременно попросившие ещё не загруженную
        # модель, оба звали бы `get_asr()`/`WhisperModel(...)` параллельно. Реальный прогон
        # показал, что это не просто «лишняя работа» — параллельная загрузка одной и той же
        # модели через huggingface_hub падает (гонка в его собственном tqdm-прогрессбаре),
        # роняя один из worker-потоков планировщика молча (без восстановления).
        self._creating_locks: dict[ModelKey, threading.Lock] = {}
        self._failed: dict[ModelKey, tuple[float, str]] = {}   # ключ -> (когда, чем)
        # Ключи, которые прямо сейчас создаются. Считаются в бюджете вытеснения наравне с уже
        # загруженными: вытеснение и вставка идут в РАЗНЫХ секциях `self._lock`, а `creating_lock`
        # персональный на ключ — значит два потока с РАЗНЫМИ ключами оба увидят место, каждый
        # после чужого вытеснения, и оба вставят. На карте оказалось бы три модели вместо двух.
        self._creating: set[ModelKey] = set()

    def default_key(self) -> ModelKey:
        return (config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE)

    def is_loaded(self, key: ModelKey) -> bool:
        with self._lock:
            return key in self._providers

    def loaded_keys(self) -> list[ModelKey]:
        with self._lock:
            return list(self._lru.keys())

    def _touch_locked(self, key: ModelKey) -> ASRProvider:
        """Вызывать под `self._lock`, когда `self._providers[key]` уже точно существует."""
        self._refcount[key] = self._refcount.get(key, 0) + 1
        self._lru[key] = None
        self._lru.move_to_end(key)
        return self._providers[key]

    def acquire(self, key: ModelKey) -> ASRProvider:
        """Возвращает провайдер для `key`, загружая (и прогревая) его при необходимости.
        Обязательно парная вызову `release(key)` — иначе рефкаунт никогда не дойдёт до 0 и
        модель не сможет быть вытеснена. Безопасно вызывать конкурентно на один и тот же
        `key` из нескольких потоков — фактически создаст модель только один из них."""
        with self._lock:
            if key in self._providers:
                return self._touch_locked(key)
            self._raise_if_failed_locked(key)
            creating_lock = self._creating_locks.setdefault(key, threading.Lock())

        with creating_lock:
            # Пока ждали лок, модель мог успеть создать другой поток — тогда просто используем её.
            # Или, наоборот, уронить: тогда повторять ту же загрузку следом бессмысленно.
            with self._lock:
                if key in self._providers:
                    return self._touch_locked(key)
                self._raise_if_failed_locked(key)
                self._creating.add(key)
                evicted = self._evict_if_needed_locked()

            # Освобождаем ВЫТЕСНЕННЫЕ модели до создания новой и вне лока. На CPU порядок не
            # принципиален (ОС отдаст память под своп), на GPU — принципиален: VRAM жёстко
            # ограничена, и если старая модель ещё жива в момент загрузки новой, на карте
            # оказываются обе (large-v3 float16 — ~3 ГБ каждая) и загрузка падает с out of
            # memory. Плюс явный `gc.collect()`: память CTranslate2 освобождается в деструкторе
            # C++-объекта, а он ждёт, пока Python досчитает ссылки, — при циклах это отложенно.
            if evicted:
                evicted.clear()
                gc.collect()

            model, device, compute = key
            try:
                provider = get_asr(model, device, compute)
                provider.warmup()
            except Exception as exc:
                with self._lock:
                    self._failed[key] = (time.monotonic(), f"{type(exc).__name__}: {exc}")
                    self._creating.discard(key)
                    self._creating_locks.pop(key, None)
                raise

            with self._lock:
                self._providers[key] = provider
                self._creating.discard(key)     # место под этот ключ уже занято реальным
                self._failed.pop(key, None)
                result = self._touch_locked(key)
                self._creating_locks.pop(key, None)
            return result

    def _raise_if_failed_locked(self, key: ModelKey) -> None:
        """Вызывать под `self._lock`. Отдаёт ЗАПОМНЕННУЮ ошибку вместо повторной попытки,
        пока не истёк `_FAIL_TTL_S`. Текст исходного исключения сохраняем дословно: на него
        смотрит `main.py::switch_model` (распознаёт «out of memory», чтобы подсказать про VRAM)."""
        rec = self._failed.get(key)
        if rec is None:
            return
        when, why = rec
        if time.monotonic() - when >= self._FAIL_TTL_S:
            self._failed.pop(key, None)
            return
        raise RuntimeError(f"модель {key[0]} на {key[1]}/{key[2]} не поднялась: {why}")

    def release(self, key: ModelKey) -> None:
        with self._lock:
            if key in self._refcount:
                self._refcount[key] = max(0, self._refcount[key] - 1)

    def _evict_if_needed_locked(self) -> list[ASRProvider]:
        """Вызывается под `self._lock`. Возвращает вытесненные провайдеры, чтобы вызывающий
        отпустил их ВНЕ лока (освобождение VRAM в деструкторе CTranslate2 — не мгновенное).

        Бюджет считается по загруженным ПЛЮС создаваемым сейчас (`self._creating`, куда
        вызывающий уже добавил свой ключ): иначе параллельные создания разных ключей делят
        одно и то же освободившееся место."""
        evicted: list[ASRProvider] = []
        if len(self._providers) + len(self._creating) <= self._max_cached:
            return evicted
        for key in list(self._lru.keys()):
            if len(self._providers) + len(self._creating) <= self._max_cached:
                break
            if self._refcount.get(key, 0) == 0:
                self._lru.pop(key, None)
                provider = self._providers.pop(key, None)
                self._refcount.pop(key, None)
                if provider is not None:
                    evicted.append(provider)
        return evicted
