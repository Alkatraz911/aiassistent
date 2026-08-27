"""Владение экземплярами ASR-моделей с подсчётом ссылок (Блок 4 плана).

Модели-провайдеры (`FasterWhisperASR` и т.п.) тяжёлые (сотни МБ – единицы ГБ), поэтому их
переключение из UI кэшируется. Раньше `Session` держала бы прямую ссылку на инстанс — тогда
эвикция из LRU не освобождает память, если сессия ещё держит ссылку (LRU думает, что моделей 2,
а RAM/VRAM реально держит 3–4). Поэтому `Session` хранит только `key` (см. `session.py`), а
владение и подсчёт ссылок — здесь; эвикция разрешена только когда `refcount == 0`.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

from . import get_asr
from .base import ASRProvider
from .. import config

ModelKey = tuple[str, str, str]   # (model, device, compute)


class ModelManager:
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
            creating_lock = self._creating_locks.setdefault(key, threading.Lock())

        with creating_lock:
            # Пока ждали лок, модель мог успеть создать другой поток — тогда просто используем её.
            with self._lock:
                if key in self._providers:
                    return self._touch_locked(key)
                self._evict_if_needed_locked()

            model, device, compute = key
            provider = get_asr(model, device, compute)
            provider.warmup()

            with self._lock:
                self._providers[key] = provider
                result = self._touch_locked(key)
                self._creating_locks.pop(key, None)
            return result

    def release(self, key: ModelKey) -> None:
        with self._lock:
            if key in self._refcount:
                self._refcount[key] = max(0, self._refcount[key] - 1)

    def _evict_if_needed_locked(self) -> None:
        """Вызывается под `self._lock`."""
        if len(self._providers) < self._max_cached:
            return
        for key in list(self._lru.keys()):
            if len(self._providers) < self._max_cached:
                break
            if self._refcount.get(key, 0) == 0:
                self._lru.pop(key, None)
                self._providers.pop(key, None)
                self._refcount.pop(key, None)
