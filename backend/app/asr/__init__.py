"""Выбор ASR-провайдера по конфигурации."""
from __future__ import annotations

from .. import config
from .base import ASRProvider, ASRResult, ASRWord


def get_asr() -> ASRProvider:
    if config.ASR_PROVIDER == "stub":
        from .stub_provider import StubASR
        return StubASR()
    from .faster_whisper_provider import FasterWhisperASR
    return FasterWhisperASR()


__all__ = ["get_asr", "ASRProvider", "ASRResult", "ASRWord"]
