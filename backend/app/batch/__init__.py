"""Пакетная (офлайн) расшифровка готовых аудиозаписей в текстовые файлы.

Отдельный от live-режима путь: там во главе угла задержка (партиалы, окно 12с, beam=1), здесь —
качество и способность дожевать многочасовую запись, не съев память и не потеряв работу при
обрыве. Общее с live — модель, устройство и текстовые фильтры галлюцинаций (`app/asr/base.py`),
их дублировать нельзя: это накопленный на реальных записях опыт.

Точка входа — CLI `app/transcribe_files.py`.
"""
from __future__ import annotations

from .pipeline import BatchTranscriber, Options, Segment, Transcript

__all__ = ["BatchTranscriber", "Options", "Segment", "Transcript"]
