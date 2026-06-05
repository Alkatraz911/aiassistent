"""Модели данных протокола.

Сегмент = одна реплика одного спикера с тайм-кодами и словами.
Слой правок (audit) хранит оригинальный текст ASR и историю изменений, чтобы
сохранить юридическую значимость: оригинал фонограммы и расшифровки неизменяем,
правки накладываются сверху с указанием автора и времени.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


def _id() -> str:
    return uuid.uuid4().hex[:12]


class Word(BaseModel):
    """Слово с тайм-кодами относительно начала фонограммы (в секундах)."""
    text: str
    start: float
    end: float
    prob: float = 1.0


class Edit(BaseModel):
    """Запись аудита одной правки текста сегмента."""
    ts: float = Field(default_factory=time.time)
    author: str = "interviewer"
    field: Literal["text", "speaker"] = "text"
    old: str = ""
    new: str = ""


class Segment(BaseModel):
    """Реплика одного спикера."""
    id: str = Field(default_factory=_id)
    channel: int = 0                      # канал/микрофон, с которого пришёл звук
    speaker: str = "Голос-1"              # отображаемая метка (редактируется)
    speaker_auto: str = "Голос-1"         # исходная авто-метка ИИ
    start: float = 0.0
    end: float = 0.0
    text: str = ""                        # текущий (возможно отредактированный) текст
    text_original: str = ""               # исходная расшифровка ASR (неизменяема)
    words: list[Word] = Field(default_factory=list)
    edited: bool = False
    edits: list[Edit] = Field(default_factory=list)


class QuestionnaireField(BaseModel):
    """Поле анкеты, заполняемое AI-ассистентом."""
    key: str
    label: str
    value: str = ""
    confirmed: bool = False


class Protocol(BaseModel):
    """Полный протокол сессии."""
    session_id: str
    created_at: float = Field(default_factory=time.time)
    title: str = "Протокол опроса"
    questionnaire: list[QuestionnaireField] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    # Соответствие канал -> метка спикера, заданное интервьюером
    speaker_map: dict[int, str] = Field(default_factory=dict)
    audio_path: Optional[str] = None
