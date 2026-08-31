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
    aligned: bool = False                 # тайм-коды уточнены forced-alignment
    edits: list[Edit] = Field(default_factory=list)
    # Кросс-ток (Блок 3.7): средний own-SNR канала за время реплики (для пост-ASR сравнения),
    # и пометка «вероятный дубль протёкшего голоса» — НЕ удаляется, только приглушается в UI,
    # решение остаётся за оператором (тот же принцип, что и audit-слой `Edit`).
    own_snr_db: float = 0.0
    likely_bleed: bool = False
    bleed_score: float = 0.0

    def to_ws_dict(self) -> dict:
        """Представление сегмента для WS-сообщений `segment`/`asr_final` — единая точка,
        чтобы формат не расходился между live-стримингом и остальными местами отправки."""
        return {
            "id": self.id, "channel": self.channel, "speaker": self.speaker,
            "speaker_auto": self.speaker_auto, "start": self.start, "end": self.end,
            "text": self.text, "likely_bleed": self.likely_bleed, "bleed_score": self.bleed_score,
            "words": [{"text": w.text, "start": w.start, "end": w.end} for w in self.words],
        }


class QuestionnaireField(BaseModel):
    """Поле анкеты, заполняемое AI-ассистентом."""
    key: str
    label: str
    value: str = ""
    confirmed: bool = False


class TemplateStep(BaseModel):
    """Один шаг анкеты внутри шаблона (Блок 5)."""
    key: str
    label: str
    kind: Literal["field", "confirm", "info"] = "field"
    statement: str = ""     # текст для зачитывания (разъяснение, опционально)
    question: str = ""      # вопрос, требующий ответа (для field/confirm)
    extractor: str = "plain"    # ключ из EXTRACTORS (questionnaire.py): fio/birth/yesno/plain/none
    # Генерация .docx-протокола (Блок 6): токен mail-merge докс-шаблона (без "#{}"), в который
    # уйдёт ответ на этот шаг, например "T1.PARTICIP_SURNAME". Пусто — шаг не привязан к докс-полю.
    placeholder: str = ""
    # UI-подсказка клиенту: "asr" — спрашивать голосом как сегодня, "manual" — просто текстовое
    # поле (для полей шапки протокола, которые опрашиваемый вслух не произносит — имя следователя,
    # место проведения и т.п.). Бэкенд оба варианта хранит и обрабатывает одинаково.
    source: Literal["asr", "manual"] = "asr"


class Template(BaseModel):
    """Именованный шаблон анкеты — пользователь может завести несколько под разные сценарии
    опроса (Блок 5). `is_builtin` — неизменяемый и неудаляемый сид, редактируется только через
    копию. `version` растёт при каждом сохранении — `Protocol.template_snapshot` фиксирует
    формулировки вопросов ровно на момент прохождения анкеты, даже если шаблон потом изменят."""
    id: str = Field(default_factory=_id)
    name: str
    description: str = ""
    is_builtin: bool = False
    version: int = 1
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    steps: list[TemplateStep] = Field(default_factory=list)
    # Генерация .docx-протокола (Блок 6): имя (не путь) нормализованного докс-файла под
    # `storage/templates/docx/`, например "<template_id>.docx". None — к шаблону не привязан
    # докс (обычный анкетный шаблон без генерации документа).
    docx_filename: Optional[str] = None
    # Токен докс-шаблона, в который уходит полная стенограмма «Вопрос/Ответ» допроса.
    qa_placeholder: Optional[str] = None


class TemplateSummary(BaseModel):
    """Облегчённое представление шаблона для списка (без шагов)."""
    id: str
    name: str
    description: str = ""
    is_builtin: bool = False
    step_count: int = 0
    updated_at: float = 0.0


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
    # Снапшот модели ASR, использованной в этой сессии (Блок 4) — активная модель могла
    # смениться в UI после записи, поэтому протокол хранит именно ту, что реально распознавала.
    asr_model: Optional[str] = None
    asr_device: Optional[str] = None
    asr_compute: Optional[str] = None
    # Снапшот шаблона анкеты (Блок 5) — id/имя для ссылки + полная копия шагов на момент старта
    # анкеты, чтобы последующее редактирование шаблона не искажало историю уже пройденных сессий.
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    template_version: Optional[int] = None
    template_snapshot: Optional[Template] = None
