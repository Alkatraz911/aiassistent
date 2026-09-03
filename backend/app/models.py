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
    # Момент реального времени, когда реплика была зафиксирована (Блок 6) — в отличие от
    # start/end (секунды от начала фонограммы), это wall-clock, нужный, чтобы посчитать
    # реальное время начала/окончания допроса для докс-полей с source="auto"
    # (docgen.AUTO_FIELDS). Ставится автоматически в момент создания сегмента
    # (Session._finish_utterance) — небольшая задержка на распознавание тут не критична,
    # для отображения с точностью до минуты этого достаточно.
    created_at: float = Field(default_factory=time.time)
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
    # Источник значения: "asr" — спрашивать голосом как обычно, "manual" — текстовое поле,
    # заполняемое оператором заново в КАЖДОЙ сессии (для разовых деталей шапки протокола),
    # "profile" — значение общее для всех сессий этого пользователя программы (например «автор
    # документа» — следователь не меняется от допроса к допросу), заполняется один раз в профиле
    # оператора (см. assistant/profile.py) и дальше подставляется автоматически; "auto" —
    # вычисляется из данных самой сессии (дата, время начала/окончания опроса, см.
    # docgen.AUTO_FIELDS) — какое именно значение вычислять, задаёт `extractor` (для source="auto"
    # это не имя ASR-извлекателя, а ключ из docgen.AUTO_FIELDS: date/time_start/time_end/
    # time_range). Шаги с source="profile" и source="auto" в голосовую анкету вообще не попадают
    # (см. questionnaire.build_script) — оба берут значение не из ответов сессии.
    source: Literal["asr", "manual", "profile", "auto"] = "asr"


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


class Project(BaseModel):
    """Дело/проект (Блок 7) — папка, в которую пользователь складывает проведённые допросы.
    Без проекта начать сессию нельзя: список допросов в клиенте строится по проектам, а не
    плоским списком всех когда-либо сохранённых сессий."""
    id: str = Field(default_factory=_id)
    name: str
    created_at: float = Field(default_factory=time.time)


class SessionSummary(BaseModel):
    """Одна строка списка «допросы этого проекта» (Блок 7) — без полной расшифровки/аудио,
    только то, что нужно узнать нужную запись в списке."""
    session_id: str
    template_name: str = ""
    display_name: str = ""    # ФИО опрашиваемого, по возможности (см. sessions_index.py)
    date: str = ""            # DD.MM.YYYY, реальная дата допроса (см. docgen._auto_date)
    created_at: float = 0.0


class Protocol(BaseModel):
    """Полный протокол сессии."""
    session_id: str
    created_at: float = Field(default_factory=time.time)
    # Wall-clock момент первого же старта записи диалога (Session.attach_ws, только при первом
    # вызове — повторные «начать/остановить» его не трогают). Отдельно от `created_at`: сессия
    # создаётся при старте анкеты, которая может идти долго ДО того, как реально нажали «Начать
    # запись» — `created_at` в это время был бы неверным якорем для времени допроса.
    recording_started_at: Optional[float] = None
    # Проект (Блок 7), к которому относится этот допрос — задаётся один раз, при создании сессии
    # (см. POST /api/session/init), до начала анкеты/записи. None — сессии, заведённые до Блока 7
    # (обратная совместимость), в списки допросов проекта не попадают.
    project_id: Optional[str] = None
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
