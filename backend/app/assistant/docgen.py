"""Генерация итогового .docx-протокола допроса из завершённой сессии (Блок 6).

Заполняет нормализованный докс-шаблон (см. `docx_import.normalize_docx`) значениями анкеты и,
опционально, полной стенограммой «Вопрос/Ответ». Модуль намеренно не знает про `session.py`/
`config` — оба пути (шаблон и куда сохранить) резолвит вызывающий код (`main.py`), чтобы генерацию
документа можно было проверить в тестах без поднятия сессии/сервера."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from docxtpl import DocxTemplate

from ..models import Protocol, Segment
from .docx_import import jinja_key, scan_jinja_keys
from .profile import store as profile_store

MISSING_MARKER = "«____не заполнено____»"


# --- авто-вычисляемые поля (Блок 6: source="auto") -------------------------------------------
# Для таких шагов `extractor` — не имя ASR-извлекателя, а ключ из этого реестра (выбирается в
# редакторе шаблона отдельным полем «Авто-поле»). Дата и время начала/окончания опроса берутся не
# из системных часов на момент генерации документа, а из РЕАЛЬНОГО времени записи: `Segment.
# created_at` — wall-clock момент, когда каждая реплика была зафиксирована (см. models.py),
# первая и последняя реплика по таймлайну сессии (`Segment.start`) дают начало/конец допроса.

def _auto_date(protocol: Protocol) -> str | None:
    ts = protocol.segments[0].created_at if protocol.segments else protocol.created_at
    return datetime.fromtimestamp(ts).strftime("%d.%m.%Y")


def _ordered_segments(protocol: Protocol) -> list[Segment]:
    return sorted(protocol.segments, key=lambda s: s.start)


def _auto_time_start(protocol: Protocol) -> str | None:
    segs = _ordered_segments(protocol)
    if not segs:
        return None   # нечего вычислять — опрос ещё не записывался
    # С секундами, не только часы:минуты — короткая сессия (в т.ч. просто тестовый прогон)
    # укладывается в одну и ту же минуту целиком, и начало с окончанием визуально совпадали бы,
    # даже будучи технически разными (реальная жалоба пользователя). Для настоящего протокола
    # секунды тоже не лишние — это только точнее.
    return datetime.fromtimestamp(segs[0].created_at).strftime("%H:%M:%S")


def _auto_time_end(protocol: Protocol) -> str | None:
    segs = _ordered_segments(protocol)
    if not segs:
        return None
    return datetime.fromtimestamp(segs[-1].created_at).strftime("%H:%M:%S")


def _auto_time_range(protocol: Protocol) -> str | None:
    start, end = _auto_time_start(protocol), _auto_time_end(protocol)
    if not start or not end:
        return None
    return f"{start} - {end}"


AUTO_FIELDS: dict[str, Callable[[Protocol], str | None]] = {
    "date": _auto_date,
    "time_start": _auto_time_start,
    "time_end": _auto_time_end,
    "time_range": _auto_time_range,
}


def build_qa_transcript(segments: list[Segment], speaker_label: Callable[[int], str]) -> str:
    """Стенограмма допроса как чередование «Вопрос:»/«Ответ:». `speaker_label` в сигнатуре — для
    единообразия с остальным кодом сессии, где подпись спикера всегда резолвится через колбэк;
    сам текст здесь размечается не по имени, а по каналу (см. допущение ниже).

    Допущение (как и везде в проекте, канал 0 = интервьюер по умолчанию): канал 0 — «Вопрос:»,
    любой другой канал — «Ответ:». Это тот же принцип, что и `Session.speaker_label`.
    """
    ordered = sorted(segments, key=lambda s: s.start)

    paragraphs: list[str] = []
    cur_channel: int | None = None
    cur_parts: list[str] = []

    def flush():
        if cur_channel is None or not cur_parts:
            return
        prefix = "Вопрос: " if cur_channel == 0 else "Ответ: "
        paragraphs.append(prefix + " ".join(cur_parts))

    for seg in ordered:
        text = seg.text.strip()
        if not text:
            continue
        if seg.channel != cur_channel:
            flush()
            cur_channel = seg.channel
            cur_parts = [text]
        else:
            cur_parts.append(text)
    flush()

    return "\n\n".join(paragraphs)


def render(protocol: Protocol, docx_template_path: Path, out_path: Path) -> Path:
    """Заполняет докс-шаблон значениями протокола и сохраняет в `out_path`."""
    if protocol.template_snapshot is None or not protocol.template_snapshot.docx_filename:
        raise ValueError(
            "у протокола нет снапшота шаблона с привязанным .docx — сгенерировать документ нечем"
        )

    answers = {f.key: f.value for f in protocol.questionnaire}
    profile_values = profile_store.get_all()

    context: dict[str, str] = {}
    for step in protocol.template_snapshot.steps:
        token = step.placeholder or step.key
        if not token:
            continue
        # Профильные и авто-поля (Блок 6) — не про эту сессию/не про ответы анкеты (их там и не
        # может быть, см. questionnaire.build_script): профиль берётся из общего хранилища
        # оператора, авто — вычисляется из реального времени записи (см. AUTO_FIELDS выше).
        if step.source == "profile":
            value = profile_values.get(token)
        elif step.source == "auto":
            auto_fn = AUTO_FIELDS.get(step.extractor)
            value = auto_fn(protocol) if auto_fn else None
        else:
            value = answers.get(step.key)
        context[jinja_key(token)] = value or MISSING_MARKER

    qa_placeholder = protocol.template_snapshot.qa_placeholder
    if qa_placeholder:
        transcript = build_qa_transcript(protocol.segments, protocol.speaker_map.get)
        context[jinja_key(qa_placeholder)] = transcript or MISSING_MARKER

    # Страховка от расхождения шагов шаблона с реальным бланком (шаг удалили, переименовали
    # плейсхолдер с опечаткой и т.п. — реальный случай, воспроизведённый пользователем): без
    # неё docxtpl тихо подставляет пустую строку вместо MISSING_MARKER на любой Jinja-переменной,
    # которой не нашлось в `context`, и пропуск в документе остаётся незамеченным при вычитке.
    for key in scan_jinja_keys(docx_template_path):
        context.setdefault(key, MISSING_MARKER)

    tpl = DocxTemplate(str(docx_template_path))
    tpl.render(context)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tpl.save(str(out_path))
    return out_path
