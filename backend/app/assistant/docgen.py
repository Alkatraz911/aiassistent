"""Генерация итогового .docx-протокола допроса из завершённой сессии (Блок 6).

Заполняет нормализованный докс-шаблон (см. `docx_import.normalize_docx`) значениями анкеты и,
опционально, полной стенограммой «Вопрос/Ответ». Модуль намеренно не знает про `session.py`/
`config` — оба пути (шаблон и куда сохранить) резолвит вызывающий код (`main.py`), чтобы генерацию
документа можно было проверить в тестах без поднятия сессии/сервера."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from docxtpl import DocxTemplate

from ..models import Protocol, Segment
from .docx_import import jinja_key

MISSING_MARKER = "«____не заполнено____»"


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

    context: dict[str, str] = {}
    for step in protocol.template_snapshot.steps:
        token = step.placeholder or step.key
        if not token:
            continue
        context[jinja_key(token)] = answers.get(step.key) or MISSING_MARKER

    qa_placeholder = protocol.template_snapshot.qa_placeholder
    if qa_placeholder:
        transcript = build_qa_transcript(protocol.segments, protocol.speaker_map.get)
        context[jinja_key(qa_placeholder)] = transcript or MISSING_MARKER

    tpl = DocxTemplate(str(docx_template_path))
    tpl.render(context)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tpl.save(str(out_path))
    return out_path
