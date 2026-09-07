"""Список сохранённых допросов внутри проекта (Блок 7). Отдельного индекса на диске не держим —
сессий на один проект обычно единицы-десятки, а `protocol.json` уже лежит по одному на сессию
(см. `Session.save`) — сканируем `storage/sessions/*/protocol.json`, как `TemplateStore.list()`
сканирует `storage/templates/*.json`: тот же принцип, тот же codebase-стиль."""
from __future__ import annotations

from datetime import datetime

from .. import config
from ..models import Protocol, SessionSummary

# Шаг анкеты считаем частью ФИО, если он явно помечен извлекателем "fio" (см. questionnaire.
# EXTRACTORS), ИЛИ его название содержит одно из этих слов — оба сигнала объединяем: у builtin-
# шаблона это extractor="fio", у реальных бланков протоколов (Блок 6) — обычно 1-3 отдельных шага
# с человекочитаемыми названиями вроде "Фамилия", "Имя и отчество" и т.п.
_FIO_LABEL_WORDS = ("фамили", "имя", "отчеств", "фио")

# Запасной сигнал — латинские токены плейсхолдеров, которые генерирует импорт .docx (Блок 6,
# `docx_import.normalize_docx`): шаг тогда создаётся с label = сам токен (напр. "T1.PARTICIP_
# SURNAME"), пока оператор не переименует его вручную (см. main.py::import_docx_template) —
# ни одно из русских слов _FIO_LABEL_WORDS в такой label не встречается, и допрос до переименования
# показывался в списке как «Без названия» (реальный случай). Ищем эти же слова уже в токене
# плейсхолдера, а не в названии шага.
_FIO_PLACEHOLDER_WORDS = ("surname", "фамили", "name", "имя", "patronymic", "отчеств", "fio", "фио")


def _display_name(protocol: Protocol) -> str:
    if protocol.template_snapshot is None:
        return ""
    answers = {f.key: f.value for f in protocol.questionnaire}
    steps = protocol.template_snapshot.steps
    fio_steps = [s for s in steps if s.extractor == "fio"]
    if not fio_steps:
        fio_steps = [s for s in steps if any(w in s.label.lower() for w in _FIO_LABEL_WORDS)]
    if not fio_steps:
        fio_steps = [
            s for s in steps
            if any(w in (s.placeholder or "").lower() for w in _FIO_PLACEHOLDER_WORDS)
        ]
    # Порядок шагов — порядок появления плейсхолдеров в бланке (см. docx_import), а у реальных
    # протоколов это фамилия/имя/отчество именно в таком порядке — переставлять не нужно.
    parts = [answers.get(s.key, "").strip() for s in fio_steps]
    return " ".join(p for p in parts if p)


def _date(protocol: Protocol) -> str:
    ts = protocol.segments[0].created_at if protocol.segments else protocol.created_at
    return datetime.fromtimestamp(ts).strftime("%d.%m.%Y")


def list_project_sessions(project_id: str) -> list[SessionSummary]:
    out: list[SessionSummary] = []
    for p in config.STORAGE_DIR.glob("*/protocol.json"):
        try:
            protocol = Protocol.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            continue   # битый/чужой файл — пропускаем, не роняем список (тот же принцип, что в TemplateStore)
        if protocol.project_id != project_id:
            continue
        out.append(SessionSummary(
            session_id=protocol.session_id,
            template_name=protocol.template_name or "",
            display_name=_display_name(protocol),
            date=_date(protocol),
            created_at=protocol.created_at,
        ))
    out.sort(key=lambda s: s.created_at, reverse=True)   # свежие допросы — сверху
    return out
