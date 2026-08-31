"""Хранилище именованных шаблонов анкеты (Блок 5 плана).

Один JSON-файл на шаблон в `storage/templates/<id>.json`. Атомарная запись (временный файл +
`os.replace()`) — обрыв питания/процесса посередине сохранения не должен оставлять битый JSON,
из-за которого потом не запустится анкета. Builtin-шаблон (засеивается из
`DEFAULT_TEMPLATE_STEPS` при первом обращении, если хранилище пустое) неизменяем и неудаляем —
редактирование доступно только через «Создать копию» (см. `create()`).
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .. import config
from ..models import Template, TemplateStep, TemplateSummary
from .questionnaire import DEFAULT_TEMPLATE_STEPS

BUILTIN_ID = "default"
BUILTIN_NAME = "Стандартный (протокол опроса)"


def _validate_unique_keys(steps: list[TemplateStep]) -> None:
    seen: set[str] = set()
    for s in steps:
        if not s.key:
            raise ValueError("у шага анкеты не может быть пустого ключа")
        if s.key in seen:
            raise ValueError(f"дублирующийся ключ шага: {s.key!r}")
        seen.add(s.key)


class TemplateStore:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seed_builtin_if_empty()

    def _path(self, template_id: str) -> Path:
        return self._dir / f"{template_id}.json"

    def _write_locked(self, tmpl: Template) -> None:
        path = self._path(tmpl.id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(tmpl.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp_path, path)   # атомарно на одной ФС (Windows/NTFS — тоже)

    def _seed_builtin_if_empty(self) -> None:
        with self._lock:
            if any(self._dir.glob("*.json")):
                return
            tmpl = Template(id=BUILTIN_ID, name=BUILTIN_NAME, is_builtin=True,
                             steps=list(DEFAULT_TEMPLATE_STEPS))
            self._write_locked(tmpl)

    def list(self) -> list[TemplateSummary]:
        out: list[TemplateSummary] = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                tmpl = Template.model_validate_json(p.read_text(encoding="utf-8"))
            except Exception:
                continue   # битый/чужой файл в папке — пропускаем, не роняем список
            out.append(TemplateSummary(
                id=tmpl.id, name=tmpl.name, description=tmpl.description,
                is_builtin=tmpl.is_builtin, step_count=len(tmpl.steps), updated_at=tmpl.updated_at,
            ))
        out.sort(key=lambda t: (not t.is_builtin, t.name.lower()))
        return out

    def get(self, template_id: str) -> Template | None:
        p = self._path(template_id)
        if not p.exists():
            return None
        try:
            return Template.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def create(self, name: str, description: str, steps: list[TemplateStep],
               id: str | None = None, docx_filename: str | None = None,
               qa_placeholder: str | None = None) -> Template:
        """`id` — явно заданный id (например, черновик из `/api/templates/import_docx`, для
        которого докс-файл уже сохранён под этим id на диске: без переиспользования id тут
        сохранённый шаблон получил бы ДРУГОЙ id, и файл `docx/<новый_id>.docx` осиротел бы,
        не будучи ни на что не сославшимся). `None` — обычное поведение, id генерируется сам."""
        _validate_unique_keys(steps)
        kwargs = dict(name=name, description=description, steps=steps,
                      docx_filename=docx_filename, qa_placeholder=qa_placeholder)
        if id:
            kwargs["id"] = id
        tmpl = Template(**kwargs)
        with self._lock:
            self._write_locked(tmpl)
        return tmpl

    def update(self, template_id: str, name: str, description: str,
               steps: list[TemplateStep], docx_filename: str | None = None,
               qa_placeholder: str | None = None) -> Template | None:
        """`None` — шаблон не найден. Поднимает `PermissionError` для builtin.
        `docx_filename`/`qa_placeholder` явно принимаются (а не наследуются молча), чтобы
        редактирование импортированного из .docx шаблона могло и сохранить, и поменять
        привязку к докс-файлу."""
        existing = self.get(template_id)
        if existing is None:
            return None
        if existing.is_builtin:
            raise PermissionError("builtin-шаблон нельзя редактировать — создайте копию")
        _validate_unique_keys(steps)
        updated = existing.model_copy(update={
            "name": name, "description": description, "steps": steps,
            "docx_filename": docx_filename, "qa_placeholder": qa_placeholder,
            "version": existing.version + 1, "updated_at": time.time(),
        })
        with self._lock:
            self._write_locked(updated)
        return updated

    def delete(self, template_id: str) -> bool:
        """`False` — шаблон не найден. Поднимает `PermissionError` для builtin, `ValueError`,
        если это последний оставшийся шаблон (анкете нужен хотя бы один)."""
        existing = self.get(template_id)
        if existing is None:
            return False
        if existing.is_builtin:
            raise PermissionError("builtin-шаблон нельзя удалить")
        with self._lock:
            if len(list(self._dir.glob("*.json"))) <= 1:
                raise ValueError("нельзя удалить последний оставшийся шаблон")
            self._path(template_id).unlink(missing_ok=True)
        return True


store = TemplateStore(config.TEMPLATES_DIR)
