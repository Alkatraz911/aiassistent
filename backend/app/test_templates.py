"""Проверка хранилища шаблонов анкеты (Блок 5/6) — в частности, удаления builtin-шаблона.

Запуск: `py -3.11 -m app.test_templates` или `pytest app/test_templates.py -v`
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .assistant.templates import BUILTIN_ID, TemplateStore
from .models import TemplateStep


def test_builtin_template_can_be_deleted_when_another_exists() -> None:
    """Builtin неизменяем напрямую (редактирование — только через копию), но удалять его можно
    как любой другой, если он не нужен и в хранилище остаётся хотя бы один шаблон."""
    with tempfile.TemporaryDirectory() as tmp:
        store = TemplateStore(Path(tmp))
        assert store.get(BUILTIN_ID) is not None, "builtin должен засеяться сам при пустом хранилище"
        store.create("Свой шаблон", "", [TemplateStep(key="a", label="A")])

        ok = store.delete(BUILTIN_ID)
        assert ok is True
        assert store.get(BUILTIN_ID) is None


def test_last_remaining_template_cannot_be_deleted_even_if_builtin() -> None:
    """Правило «нельзя удалить последний оставшийся шаблон» действует независимо от builtin —
    анкету всегда должно быть чем пройти."""
    with tempfile.TemporaryDirectory() as tmp:
        store = TemplateStore(Path(tmp))   # только что засеянный builtin — единственный шаблон
        raised = False
        try:
            store.delete(BUILTIN_ID)
        except ValueError:
            raised = True
        assert raised, "последний оставшийся шаблон (даже builtin) удалять нельзя"
        assert store.get(BUILTIN_ID) is not None


def main() -> None:
    test_builtin_template_can_be_deleted_when_another_exists()
    test_last_remaining_template_cannot_be_deleted_even_if_builtin()
    print("\nTEST_TEMPLATES OK ✅")


if __name__ == "__main__":
    main()
