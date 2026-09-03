"""Проверка хранилища проектов/дел и построения списка допросов проекта (Блок 7).

Запуск: `py -3.11 -m app.test_projects` или `pytest app/test_projects.py -v`
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import config
from .assistant import sessions_index
from .assistant.projects import ProjectStore
from .models import Protocol, QuestionnaireField, Segment, Template, TemplateStep


def test_project_store_create_list_get() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ProjectStore(Path(tmp))
        assert store.list() == []
        p1 = store.create("Дело №1")
        p2 = store.create("Дело №2")
        assert store.get(p1.id).name == "Дело №1"
        assert {p.id for p in store.list()} == {p1.id, p2.id}
        assert store.get("nonexistent") is None


def test_list_project_sessions_filters_by_project_and_extracts_fio_and_date() -> None:
    """Реальный сценарий (Блок 7): в хранилище лежат допросы из РАЗНЫХ проектов вперемешку
    (один файл на сессию, как и с шаблонами) — список должен вернуть только свои, с именем
    бланка, датой и ФИО, собранным по шагам с placeholder'ами T1.*SURNAME/NAME (порядок шагов —
    порядок появления в бланке, см. docx_import — «Фамилия» раньше «Имени»)."""
    with tempfile.TemporaryDirectory() as tmp:
        storage_dir = Path(tmp)
        original_dir = config.STORAGE_DIR
        config.STORAGE_DIR = storage_dir
        try:
            steps = [
                TemplateStep(key="surname", label="Фамилия", placeholder="T1.SURNAME"),
                TemplateStep(key="name", label="Имя", placeholder="T1.NAME"),
            ]
            tmpl = Template(id="t1", name="Протокол допроса обвиняемого", steps=steps)
            date_ts = datetime(2025, 1, 15, 10, 0, 0).timestamp()

            proto_a = Protocol(
                session_id="sess-a", project_id="proj-1",
                template_name=tmpl.name, template_snapshot=tmpl,
                questionnaire=[
                    QuestionnaireField(key="surname", label="Фамилия", value="Иванов"),
                    QuestionnaireField(key="name", label="Имя", value="Пётр"),
                ],
                segments=[Segment(channel=1, text="ответ", created_at=date_ts)],
            )
            proto_b = Protocol(session_id="sess-b", project_id="proj-2")   # другой проект
            corrupt_dir = storage_dir / "sess-corrupt"
            corrupt_dir.mkdir(parents=True)
            (corrupt_dir / "protocol.json").write_text("{не json", encoding="utf-8")

            for proto in (proto_a, proto_b):
                d = storage_dir / proto.session_id
                d.mkdir(parents=True)
                (d / "protocol.json").write_text(proto.model_dump_json(), encoding="utf-8")

            result = sessions_index.list_project_sessions("proj-1")
            print(f"результат: {result}")
            assert len(result) == 1, "битый файл и чужой проект не должны попасть в список"
            assert result[0].session_id == "sess-a"
            assert result[0].template_name == "Протокол допроса обвиняемого"
            assert result[0].display_name == "Иванов Пётр"
            assert result[0].date == "15.01.2025"
        finally:
            config.STORAGE_DIR = original_dir


def main() -> None:
    test_project_store_create_list_get()
    test_list_project_sessions_filters_by_project_and_extracts_fio_and_date()
    print("\nTEST_PROJECTS OK ✅")


if __name__ == "__main__":
    main()
