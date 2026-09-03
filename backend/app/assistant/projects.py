"""Хранилище проектов/дел (Блок 7 плана) — папка, в которую пользователь складывает проведённые
допросы. Один JSON-файл на проект в `storage/projects/<id>.json`, атомарная запись (тот же приём,
что в `templates.py`) — обрыв питания/процесса посередине сохранения не должен оставлять битый
JSON, из-за которого список проектов молча потеряет одну запись."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from .. import config
from ..models import Project


class ProjectStore:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, project_id: str) -> Path:
        return self._dir / f"{project_id}.json"

    def list(self) -> list[Project]:
        out: list[Project] = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                out.append(Project.model_validate_json(p.read_text(encoding="utf-8")))
            except Exception:
                continue   # битый/чужой файл в папке — пропускаем, не роняем список
        out.sort(key=lambda proj: proj.created_at, reverse=True)   # свежие проекты — сверху
        return out

    def get(self, project_id: str) -> Project | None:
        p = self._path(project_id)
        if not p.exists():
            return None
        try:
            return Project.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def create(self, name: str) -> Project:
        proj = Project(name=name)
        path = self._path(proj.id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(proj.model_dump_json(indent=2), encoding="utf-8")
        with self._lock:
            os.replace(tmp_path, path)   # атомарно на одной ФС (Windows/NTFS — тоже)
        return proj


store = ProjectStore(config.PROJECTS_DIR)
