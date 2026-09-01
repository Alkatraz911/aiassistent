"""Профиль оператора (Блок 6): значения полей шаблона с `source="profile"` — общие для всех
сессий одного пользователя программы (типичный пример — «автор документа»: тот же следователь
ведёт много допросов подряд, и заново диктовать/вписывать его ФИО и должность в каждом протоколе
не нужно). Заполняется один раз через клиент, дальше `docgen.render` берёт значения отсюда,
минуя голосовую анкету (см. `questionnaire.build_script`, который такие шаги из сценария
вырезает).

Один плоский JSON-файл `storage/operator_profile.json`: `{placeholder_token: значение}`. Ключ —
токен докс-плейсхолдера (`TemplateStep.placeholder or .key`), не id шаблона: одно и то же
реальное значение («автор документа») в разных шаблонах протоколов почти наверняка лежит под
одним и тем же токеном (все 4 бланка — из одной и той же DB-экспортной схемы), так что значение,
введённое один раз, подхватывается сразу везде, где встречается этот токен."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .. import config


class ProfileStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get_all(self) -> dict[str, str]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            # Файла ещё нет (профиль не заполняли) или он битый — считаем профиль пустым, а не
            # роняем генерацию документа: пустые профильные поля просто попадут под MISSING_MARKER.
            return {}

    def update(self, values: dict[str, str]) -> dict[str, str]:
        """Мёрджит переданные значения поверх текущего профиля (частичное обновление — клиент
        может прислать только изменившиеся поля) и сохраняет атомарно."""
        with self._lock:
            current = self.get_all()
            current.update(values)
            tmp_path = self._path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._path)
            return current


store = ProfileStore(config.OPERATOR_PROFILE_PATH)
