"""AI-ассистент вводной части: сценарий анкеты как state machine.

Каждый шаг: ассистент произносит реплику (озвучивается на клиенте офлайн-TTS: сначала
разъяснительный текст `statement`, если есть, затем сам вопрос `question`), получает ответ
опрашиваемого (расшифрованный ASR), извлекает значение поля и переходит дальше. Часть шагов —
разъяснения прав с контрольным вопросом «понятно ли», ответ фиксируется в протокол.

Сценарий (`script`) больше не жёстко зашит: `AssistantSession.load_script()` принимает список
шагов, собранный из пользовательского шаблона (Блок 5, см. `templates.py`). `SCRIPT` /
`DEFAULT_TEMPLATE_STEPS` ниже — дефолт/сид для builtin-шаблона, сохраняет обратную совместимость
там, где шаблон явно не выбран (например `smoke.py`).

Извлечение значений здесь правило-ориентированное (regex/нормализация). На боевом сервере шаг
extract можно заменить на локальную LLM (Ollama/vLLM, function-calling по JSON-схеме анкеты) —
интерфейс шага не изменится; строковый `extractor` в `TemplateStep` — та точка расширения.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..models import TemplateStep


@dataclass
class Step:
    key: str
    label: str
    kind: str = "field"               # field | confirm | info
    statement: str = ""               # текст для зачитывания (разъяснение, может быть пустым)
    question: str = ""                # вопрос, требующий ответа (может быть пустым для info)
    # Источник значения (Блок 6, см. models.TemplateStep.source) — клиенту нужен, чтобы для
    # source="manual" не запускать голосовое автослушание, а сразу дать текстовое поле
    # (см. main.py::_step_payload и showStep в app.js).
    source: str = "asr"
    extract: Optional[Callable[[str], str]] = None

    @property
    def prompt(self) -> str:
        """Полный текст шага одной строкой — для обратной совместимости API/клиента, которые
        ещё не разделяют statement/question по отдельности. Черновики, импортированные из .docx,
        часто остаются без текста вопроса (оператор донастраивает формулировки позже) — без
        подстраховки на `label` шаг звучал бы и выглядел как полностью пустой (реальная жалоба:
        «ассистент их вообще не спрашивает» — технически спрашивал, но пустой фразой)."""
        text = " ".join(p for p in (self.statement, self.question) if p)
        return text or self.label


# --- извлекатели значений из реплики ------------------------------------

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,!?")


def extract_fio(answer: str) -> str:
    # берём словосочетание из 2-4 слов с заглавных, иначе всю фразу
    caps = re.findall(r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,3}", answer)
    return caps[0] if caps else _clean(answer)


def extract_birth(answer: str) -> str:
    m = re.search(r"\d{1,2}[.\s/-]\d{1,2}[.\s/-]\d{2,4}", answer)
    if m:
        return m.group(0)
    m = re.search(r"\d{4}\s*год", answer)
    return m.group(0) if m else _clean(answer)


def extract_yesno(answer: str) -> str:
    a = answer.lower()
    if re.search(r"\b(да|понятно|понимаю|согласен|ясно)\b", a):
        return "да"
    if re.search(r"\b(нет|не понятно|не понимаю|не согласен)\b", a):
        return "нет"
    return _clean(answer)


def extract_none(_answer: str) -> str:
    return ""


# Реестр извлекателей по строковому ключу — нужен, чтобы шаблон анкеты (Блок 5) был
# JSON-сериализуемым: TemplateStep.extractor хранит имя, не сам callable.
EXTRACTORS: dict[str, Callable[[str], str]] = {
    "fio": extract_fio,
    "birth": extract_birth,
    "yesno": extract_yesno,
    "plain": _clean,
    "none": extract_none,
}


def build_script(steps: list[TemplateStep]) -> list[Step]:
    """Собирает рантайм-шаги (`Step`) из шаблона (`TemplateStep`, Блок 5).

    Шаги с `source="profile"` (Блок 6) в голосовой сценарий не попадают вовсе: их значение —
    не про этого опрашиваемого и не про эту сессию, а про оператора программы («автор
    документа» и т.п.), один раз задаётся в профиле оператора и оттуда же берётся при генерации
    .docx (см. `docgen.render`). Спрашивать его голосом на каждом допросе бессмысленно.

    Шаги с `source="auto"` (Блок 6) — по той же причине: значение (дата, время начала/окончания
    опроса) вычисляется из данных сессии при генерации документа (см. `docgen.AUTO_FIELDS`), а
    не отвечается голосом."""
    return [
        Step(key=s.key, label=s.label, kind=s.kind, statement=s.statement, question=s.question,
             source=s.source, extract=EXTRACTORS.get(s.extractor, _clean))
        for s in steps if s.source not in ("profile", "auto")
    ]


# --- дефолтный сценарий (сид builtin-шаблона) -----------------------------

DEFAULT_TEMPLATE_STEPS: list[TemplateStep] = [
    TemplateStep(key="greeting", label="Приветствие", kind="info",
                 statement="Здравствуйте. Я ассистент, который поможет оформить вводную "
                           "часть протокола. Сейчас задам несколько вопросов."),
    TemplateStep(key="fio", label="ФИО", kind="field",
                 question="Назовите, пожалуйста, вашу фамилию, имя и отчество полностью.",
                 extractor="fio"),
    TemplateStep(key="birth", label="Дата рождения", kind="field",
                 question="Назовите дату вашего рождения.", extractor="birth"),
    TemplateStep(key="address", label="Адрес проживания", kind="field",
                 question="Назовите адрес вашего проживания.", extractor="plain"),
    TemplateStep(key="rights", label="Разъяснение прав", kind="confirm",
                 statement="Разъясняю: вы вправе не свидетельствовать против себя и близких, "
                           "пользоваться родным языком и помощью переводчика, приносить замечания.",
                 question="Вам понятны ваши права? Ответьте «да» или «нет».",
                 extractor="yesno"),
    TemplateStep(key="purpose", label="Понимание цели", kind="confirm",
                 statement="Цель беседы — зафиксировать ваши пояснения по существу.",
                 question="Вам понятна цель проводимого действия?",
                 extractor="yesno"),
]

SCRIPT: list[Step] = build_script(DEFAULT_TEMPLATE_STEPS)


@dataclass
class AssistantSession:
    idx: int = -1
    script: list[Step] = field(default_factory=lambda: list(SCRIPT))
    answers: dict[str, str] = field(default_factory=dict)
    finished: bool = False
    # Контекст для ASR-распознавания ответов анкеты (Блок 6, тот же приём, что `Session.
    # _remember_prompt` для потоковых каналов, — накопленные ПОДТВЕРЖДЁННЫЕ ответы этой же
    # сессии). Без него `/api/transcribe` каждый раз получал только статичный WHISPER_PROMPT
    # без какого-либо контекста, и короткие ответы (даты, ФИО) распознавались заметно хуже, чем
    # реплики основного потока, у которого initial_prompt — живой хвост распознанной речи.
    recent_context: str = ""

    def load_script(self, script: list[Step]) -> None:
        """Переключить сценарий (выбор шаблона, Блок 5) — сбрасывает прогресс анкеты."""
        self.script = script
        self.idx = -1
        self.answers = {}
        self.finished = False

    def _remember_context(self, text: str) -> None:
        words = (self.recent_context + " " + text).split()
        self.recent_context = " ".join(words[-60:])

    def current(self) -> Optional[Step]:
        if 0 <= self.idx < len(self.script):
            return self.script[self.idx]
        return None

    def start(self) -> Step:
        self.idx = 0
        return self.script[0]

    def submit_answer(self, raw_answer: str) -> dict:
        """Сохраняет ответ на текущий шаг, возвращает next-инструкцию для клиента."""
        step = self.current()
        if step is not None and step.kind != "info":
            value = step.extract(raw_answer) if step.extract else _clean(raw_answer)
            self.answers[step.key] = value
            # Сырой (не извлечённый/нормализованный) ответ — ближе к тому, что реально прозвучит
            # дальше по стилю/лексике, чем нормализованная дата или ФИО без падежей.
            if raw_answer.strip():
                self._remember_context(raw_answer)
        return self.advance()

    def set_answer(self, key: str, value: str) -> bool:
        """Прямая правка значения поля анкеты по ключу — независимо от текущего шага `idx`
        (Блок 6). Нужна по двум причинам: (1) окно автоподтверждения голосового ответа висит
        всего пару секунд — оператор часто не успевает нажать «исправить» до автоподтверждения
        (реальная жалоба пользователя); (2) поля с `source="manual"` вообще не рассчитаны на
        голосовой ответ — таблица полей анкеты должна быть основным способом их заполнения, а не
        только резервным способом правки. Возвращает `False`, если такого ключа нет в текущем
        сценарии (info-шаги в анкету значений не пишут вовсе — см. `to_fields`)."""
        for step in self.script:
            if step.key == key and step.kind != "info":
                self.answers[key] = _clean(value)
                return True
        return False

    def advance(self) -> dict:
        self.idx += 1
        if self.idx >= len(self.script):
            self.finished = True
            return {"finished": True, "answers": self.answers}
        step = self.script[self.idx]
        return {
            "finished": False,
            "step": step.key,
            "label": step.label,
            "kind": step.kind,
            "prompt": step.prompt,
            "statement": step.statement,
            "question": step.question,
            "needs_answer": step.kind != "info",
            "source": step.source,
        }

    def to_fields(self) -> list[dict]:
        out = []
        for s in self.script:
            if s.kind == "info":
                continue
            out.append({
                "key": s.key, "label": s.label,
                "value": self.answers.get(s.key, ""),
                "confirmed": s.key in self.answers,
            })
        return out
