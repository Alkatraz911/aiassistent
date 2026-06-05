"""AI-ассистент вводной части: сценарий анкеты как state machine.

Каждый шаг: ассистент произносит реплику (озвучивается на клиенте офлайн-TTS),
получает ответ опрашиваемого (расшифрованный ASR), извлекает значение поля и
переходит дальше. Часть шагов — разъяснения прав с контрольным вопросом
«понятно ли», ответ фиксируется в протокол.

Извлечение значений здесь правило-ориентированное (regex/нормализация). На
боевом сервере шаг extract можно заменить на локальную LLM (Ollama/vLLM,
function-calling по JSON-схеме анкеты) — интерфейс шага не изменится.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Step:
    key: str
    label: str
    prompt: str                       # что говорит ассистент
    kind: str = "field"               # field | confirm | info
    extract: Optional[Callable[[str], str]] = None


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


# --- сценарий ------------------------------------------------------------

SCRIPT: list[Step] = [
    Step("greeting", "Приветствие", kind="info",
         prompt="Здравствуйте. Я ассистент, который поможет оформить вводную "
                "часть протокола. Сейчас задам несколько вопросов."),
    Step("fio", "ФИО",
         prompt="Назовите, пожалуйста, вашу фамилию, имя и отчество полностью.",
         extract=extract_fio),
    Step("birth", "Дата рождения",
         prompt="Назовите дату вашего рождения.",
         extract=extract_birth),
    Step("address", "Адрес проживания",
         prompt="Назовите адрес вашего проживания.",
         extract=_clean),
    Step("rights", "Разъяснение прав", kind="confirm",
         prompt="Разъясняю: вы вправе не свидетельствовать против себя и близких, "
                "пользоваться родным языком и помощью переводчика, приносить "
                "замечания. Вам понятны ваши права? Ответьте «да» или «нет».",
         extract=extract_yesno),
    Step("purpose", "Понимание цели", kind="confirm",
         prompt="Цель беседы — зафиксировать ваши пояснения по существу. "
                "Вам понятна цель проводимого действия?",
         extract=extract_yesno),
]


@dataclass
class AssistantSession:
    idx: int = -1
    answers: dict[str, str] = field(default_factory=dict)
    finished: bool = False

    def current(self) -> Optional[Step]:
        if 0 <= self.idx < len(SCRIPT):
            return SCRIPT[self.idx]
        return None

    def start(self) -> Step:
        self.idx = 0
        return SCRIPT[0]

    def submit_answer(self, raw_answer: str) -> dict:
        """Сохраняет ответ на текущий шаг, возвращает next-инструкцию для клиента."""
        step = self.current()
        if step is not None and step.kind != "info":
            value = step.extract(raw_answer) if step.extract else _clean(raw_answer)
            self.answers[step.key] = value
        return self.advance()

    def advance(self) -> dict:
        self.idx += 1
        if self.idx >= len(SCRIPT):
            self.finished = True
            return {"finished": True, "answers": self.answers}
        step = SCRIPT[self.idx]
        return {
            "finished": False,
            "step": step.key,
            "label": step.label,
            "kind": step.kind,
            "prompt": step.prompt,
            "needs_answer": step.kind != "info",
        }

    def to_fields(self) -> list[dict]:
        out = []
        for s in SCRIPT:
            if s.kind == "info":
                continue
            out.append({
                "key": s.key, "label": s.label,
                "value": self.answers.get(s.key, ""),
                "confirmed": s.key in self.answers,
            })
        return out
