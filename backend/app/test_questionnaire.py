"""Проверка сборки голосового сценария анкеты из шаблона (Блок 5/6).

Запуск: `py -3.11 -m app.test_questionnaire` или `pytest app/test_questionnaire.py -v`
"""
from __future__ import annotations

from .assistant.questionnaire import AssistantSession, build_script
from .models import TemplateStep


def test_build_script_skips_profile_source_steps() -> None:
    """Шаги с source="profile" (Блок 6, напр. «автор документа») — не про эту сессию и не про
    опрашиваемого, значение приходит из профиля оператора (см. docgen.render), поэтому в
    голосовой сценарий вообще не должны попадать."""
    steps = [
        TemplateStep(key="fio", label="ФИО", kind="field", question="Ваше ФИО?"),
        TemplateStep(key="author", label="Автор документа", kind="field",
                     question="Кто автор?", source="profile"),
        TemplateStep(key="addr", label="Адрес", kind="field", question="Ваш адрес?",
                     source="manual"),
    ]
    script = build_script(steps)
    keys = [s.key for s in script]
    assert keys == ["fio", "addr"], "profile-шаг должен быть вырезан, manual — остаться"


def test_submit_answer_accumulates_recent_context() -> None:
    """Блок 6: подтверждённые ответы анкеты копятся в `recent_context` — это то, что
    main.py::transcribe_oneshot подмешивает в initial_prompt следующего ответа той же сессии."""
    steps = [
        TemplateStep(key="fio", label="ФИО", kind="field", question="Ваше ФИО?"),
        TemplateStep(key="birth", label="Дата рождения", kind="field", question="Дата рождения?"),
    ]
    session = AssistantSession()
    session.load_script(build_script(steps))
    session.start()
    session.submit_answer("Иванов Иван Иванович")
    assert session.recent_context == "Иванов Иван Иванович"
    session.submit_answer("второе июня две тысячи пятого года")
    assert session.recent_context == "Иванов Иван Иванович второе июня две тысячи пятого года"


def test_set_answer_edits_field_out_of_order() -> None:
    """Блок 6: таблица «Поля анкеты» в клиенте редактируема напрямую — правка должна работать
    независимо от того, на каком шаге сейчас анкета, и отклоняться для неизвестного ключа."""
    steps = [
        TemplateStep(key="fio", label="ФИО", kind="field", question="Ваше ФИО?"),
        TemplateStep(key="birth", label="Дата рождения", kind="field", question="Дата рождения?"),
    ]
    session = AssistantSession()
    session.load_script(build_script(steps))
    session.start()   # текущий шаг — fio, но правим birth, который ещё не задавался
    assert session.set_answer("birth", " 02.06.2005 ") is True
    assert session.answers["birth"] == "02.06.2005"
    assert session.set_answer("nonexistent", "x") is False


def test_step_prompt_falls_back_to_label_when_empty() -> None:
    """Черновики, импортированные из .docx, часто остаются без текста вопроса — без фолбэка на
    label ассистент озвучивал/показывал бы пустую строку (реальная жалоба: «вообще не
    спрашивает», хотя технически шаг посещался)."""
    steps = [TemplateStep(key="place", label="Место проведения", kind="field",
                          statement="", question="")]
    script = build_script(steps)
    assert script[0].prompt == "Место проведения"


def test_build_script_carries_source_through_to_step() -> None:
    """Клиенту нужен `source` на каждом шаге (Блок 6), чтобы для source="manual" не запускать
    голосовое автослушание — раньше это поле терялось между TemplateStep и рантайм-Step."""
    steps = [TemplateStep(key="addr", label="Адрес", kind="field", source="manual")]
    script = build_script(steps)
    assert script[0].source == "manual"


def main() -> None:
    test_build_script_skips_profile_source_steps()
    test_submit_answer_accumulates_recent_context()
    test_set_answer_edits_field_out_of_order()
    test_step_prompt_falls_back_to_label_when_empty()
    test_build_script_carries_source_through_to_step()
    print("\nTEST_QUESTIONNAIRE OK ✅")


if __name__ == "__main__":
    main()
