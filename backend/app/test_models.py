"""Проверка нормализации токена плейсхолдера (Блок 6) — `models.normalize_placeholder_token` и
валидаторы `TemplateStep.placeholder`/`Template.qa_placeholder`.

Реальный случай на живых данных: в `storage/operator_profile.json` пользователя оказались ДВЕ
РАЗНЫЕ записи — `T1.DOC_AUTHOR_FULL_INFO` и `T1.DOC_AUTHOR_FULL_INFO}` (с опечаткой-скобкой на
конце) — потому что свободное текстовое поле «Плейсхолдер .docx» в редакторе шаблона не
нормализовалось на вводе, и обе строки-ключа физически разные, хотя должны быть одним полем.

Запуск: `pytest app/test_models.py -v`
"""
from __future__ import annotations

from .models import Template, TemplateStep, normalize_placeholder_token


def test_normalize_placeholder_token_strips_stray_braces_and_hash() -> None:
    assert normalize_placeholder_token("T1.DOC_AUTHOR_FULL_INFO}") == "T1.DOC_AUTHOR_FULL_INFO"
    assert normalize_placeholder_token("#{T1.DOC_AUTHOR_FULL_INFO}") == "T1.DOC_AUTHOR_FULL_INFO"
    assert normalize_placeholder_token("  T1.SURNAME  ") == "T1.SURNAME"
    assert normalize_placeholder_token("") == ""


def test_template_step_placeholder_is_normalized_on_construction() -> None:
    step = TemplateStep(key="author", label="Автор документа",
                         placeholder="T1.DOC_AUTHOR_FULL_INFO}", source="profile")
    assert step.placeholder == "T1.DOC_AUTHOR_FULL_INFO", \
        "опечатка-скобка не должна создавать вторую, отдельную от настоящего токена запись"


def test_template_qa_placeholder_is_normalized_on_construction() -> None:
    tmpl = Template(id="t1", name="Т", qa_placeholder="#{T1.TRANSCRIPT}")
    assert tmpl.qa_placeholder == "T1.TRANSCRIPT"

    tmpl_none = Template(id="t2", name="Т2", qa_placeholder=None)
    assert tmpl_none.qa_placeholder is None, "пустой/отсутствующий плейсхолдер не должен превращаться в пустую строку"


def main() -> None:
    test_normalize_placeholder_token_strips_stray_braces_and_hash()
    test_template_step_placeholder_is_normalized_on_construction()
    test_template_qa_placeholder_is_normalized_on_construction()
    print("\nTEST_MODELS OK ✅")


if __name__ == "__main__":
    main()
