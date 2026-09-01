"""Проверка импорта .docx-бланков протокола (Блок 6): поиск и нормализация DB mail-merge
токенов `#{NS.FIELD}`.

Ключевой случай — Word нередко режет один токен на несколько `<w:r>`-ранов (спелл-чекер/
автозамена), поэтому здесь плейсхолдер СОЗНАТЕЛЬНО собирается из нескольких `add_run()` с
разрывом посередине токена: наивная замена по одному рану за раз такое пропустит.

Запуск: `py -3.11 -m app.test_docx_import` или `pytest app/test_docx_import.py -v`
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from docx import Document

from .assistant.docx_import import normalize_docx, scan_placeholders, uncovered_placeholders
from .models import Template, TemplateStep


def _build_fixture_docx() -> io.BytesIO:
    doc = Document()

    # Абзац 1: плейсхолдер разбит на 3 рана ровно посередине токена — имитация
    # автозамены/спелл-чекера Word.
    p1 = doc.add_paragraph()
    p1.add_run("#{T1.FO")
    p1.add_run("O}")

    # Абзац 2: текст до и после плейсхолдера должен пережить замену.
    p2 = doc.add_paragraph()
    p2.add_run("Значение: ")
    p2.add_run("#{T1.BAR} ")
    p2.add_run("конец.")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def test_scan_placeholders_finds_split_and_inline_tokens() -> None:
    buf = _build_fixture_docx()
    tokens = scan_placeholders(buf)
    print(f"найденные токены: {tokens}")
    assert tokens == ["T1.BAR", "T1.FOO"]


def test_normalize_docx_rewrites_tokens_and_keeps_surrounding_text() -> None:
    buf = _build_fixture_docx()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "normalized.docx"
        found = normalize_docx(buf, out_path)
        print(f"нормализовано токенов: {found}")
        assert found == ["T1.BAR", "T1.FOO"]
        doc = Document(out_path)
    full_text = "\n".join("".join(r.text for r in p.runs) for p in doc.paragraphs)
    print(f"текст после нормализации:\n{full_text}")

    assert "{{ T1_FOO }}" in full_text
    assert "{{ T1_BAR }}" in full_text
    assert "#{" not in full_text
    assert "Значение:" in full_text
    assert "конец." in full_text


def test_uncovered_placeholders_flags_step_without_matching_token() -> None:
    """Реальный случай из багрепорта: оператор при редактировании шаблона удалил шаг или
    переименовал плейсхолдер с опечаткой — токен остаётся в .docx, но ни один шаг на него не
    ссылается. `T1.BAR` покрыт шагом, `T1.FOO` — нет (ни как placeholder шага, ни как
    qa_placeholder)."""
    buf = _build_fixture_docx()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "normalized.docx"
        normalize_docx(buf, out_path)
        tmpl = Template(
            id="t", name="t", docx_filename="normalized.docx",
            steps=[TemplateStep(key="bar", label="Bar", placeholder="T1.BAR")],
        )
        missing = uncovered_placeholders(tmpl, out_path)
        print(f"непокрытые плейсхолдеры: {missing}")
        assert missing == ["T1_FOO"]

        # А если T1.FOO отдан под стенограмму — покрытие полное.
        tmpl_with_qa = tmpl.model_copy(update={"qa_placeholder": "T1.FOO"})
        assert uncovered_placeholders(tmpl_with_qa, out_path) == []


def main() -> None:
    test_scan_placeholders_finds_split_and_inline_tokens()
    test_normalize_docx_rewrites_tokens_and_keeps_surrounding_text()
    test_uncovered_placeholders_flags_step_without_matching_token()
    print("\nTEST_DOCX_IMPORT OK ✅")


if __name__ == "__main__":
    main()
