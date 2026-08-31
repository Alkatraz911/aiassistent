"""Проверка генерации итогового .docx-протокола (Блок 6): заполнение полей анкеты и вставка
стенограммы «Вопрос/Ответ».

Синтетический докс-фикстура строится напрямую через python-docx с уже-Jinja-плейсхолдерами
(`{{ TOKEN }}`) — то, что получится ПОСЛЕ `docx_import.normalize_docx()` на реальном бланке.

Запуск: `py -3.11 -m app.test_docgen` или `pytest app/test_docgen.py -v`
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from docx import Document

from .assistant import docgen
from .assistant.docgen import MISSING_MARKER
from .models import Protocol, QuestionnaireField, Segment, Template, TemplateStep


def _build_fixture_docx(path: Path) -> None:
    doc = Document()
    doc.add_paragraph("ФИО: {{ T1_SURNAME }}")
    doc.add_paragraph("Адрес: {{ T1_ADDRESS }}")
    doc.add_paragraph("Стенограмма:")
    doc.add_paragraph("{{ T1_TRANSCRIPT }}")
    doc.save(str(path))


def _build_protocol() -> Protocol:
    steps = [
        TemplateStep(key="fio", label="ФИО", kind="field", placeholder="T1.SURNAME"),
        TemplateStep(key="address", label="Адрес", kind="field", placeholder="T1.ADDRESS"),
        TemplateStep(key="birth", label="Дата рождения", kind="field", placeholder="T1.BIRTH"),
    ]
    tmpl = Template(
        id="tmpl1", name="Тестовый", steps=steps,
        docx_filename="tmpl1.docx", qa_placeholder="T1.TRANSCRIPT",
    )
    protocol = Protocol(
        session_id="sess1",
        questionnaire=[
            QuestionnaireField(key="fio", label="ФИО", value="Иванов Иван Иванович"),
            QuestionnaireField(key="address", label="Адрес", value="г. Москва"),
            QuestionnaireField(key="birth", label="Дата рождения", value=""),   # незаполненное
        ],
        segments=[
            Segment(channel=0, speaker="Интервьюер", start=0.0, end=1.0,
                    text="Назовите ваше имя."),
            Segment(channel=1, speaker="Опрашиваемый", start=1.0, end=2.0,
                    text="Иванов Иван"),
            Segment(channel=1, speaker="Опрашиваемый", start=2.0, end=3.0,
                    text="Иванович."),   # тот же канал подряд — должно слиться в один абзац
            Segment(channel=0, speaker="Интервьюер", start=3.0, end=4.0,
                    text="Ваш адрес?"),
            Segment(channel=1, speaker="Опрашиваемый", start=4.0, end=5.0,
                    text="Москва."),
        ],
        speaker_map={0: "Интервьюер", 1: "Опрашиваемый"},
        template_id=tmpl.id, template_name=tmpl.name, template_version=tmpl.version,
        template_snapshot=tmpl,
    )
    return protocol


def _doc_text(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def test_build_qa_transcript_merges_consecutive_same_channel() -> None:
    protocol = _build_protocol()
    transcript = docgen.build_qa_transcript(protocol.segments, protocol.speaker_map.get)
    print(f"стенограмма:\n{transcript}")
    paragraphs = transcript.split("\n\n")
    assert paragraphs == [
        "Вопрос: Назовите ваше имя.",
        "Ответ: Иванов Иван Иванович.",   # два подряд сегмента канала 1 слиты в один абзац
        "Вопрос: Ваш адрес?",
        "Ответ: Москва.",
    ]


def test_render_fills_fields_and_transcript() -> None:
    protocol = _build_protocol()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        template_path = tmp_path / "tmpl1.docx"
        _build_fixture_docx(template_path)
        out_path = tmp_path / "out" / "protocol.docx"

        result_path = docgen.render(protocol, template_path, out_path)
        assert result_path == out_path
        assert out_path.exists()

        text = _doc_text(out_path)
        print(f"итоговый документ:\n{text}")

        assert "Иванов Иван Иванович" in text
        assert "г. Москва" in text
        assert MISSING_MARKER not in text.split("Стенограмма:")[0].split("\n")[0]  # ФИО заполнено
        # birth не привязан ни к одному плейсхолдеру шаблона докса, поэтому в тексте документа
        # его отсутствие не проверяется напрямую — а вот сама разметка MISSING_MARKER
        # проверяется отдельно через контекст ниже.
        assert "Вопрос: Назовите ваше имя." in text
        assert "Ответ: Иванов Иван Иванович." in text
        assert "Вопрос: Ваш адрес?" in text
        assert "Ответ: Москва." in text


def test_render_uses_missing_marker_for_empty_answer() -> None:
    steps = [TemplateStep(key="birth", label="Дата рождения", kind="field",
                          placeholder="T1.BIRTH")]
    tmpl = Template(id="tmpl2", name="Тестовый2", steps=steps, docx_filename="tmpl2.docx")
    protocol = Protocol(
        session_id="sess2",
        questionnaire=[QuestionnaireField(key="birth", label="Дата рождения", value="")],
        template_snapshot=tmpl,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        template_path = tmp_path / "tmpl2.docx"
        doc = Document()
        doc.add_paragraph("Дата рождения: {{ T1_BIRTH }}")
        doc.save(str(template_path))
        out_path = tmp_path / "protocol2.docx"

        docgen.render(protocol, template_path, out_path)
        text = _doc_text(out_path)
        print(f"итоговый документ:\n{text}")
        assert MISSING_MARKER in text


def test_render_requires_template_snapshot() -> None:
    protocol = Protocol(session_id="sess3")
    raised = False
    try:
        docgen.render(protocol, Path("does_not_matter.docx"), Path("out.docx"))
    except ValueError:
        raised = True
    assert raised, "без template_snapshot.docx_filename render() должен поднимать ValueError"


def main() -> None:
    test_build_qa_transcript_merges_consecutive_same_channel()
    test_render_fills_fields_and_transcript()
    test_render_uses_missing_marker_for_empty_answer()
    test_render_requires_template_snapshot()
    print("\nTEST_DOCGEN OK ✅")


if __name__ == "__main__":
    main()
