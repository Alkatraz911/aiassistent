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
from .assistant.profile import ProfileStore
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
            QuestionnaireField(key="fio", label="ФИО", value="Иванов Иван Иванович",
                               confirmed=True),
            QuestionnaireField(key="address", label="Адрес", value="г. Москва", confirmed=True),
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


def test_build_qa_transcript_diarized_single_mic_alternates_by_speaker_label() -> None:
    """Реальный баг: `Session.diarize_single_mic` разводит голоса общего микрофона по
    `Segment.speaker` ("Голос-1"/"Голос-2"), НЕ трогая канал — все реплики остаются `channel=0`.
    Старая проверка `channel == 0` в этом случае схлопывала весь диалог в один абзац «Вопрос:» —
    ни одного «Ответ:» не появлялось. Тот же сценарий (все сегменты на одном канале), что и
    `_build_protocol`, но роль решается по метке говорящего, а не по каналу."""
    segments = [
        Segment(channel=0, speaker="Голос-1", start=0.0, end=1.0,
                text="Назовите ваше имя."),
        Segment(channel=0, speaker="Голос-2", start=1.0, end=2.0,
                text="Иванов Иван"),
        Segment(channel=0, speaker="Голос-2", start=2.0, end=3.0,
                text="Иванович."),   # тот же говорящий подряд — должно слиться в один абзац
        Segment(channel=0, speaker="Голос-1", start=3.0, end=4.0,
                text="Ваш адрес?"),
        Segment(channel=0, speaker="Голос-2", start=4.0, end=5.0,
                text="Москва."),
    ]
    transcript = docgen.build_qa_transcript(segments, {}.get)
    print(f"стенограмма (диаризованный общий микрофон):\n{transcript}")
    paragraphs = transcript.split("\n\n")
    assert paragraphs == [
        "Вопрос: Назовите ваше имя.",
        "Ответ: Иванов Иван Иванович.",
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
    """Поле НИКОГДА не отвечали (не попало в `self.answers` — см. `AssistantSession.to_fields`,
    поэтому `confirmed` тут по умолчанию False) — это и есть «не заполнено», MISSING_MARKER
    ожидаем. Обратный случай (пусто, но подтверждено оператором) — см. тест ниже."""
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


def test_render_keeps_confirmed_empty_answer_without_missing_marker() -> None:
    """Реальный случай из багрепорта: `value or MISSING_MARKER` не отличал поле, на которое
    оператор осознанно ответил пустотой (напр. «отчества нет», подтверждено голосом/правкой —
    `confirmed=True`), от поля, которое вообще не задавали. Подтверждённая пустая строка не
    должна попадать под MISSING_MARKER."""
    steps = [TemplateStep(key="patronymic", label="Отчество", kind="field",
                          placeholder="T1.PATRONYMIC")]
    tmpl = Template(id="tmpl7", name="Тестовый7", steps=steps, docx_filename="tmpl7.docx")
    protocol = Protocol(
        session_id="sess7",
        questionnaire=[QuestionnaireField(key="patronymic", label="Отчество", value="",
                                          confirmed=True)],
        template_snapshot=tmpl,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        template_path = tmp_path / "tmpl7.docx"
        doc = Document()
        doc.add_paragraph("Отчество: {{ T1_PATRONYMIC }}#")
        doc.save(str(template_path))
        out_path = tmp_path / "protocol7.docx"

        docgen.render(protocol, template_path, out_path)
        text = _doc_text(out_path)
        print(f"итоговый документ:\n{text}")
        assert "Отчество: #" in text, "подтверждённая пустая строка не должна стать MISSING_MARKER"
        assert MISSING_MARKER not in text


def test_render_uses_profile_store_for_profile_source_steps() -> None:
    """Шаг с source="profile" (Блок 6, напр. «автор документа») не приходит через анкету сессии
    вообще (questionnaire пуст) — значение должно взяться из профиля оператора."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Изолированное хранилище профиля — не трогаем реальный storage/operator_profile.json
        # пользователя. Подменяем ссылку в docgen (не в profile-модуле — docgen импортирует её
        # к себе в пространство имён через `from .profile import store as profile_store`) и
        # восстанавливаем оригинал в finally, чтобы не протечь в другие тесты.
        original_store = docgen.profile_store
        docgen.profile_store = ProfileStore(tmp_path / "operator_profile.json")
        docgen.profile_store.update({"T1.AUTHOR": "Иванов И.И., следователь"})
        try:
            steps = [TemplateStep(key="author", label="Автор документа", kind="field",
                                  placeholder="T1.AUTHOR", source="profile")]
            tmpl = Template(id="tmpl4", name="Тестовый4", steps=steps, docx_filename="tmpl4.docx")
            protocol = Protocol(
                session_id="sess4", questionnaire=[], template_snapshot=tmpl,
            )
            template_path = tmp_path / "tmpl4.docx"
            doc = Document()
            doc.add_paragraph("Автор: {{ T1_AUTHOR }}")
            doc.save(str(template_path))
            out_path = tmp_path / "protocol4.docx"

            docgen.render(protocol, template_path, out_path)
            text = _doc_text(out_path)
            print(f"итоговый документ:\n{text}")
            assert "Иванов И.И., следователь" in text
            assert MISSING_MARKER not in text
        finally:
            docgen.profile_store = original_store


def test_render_computes_auto_date_and_time_fields() -> None:
    """Блок 6: source="auto" — дата и время начала/окончания опроса вычисляются из
    Segment.created_at (wall-clock), а не спрашиваются в анкете (questionnaire пуст).

    Начало и конец — внутри ОДНОЙ И ТОЙ ЖЕ минуты (реальный случай: короткий тестовый прогон) —
    формат обязан включать секунды, иначе оба выглядели бы одинаково («14:05») даже будучи
    технически разными (реальная жалоба пользователя после фикса recording_started_at)."""
    from datetime import datetime

    t0 = datetime(2026, 9, 1, 14, 5, 1).timestamp()
    t1 = datetime(2026, 9, 1, 14, 5, 15).timestamp()
    t2 = datetime(2026, 9, 1, 14, 5, 27).timestamp()

    steps = [
        TemplateStep(key="date", label="Дата", kind="field",
                     placeholder="T1.DATE", source="auto", extractor="date"),
        TemplateStep(key="range", label="Время допроса", kind="field",
                     placeholder="T1.RANGE", source="auto", extractor="time_range"),
    ]
    tmpl = Template(id="tmpl5", name="Тестовый5", steps=steps, docx_filename="tmpl5.docx")
    protocol = Protocol(
        session_id="sess5", questionnaire=[], template_snapshot=tmpl,
        segments=[
            Segment(channel=1, speaker="Опрашиваемый", start=0.0, end=1.0,
                    text="Первая реплика.", created_at=t0),
            Segment(channel=1, speaker="Опрашиваемый", start=1.0, end=2.0,
                    text="Вторая реплика.", created_at=t1),
            Segment(channel=1, speaker="Опрашиваемый", start=2.0, end=3.0,
                    text="Последняя реплика.", created_at=t2),
        ],
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        template_path = tmp_path / "tmpl5.docx"
        doc = Document()
        doc.add_paragraph("Дата: {{ T1_DATE }}")
        doc.add_paragraph("Время: {{ T1_RANGE }}")
        doc.save(str(template_path))
        out_path = tmp_path / "protocol5.docx"

        docgen.render(protocol, template_path, out_path)
        text = _doc_text(out_path)
        print(f"итоговый документ:\n{text}")
        assert "01.09.2026" in text
        assert "14:05:01 - 14:05:27" in text
        assert MISSING_MARKER not in text


def test_render_flags_placeholder_not_covered_by_any_step() -> None:
    """Реальный случай из багрепорта: оператор отредактировал/удалил шаг, или переименовал
    плейсхолдер в редакторе шаблона с опечаткой — токен в самом .docx остался, а среди шагов
    template_snapshot ему ничего не соответствует. Раньше docxtpl тихо подставлял пустую строку;
    теперь такой токен должен получить MISSING_MARKER, а не молча пропасть."""
    steps = [TemplateStep(key="fio", label="ФИО", kind="field", placeholder="T1.FIO")]
    tmpl = Template(id="tmpl6", name="Тестовый6", steps=steps, docx_filename="tmpl6.docx")
    protocol = Protocol(
        session_id="sess6",
        questionnaire=[QuestionnaireField(key="fio", label="ФИО", value="Иванов",
                                          confirmed=True)],
        template_snapshot=tmpl,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        template_path = tmp_path / "tmpl6.docx"
        doc = Document()
        doc.add_paragraph("ФИО: {{ T1_FIO }}")
        # T1_ORPHAN есть в бланке, но ни один шаг template_snapshot на него не ссылается.
        doc.add_paragraph("Сирота: {{ T1_ORPHAN }}")
        doc.save(str(template_path))
        out_path = tmp_path / "protocol6.docx"

        docgen.render(protocol, template_path, out_path)
        text = _doc_text(out_path)
        print(f"итоговый документ:\n{text}")
        assert "Иванов" in text
        assert "Сирота: " + MISSING_MARKER in text, "плейсхолдер без шага не должен молча пустеть"


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
    test_build_qa_transcript_diarized_single_mic_alternates_by_speaker_label()
    test_render_fills_fields_and_transcript()
    test_render_uses_missing_marker_for_empty_answer()
    test_render_keeps_confirmed_empty_answer_without_missing_marker()
    test_render_uses_profile_store_for_profile_source_steps()
    test_render_computes_auto_date_and_time_fields()
    test_render_flags_placeholder_not_covered_by_any_step()
    test_render_requires_template_snapshot()
    print("\nTEST_DOCGEN OK ✅")


if __name__ == "__main__":
    main()
