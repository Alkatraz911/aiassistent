"""Запись результата в файлы: .txt (основной), .srt, .json.

Главный формат — .txt: его открывают, читают и правят руками, поэтому он собирается репликами,
а не сегментами Whisper. Whisper режет речь по фразам (2–8 секунд), и построчная выгрузка
сегментов даёт «лесенку» из обрывков, по которой невозможно читать диалог. Подряд идущие
сегменты одного голоса склеиваются в одну реплику с одним тайм-кодом — как в обычном протоколе.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import Segment, Transcript
from .progress import hms

# Пауза внутри речи ОДНОГО голоса, после которой начинаем новую реплику. Без этого монолог на
# сорок минут становится одним абзацем с единственным тайм-кодом в начале — по такому тексту
# невозможно найти место в записи.
TURN_GAP_S = 30.0
# То же для текста без разметки по голосам: там абзацы — единственная структура, поэтому
# порог короче (обычная смена реплики в диалоге).
PARAGRAPH_GAP_S = 3.0
# Максимальная длина одной реплики в тексте. Без ограничения монолог (или отказ диаризации на
# записи с одним голосом) даёт абзац на сорок минут с единственным тайм-кодом в начале — по
# такому тексту нельзя найти место в записи, а именно за этим тайм-коды и нужны.
TURN_MAX_S = 180.0


@dataclass
class Turn:
    speaker: str | None
    start: float
    end: float
    lines: list[str] = field(default_factory=list)


def group_turns(segments: list[Segment], gap_s: float | None = None) -> list[Turn]:
    """Сегменты -> реплики: подряд идущие сегменты одного голоса, без длинных пауз внутри."""
    turns: list[Turn] = []
    for seg in segments:
        limit = gap_s if gap_s is not None else (
            TURN_GAP_S if seg.speaker else PARAGRAPH_GAP_S)
        cur = turns[-1] if turns else None
        if (cur is None or cur.speaker != seg.speaker or seg.start - cur.end > limit
                or seg.end - cur.start > TURN_MAX_S):
            turns.append(Turn(speaker=seg.speaker, start=seg.start, end=seg.end,
                              lines=[seg.text]))
        else:
            cur.lines.append(seg.text)
            cur.end = seg.end
    return turns


def write_txt(path: Path, doc: Transcript, *, timestamps: bool = True,
              header: bool = True) -> None:
    """Читаемая расшифровка.

    Кодировка — utf-8 С BOM: файл открывают в Windows, и без BOM «Блокнот» старых сборок и
    Word показывают кириллицу кракозябрами. Для .srt/.json так делать нельзя (парсеры на BOM
    спотыкаются), для текста — можно и нужно.
    """
    out: list[str] = []
    if header:
        out.append(doc.source.name)
        bits = [f"запись {hms(doc.duration)}",
                f"{doc.model} ({doc.device}/{doc.compute})"]
        if doc.speakers:
            bits.append(f"голосов: {doc.speakers}")
        bits.append(f"расшифровано {doc.created}")
        out.append(" · ".join(bits))
        out.append("")

    for turn in group_turns(doc.segments):
        head = []
        if timestamps:
            head.append(f"[{hms(turn.start)}]")
        if turn.speaker:
            head.append(turn.speaker)
        if head:
            out.append(" ".join(head))
        out.extend(turn.lines)
        out.append("")

    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8-sig")


def write_srt(path: Path, doc: Transcript) -> None:
    """Субтитры: по сегменту на реплику — здесь как раз нужна нарезка Whisper, а не склейка."""
    out: list[str] = []
    for i, seg in enumerate(doc.segments, start=1):
        text = f"{seg.speaker}: {seg.text}" if seg.speaker else seg.text
        out.append(str(i))
        out.append(f"{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}")
        out.append(text)
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def write_json(path: Path, doc: Transcript) -> None:
    """Полный дамп: сегменты, голоса и пословные тайм-коды — для дальнейшей обработки
    и привязки текста к аудио."""
    import json

    payload = {
        "source": str(doc.source),
        "duration": round(doc.duration, 3),
        "language": doc.language,
        "model": doc.model,
        "device": doc.device,
        "compute": doc.compute,
        "created": doc.created,
        "speakers": doc.speakers,
        "dropped_segments": doc.dropped,
        "segments": [s.as_dict() for s in doc.segments],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

