"""Интеграционная проверка alignment-финализации (нужны torch + wav2vec2-ru).

Проверяет, что пайплайн загружает модель, выполняет forced alignment и возвращает
слова с возрастающими тайм-кодами в пределах аудио. Качество выравнивания на
синтетике не оценивается — это проверяется на реальной речи в приложении.

Запуск:  py -3.11 -m app.test_align
"""
from __future__ import annotations

import sys
import wave

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import config
from .finalize.aligner import WordAligner
from .models import Segment, Word
from .session import manager


def main() -> None:
    # синтетическая «речь»: несколько секунд модулированного тона
    sr = config.SAMPLE_RATE
    dur = 3.0
    t = np.arange(int(dur * sr)) / sr
    audio = (0.1 * np.sin(2 * np.pi * 160 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)))
    pcm = np.clip(audio * 32768, -32768, 32767).astype("<i2")

    s = manager.create("align-test")
    p = s.channel_wav_path(0)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    s._closed = True  # WAV уже записан вручную

    seg = Segment(channel=0, speaker="Опрашиваемый", start=0.2, end=2.8,
                  text="Привет, как дела сегодня?",
                  text_original="Привет, как дела сегодня?",
                  words=[Word(text="привет", start=0.2, end=2.8)])
    s.protocol.segments.append(seg)

    print("загрузка модели и выравнивание (первый раз — скачивание ~1.2 ГБ)…")
    n = s.finalize_alignment(WordAligner())
    print(f"обработано сегментов: {n}")
    seg = s.protocol.segments[0]
    print(f"aligned={seg.aligned}, слов={len(seg.words)}")
    for w in seg.words:
        print(f"  {w.text!r}: {w.start:.2f}-{w.end:.2f}")

    assert seg.aligned, "сегмент должен быть выровнен"
    assert len(seg.words) >= 1
    starts = [w.start for w in seg.words]
    assert starts == sorted(starts), "тайм-коды слов должны возрастать"
    assert seg.words[0].start >= 0 and seg.words[-1].end <= 3.0 + 0.3

    # Выравнивание уточняет ТАЙМ-КОДЫ и не должно переписывать текст. Внутри для сопоставления
    # с CTC-эмиссией слова приводятся к нижнему регистру (словарь wav2vec2 строчный), и раньше
    # наружу отдавались именно они: протокол рисуется по словам, поэтому после нажатия
    # «Уточнить тайм-коды» «Миша» превращалось в «миша», а точки и запятые пропадали.
    joined = " ".join(w.text for w in seg.words)
    assert any(c.isupper() for c in joined), f"регистр потерян при выравнивании: {joined!r}"
    assert any(c in ",?." for c in joined), f"пунктуация потеряна при выравнивании: {joined!r}"
    print("\nALIGN INTEGRATION OK ✅")


if __name__ == "__main__":
    main()
