"""Интеграционная проверка диаризации общего микрофона (нужен speechbrain + sklearn).

Проверяет, что пайплайн (ECAPA-эмбеддинги + кластеризация) отрабатывает и
расставляет метки голосов. Качество на синтетике не оценивается — реальная
проверка на записи двух людей в одном микрофоне.

Запуск:  py -3.11 -m app.test_diarize
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
from .finalize.diarizer import Diarizer
from .models import Segment
from .session import manager


def _tone(freq: float, dur: float, sr: int) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    sig = 0.1 * (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t))
    return sig.astype(np.float32)


def main() -> None:
    sr = config.SAMPLE_RATE
    # «два голоса»: разные тембры; по два сегмента на каждый, по очереди
    blocks = [(_tone(120, 1.5, sr), 120), (_tone(240, 1.5, sr), 240),
              (_tone(120, 1.5, sr), 120), (_tone(240, 1.5, sr), 240)]
    audio = np.concatenate([b for b, _ in blocks])
    pcm = np.clip(audio * 32768, -32768, 32767).astype("<i2")

    s = manager.create("diar-test")
    with wave.open(str(s.channel_wav_path(0)), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    s._closed = True

    pos = 0.0
    for _, freq in blocks:
        s.protocol.segments.append(Segment(
            channel=0, speaker="Интервьюер", start=pos, end=pos + 1.5,
            text=f"фрагмент {freq}"))
        pos += 1.5

    print("загрузка ECAPA и кластеризация (первый раз — скачивание модели)…")
    n = s.diarize_single_mic(Diarizer(), channel=0, num_speakers=2)
    print(f"найдено голосов: {n}")
    for seg in s.protocol.segments:
        print(f"  {seg.start:.1f}-{seg.end:.1f}  {seg.speaker}")

    assert n == 2, "при num_speakers=2 ожидаем 2 кластера"
    assert all(seg.speaker.startswith("Голос-") for seg in s.protocol.segments)
    print("\nDIARIZE INTEGRATION OK ✅")


if __name__ == "__main__":
    main()
