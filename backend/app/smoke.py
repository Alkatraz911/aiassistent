"""Smoke-тест пайплайна без сети и UI.

Прогоняет синтетическое аудио через сессию: VAD-чанкинг -> ASR -> сегменты,
проверяет маркировку спикеров, правку с аудитом, сценарий анкеты и сохранение.
Запуск:  ASR_PROVIDER=stub  py -3.11 -m app.smoke
"""
from __future__ import annotations

import sys

import numpy as np

# Консоль Windows может быть cp1251 — выводим протокол/кириллицу в UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .session import manager


def _speech(seconds: float, rate: int = 16000, amp: float = 0.1) -> np.ndarray:
    n = int(seconds * rate)
    tone = (amp * np.sin(2 * np.pi * 180 * np.arange(n) / rate))
    return (tone * 32767).astype("<i2")


def _silence(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype="<i2")


def main() -> None:
    s = manager.create("smoke")

    # --- анкета ---
    step = s.assistant.start()
    print("assistant start:", step.key, "-", step.prompt[:40], "...")
    s.assistant.advance()                       # пропустить greeting (info)
    s.assistant.submit_answer("Иванов Иван Иванович")
    s.assistant.submit_answer("12.05.1990")
    print("fields after 2 answers:", s.assistant.answers)

    # --- стриминг: канал 0 (Интервьюер), канал 1 (Опрашиваемый) ---
    for ch in (0, 1):
        for part in (_speech(1.2), _silence(0.7), _speech(1.0), _silence(0.7)):
            s.ingest(ch, part)
        s.finalize_channel(ch)

    print(f"segments: {len(s.protocol.segments)}")
    for seg in s.protocol.segments:
        print(f"  [{seg.speaker}] {seg.start:.2f}-{seg.end:.2f}: {seg.text!r} "
              f"({len(seg.words)} слов)")

    assert s.protocol.segments, "ожидались сегменты"

    # --- маркировка спикера ---
    n = s.set_speaker(1, "Помощник")
    print(f"переименован канал 1 -> Помощник, обновлено сегментов: {n}")
    assert all(seg.speaker == "Помощник" for seg in s.protocol.segments if seg.channel == 1)

    # --- правка текста с аудитом ---
    seg0 = s.protocol.segments[0]
    s.edit_segment(seg0.id, "исправленный текст")
    assert seg0.edited and seg0.text == "исправленный текст"
    assert seg0.text_original != seg0.text, "оригинал должен сохраниться"
    assert seg0.edits and seg0.edits[-1].old != seg0.edits[-1].new
    print(f"правка ок: original={seg0.text_original!r} -> text={seg0.text!r}, "
          f"audit-записей: {len(seg0.edits)}")

    # --- сохранение ---
    path = s.save()
    s.close()
    print("сохранено:", path)
    print("\nSMOKE OK ✅")


if __name__ == "__main__":
    main()
