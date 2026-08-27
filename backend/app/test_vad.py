"""Проверка адаптивного порога шума (Блок 3.6 плана): `NoiseFloorTracker` должен уметь
«переучиться» на новый уровень фонового шума, а не застревать на стартовом минимуме — это была
логическая ошибка в исходной формулировке (порог обновлялся только на кадрах, уже признанных
тишиной ЭТИМ ЖЕ порогом — замкнутая зависимость).

Важно про амплитуды: для синусоиды `RMS = amplitude/√2 ≈ 0.707*amplitude`, а не сама amplitude —
исходный пример в плане («amp=0.006 на границе порога 0.008») этого не учитывал и на деле давал
RMS≈0.0042, что вообще не проверяло пограничный случай. Здесь амплитуды подобраны через RMS явно.

Запуск: `py -3.11 -m app.test_vad`
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .audio.noise import NoiseFloorTracker


def _tone_rms(seconds: float, rms: float, rate: int = 16000, freq: float = 180.0) -> np.ndarray:
    """Синус с заданным RMS (не amplitude!): amplitude = rms*sqrt(2)."""
    n = int(seconds * rate)
    amp = rms * (2 ** 0.5)
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / rate)


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) if frame.size else 0.0


def test_adaptive_floor_learns_new_noise_level() -> None:
    """Комната шумнее дефолтного минимума (0.008): фиксированный порог классифицировал бы этот
    шум как речь навсегда. Адаптивный трекер должен за несколько секунд «переучиться» и снова
    отличать настоящую речь (в разы громче шума) от этого шума."""
    tracker = NoiseFloorTracker(floor_min=0.008, rise_alpha=0.05, fall_alpha=0.15)
    frame_sec = 0.1
    noise_rms = 0.012          # выше дефолтного минимума 0.008 — старый фиксированный порог
                                 # классифицировал бы это как речь ВСЕГДА
    speech_rms = 0.05           # заметно громче шума (~12.4 дБ)

    # Кормим ~5 c шума помещения — трекер должен подняться к его уровню.
    classified_as_speech_during_noise = 0
    for _ in range(50):
        frame = _tone_rms(frame_sec, noise_rms)
        r = _rms(frame)
        if tracker.classify(r, margin_db=6.0):
            classified_as_speech_during_noise += 1

    floor_after = tracker.floor
    print(f"пол шума после {50*frame_sec:.1f}c шума ({noise_rms}): {floor_after:.4f} "
          f"(старт был {0.008})")
    assert floor_after > 0.008, "трекер должен подняться выше стартового минимума"
    # Последние кадры (когда трекер уже адаптировался) не должны считаться речью.
    late_frame = _tone_rms(frame_sec, noise_rms)
    assert not tracker.classify(_rms(late_frame), margin_db=6.0), \
        "адаптированный трекер не должен путать устоявшийся шум с речью"

    # А настоящая, заметно более громкая речь — по-прежнему речь.
    speech_frame = _tone_rms(frame_sec, speech_rms)
    assert tracker.classify(_rms(speech_frame), margin_db=6.0), \
        "речь, заметно громче адаптированного пола, должна распознаваться как речь"

    print(f"классифицировано как речь во время адаптации: {classified_as_speech_during_noise}/50 "
          f"(ожидаем убывание к 0 по мере адаптации)")
    print("ADAPTIVE_FLOOR OK ✅")


def test_no_circular_dependency() -> None:
    """Ключевая проверка исходного бага: трекер обновляется НЕЗАВИСИМО от классификации —
    даже если кадр классифицирован как речь, порог всё равно понемногу сдвигается к нему."""
    tracker = NoiseFloorTracker(floor_min=0.008)
    loud = 0.05
    floor_before = tracker.floor
    for _ in range(20):
        tracker.classify(loud, margin_db=6.0)   # всегда классифицируется как речь (loud >> floor)
    assert tracker.floor > floor_before, (
        "порог обязан сдвигаться даже на кадрах, классифицированных как речь — "
        "иначе воспроизводится циклическая зависимость исходного дизайна"
    )
    print(f"пол шума сдвинулся с {floor_before:.4f} до {tracker.floor:.4f} несмотря на то, что "
          f"все кадры были классифицированы как речь")
    print("NO_CIRCULAR_DEPENDENCY OK ✅")


def main() -> None:
    test_no_circular_dependency()
    test_adaptive_floor_learns_new_noise_level()
    print("\nTEST_VAD OK ✅")


if __name__ == "__main__":
    main()
