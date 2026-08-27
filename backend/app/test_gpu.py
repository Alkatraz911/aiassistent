"""Проверка GPU-конфигурации: устройство, нативные библиотеки, реальная скорость декодирования.

Отвечает на три вопроса, которые иначе выясняются посреди записи:
  1. Видит ли CTranslate2 карту и нашлись ли cuBLAS/cuDNN (на Windows их приходится
     регистрировать руками — `app/device.py::ensure_cuda_dlls`).
  2. Какие дефолты выбрал config под это устройство (модель/compute/каданс/батчинг).
  3. Успевает ли выбранная модель за реальным временем НА ДВУХ каналах сразу — именно это,
     а не скорость одного прохода, определяет, поедет ли live-режим (см. README,
     «Производительность»).

Мерить обязательно на РЕЧИ. Соблазн подать шум («всё равно же гоняются те же слои») даёт числа,
которым нельзя верить: на шуме модель не выдаёт токенов, работает практически один энкодер, и
large-v3 показывает сотни «x реального времени» вместо реальных полутора десятков. Поэтому:
свой файл аргументом, иначе — синтез системным русским голосом Windows.

Запуск:
    py -3.11 -m app.test_gpu                 # речь синтезируется системным TTS
    py -3.11 -m app.test_gpu запись.wav      # своя запись (16 кГц, моно, WAV)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import config, device

TTS_LINES = [
    "Представьтесь, пожалуйста, назовите вашу фамилию, имя и отчество полностью.",
    "Иванов Сергей Петрович, тысяча девятьсот восемьдесят четвёртого года рождения.",
    "Расскажите своими словами, что произошло вечером пятнадцатого марта.",
    "Я возвращался с работы и увидел, как двое мужчин спорили около подъезда.",
    "Вы можете описать внешность этих людей более подробно?",
    "Один был высокого роста, в тёмной куртке, второго я разглядел плохо.",
]


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != config.SAMPLE_RATE:
            raise ValueError(f"нужен WAV 16 кГц моно, а тут {w.getframerate()} Гц / "
                             f"{w.getnchannels()} кан.")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _tts_wav() -> np.ndarray | None:
    """Русская речь системным синтезатором Windows (System.Speech). None — если не вышло."""
    if sys.platform != "win32":
        return None
    tmp = Path(tempfile.gettempdir()) / "protocol_asr_gpu_test.wav"
    speak = "\n".join('$syn.Speak("%s"); $syn.Speak(" ")' % line for line in TTS_LINES)
    script = """Add-Type -AssemblyName System.Speech
$syn = New-Object System.Speech.Synthesis.SpeechSynthesizer
$ru = $syn.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq "ru-RU" }
if (-not $ru) { exit 2 }
$syn.SelectVoice($ru[0].VoiceInfo.Name)
$bits = [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen
$mono = [System.Speech.AudioFormat.AudioChannel]::Mono
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(%d, $bits, $mono)
$syn.SetOutputToWaveFile("%s", $fmt)
%s
$syn.SetOutputToNull(); $syn.Dispose()
""" % (config.SAMPLE_RATE, tmp, speak)
    ps = Path(tempfile.gettempdir()) / "protocol_asr_gpu_test.ps1"
    # utf-8-sig обязателен: Windows PowerShell 5.1 читает .ps1 без BOM как ANSI и ломает кириллицу.
    ps.write_text(script, encoding="utf-8-sig")
    try:
        r = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps)],
                           capture_output=True, timeout=180)
    except Exception:
        return None
    if r.returncode != 0 or not tmp.exists():
        return None
    return _read_wav(str(tmp))


def get_audio() -> tuple[np.ndarray, str]:
    if len(sys.argv) > 1:
        return _read_wav(sys.argv[1]), f"файл {sys.argv[1]}"
    speech = _tts_wav()
    if speech is not None and speech.size > 10 * config.SAMPLE_RATE:
        return speech, "системный русский TTS"
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(30 * config.SAMPLE_RATE) * 0.05).astype(np.float32)
    return noise, "ШУМ (речи нет — числа ниже завышены в разы, см. докстринг модуля)"


def report_device() -> None:
    print("--- устройство ---")
    print(f"WHISPER_DEVICE={config.WHISPER_DEVICE_REQUESTED!r} -> {config.WHISPER_DEVICE}")
    print(device.describe())
    dlls = device.ensure_cuda_dlls()
    print(f"зарегистрировано каталогов CUDA-DLL: {len(dlls)}")
    for d in dlls:
        print(f"   {d}")
    print("\n--- дефолты под это устройство ---")
    for name in ("WHISPER_MODEL", "WHISPER_COMPUTE", "WHISPER_NUM_WORKERS", "ASR_WORKER_THREADS",
                 "ASR_UPDATE_MS", "WHISPER_BEAM_SIZE_PARTIAL", "ASR_BATCH_ENABLED",
                 "ASR_BATCH_MIN_S", "ASR_OVERLOAD_LAG_MS", "ASR_MAX_CACHED_MODELS"):
        print(f"   {name} = {getattr(config, name)}")


def test_decode_speed(asr, audio: np.ndarray) -> None:
    """Один проход по окну partial и по длинному куску (если хватает материала)."""
    print(f"\n--- скорость декодирования ({config.WHISPER_MODEL} / {config.WHISPER_DEVICE}) ---")
    total_s = audio.size / config.SAMPLE_RATE
    lengths = [config.ASR_WINDOW_MS / 1000.0]
    if total_s >= config.ASR_BATCH_MIN_S + 5:
        lengths.append(min(total_s, 60.0))

    for seconds in lengths:
        chunk = audio[:int(seconds * config.SAMPLE_RATE)]
        t0 = time.perf_counter()
        res = asr.transcribe(chunk, config.SAMPLE_RATE, beam_size=config.WHISPER_BEAM_SIZE)
        dt = time.perf_counter() - t0
        batched = (config.ASR_BATCH_ENABLED and config.IS_GPU
                   and seconds >= config.ASR_BATCH_MIN_S)
        print(f"   {seconds:>5.1f}с аудио -> {dt:5.2f}с ({seconds / dt:5.1f}x реального времени)"
              f"  слов={len(res.text.split()):<3d}{'  [батчевый проход]' if batched else ''}")


def test_two_channels_realtime(asr, audio: np.ndarray) -> None:
    """Главный критерий пригодности для live: два канала, декодируемые ОДНОВРЕМЕННО, должны
    успевать за реальным временем. Именно здесь ломались тяжёлые модели на CPU — суммарная
    нагрузка вдвое выше, чем в однопоточном замере, и «0.9x на один канал» превращается в
    растущую очередь."""
    print("\n--- два канала одновременно (критерий live-пригодности) ---")
    window_s = config.ASR_WINDOW_MS / 1000.0
    chunk = audio[:int(window_s * config.SAMPLE_RATE)]
    took: list[float] = []

    def one() -> None:
        t0 = time.perf_counter()
        asr.transcribe(chunk, config.SAMPLE_RATE, beam_size=config.WHISPER_BEAM_SIZE_PARTIAL)
        took.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=one) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    budget = config.ASR_UPDATE_MS / 1000.0
    print(f"   окно {window_s:.0f}с x2 канала: wall={wall:.2f}с (каждый {max(took):.2f}с)")
    print(f"   бюджет одного раунда partial (ASR_UPDATE_MS) = {budget:.2f}с")
    if wall <= budget:
        print("   OK ✅ — партиалы успевают в заданный каданс")
    elif wall <= window_s:
        print("   OK ⚠️  — каданс растянется (адаптивный lag_ema в session.py), но очередь "
              "не растёт: декодирование быстрее реального времени")
    else:
        print("   ПЛОХО ❌ — медленнее реального времени, задержка будет расти без ограничения. "
              "Возьмите модель полегче или включите GPU.")


if __name__ == "__main__":
    if config.ASR_PROVIDER == "stub":
        print("ASR_PROVIDER=stub — проверять нечего, снимите переменную.")
        sys.exit(1)
    report_device()
    if not config.IS_GPU:
        print("\nВНИМАНИЕ: работаем на CPU. Если ожидался GPU — см. backend/README.md, "
              "«Запуск на GPU» / «Если GPU не поднялся».")

    audio, source = get_audio()
    print(f"\nматериал: {audio.size / config.SAMPLE_RATE:.1f}с, источник — {source}")

    from .asr import get_asr
    t0 = time.perf_counter()
    asr = get_asr()
    asr.warmup()
    print(f"загрузка + прогрев модели: {time.perf_counter() - t0:.1f}с")

    test_decode_speed(asr, audio)
    test_two_channels_realtime(asr, audio)
    print("\nGPU-ПРОВЕРКА ЗАВЕРШЕНА")
