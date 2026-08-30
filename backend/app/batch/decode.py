"""Декодирование записи в сырой PCM НА ДИСКЕ (16 кГц, моно, int16).

Почему не `faster_whisper.audio.decode_audio`, который умеет ровно это:
он держит весь сигнал в памяти и отдаёт float32. Для основного материала этого инструмента —
записи на час-два-три — счёт получается такой (на час звука):

    float32 в памяти (decode_audio)      230 МБ
    + копия, которую делает vad_filter   230 МБ
    + мел-спектрограмма всего файла      ~550 МБ
    итого на трёхчасовой записи          ~3 ГБ

При этом никакая часть пайплайна не требует видеть всю запись сразу. Поэтому здесь звук
декодируется ПОТОКОМ на диск, а дальше обрабатывается блоками через `np.memmap` (см. blocks.py):
расход памяти перестаёт зависеть от длительности записи вообще.

int16, а не float32, потому что ровно это отдаёт ресемплер ffmpeg (`format="s16"`), а деление
на 32768 всё равно делается поблочно на входе в модель. Хранить вдвое больше байт незачем:
115 МБ на час вместо 230.

Побочная выгода — возобновление: декодированный PCM переживает Ctrl+C и падение, и повторный
запуск не платит за декодирование заново (см. `pipeline.py`, чекпойнт).

Формат входа значения не имеет: PyAV — это те же библиотеки ffmpeg, m4a/aac, mp3, opus, wav,
дорожка из mp4 читаются одинаково, ставить ffmpeg отдельно не нужно.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .. import config

# Сколько сэмплов накапливать перед ресемплированием. Кадр AAC — 1024 сэмпла, то есть на часовой
# записи их ~170 тысяч; вызывать ресемплер на каждый смысла нет. Значение — как у faster-whisper.
_GROUP_SAMPLES = 500_000


def decode_to_pcm(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = config.SAMPLE_RATE,
    on_progress: Callable[[float], None] | None = None,
) -> int:
    """Декодирует `src` в сырой int16 LE моно @ `sample_rate` и пишет в `dst`.

    Возвращает число записанных сэмплов. Пишет через `.part`-файл и атомарный `os.replace`:
    оборванное на середине декодирование не должно выглядеть как готовый кэш при следующем
    запуске (иначе расшифровка молча оборвалась бы на том же месте).

    `on_progress` получает число уже декодированных секунд.
    """
    import av

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=sample_rate)
    tmp = dst.with_name(dst.name + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with av.open(str(src), mode="r", metadata_errors="ignore") as container:
        if not container.streams.audio:
            raise ValueError(f"в файле нет аудиодорожки: {src.name}")
        # Многопоточное декодирование: на длинной записи это единственное место, где просто так
        # лежит кратное ускорение (сам декодер AAC/mp3 однопоточен по кадру, но кадры независимы).
        container.streams.audio[0].thread_type = "AUTO"
        with open(tmp, "wb") as out:
            for chunk in _iter_pcm(container, resampler):
                out.write(chunk.tobytes())
                written += chunk.size
                if on_progress is not None:
                    on_progress(written / sample_rate)

    os.replace(tmp, dst)
    return written


def open_pcm(path: Path) -> np.memmap:
    """Открывает результат `decode_to_pcm` как одномерный int16-memmap (read-only)."""
    return np.memmap(path, dtype="<i2", mode="r")


def close_pcm(pcm: np.memmap) -> None:
    """Закрывает memmap. Обязательно перед удалением файла: Windows не даст удалить то, что
    отображено в память, а кэш декодированного звука удалять надо — это гигабайты."""
    import gc

    try:
        pcm._mmap.close()           # noqa: SLF001 — публичного способа закрыть memmap нет
    except Exception:
        pass
    gc.collect()


def to_float32(pcm: np.ndarray) -> np.ndarray:
    """int16-кусок -> float32 в [-1, 1] (то, что ждёт Whisper). Всегда делает копию —
    срез memmap иначе остался бы привязан к файлу на весь срок жизни блока."""
    return np.asarray(pcm, dtype=np.float32) / 32768.0


def probe_duration(src: Path) -> float | None:
    """Длительность записи по контейнеру, без декодирования (None — если не указана)."""
    import av

    try:
        with av.open(str(src), mode="r", metadata_errors="ignore") as container:
            if container.duration:
                return float(container.duration) / av.time_base
            if container.streams.audio:
                stream = container.streams.audio[0]
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


def _iter_pcm(container, resampler) -> Iterator[np.ndarray]:
    """Кадры контейнера -> int16-массивы нужной частоты, группами."""
    import av

    fifo = av.audio.fifo.AudioFifo()
    for frame in _ignore_invalid_frames(container.decode(audio=0)):
        # Сбрасываем pts: FIFO иначе ругается на разрывы временной шкалы, а они на записях с
        # телефона (пауза записи, смена битрейта) — обычное дело.
        frame.pts = None
        fifo.write(frame)
        if fifo.samples >= _GROUP_SAMPLES:
            yield from _resample(resampler, fifo.read())
    if fifo.samples > 0:
        yield from _resample(resampler, fifo.read())
    yield from _resample(resampler, None)      # дослать хвост из самого ресемплера


def _resample(resampler, frame) -> Iterator[np.ndarray]:
    for out in resampler.resample(frame):
        yield out.to_ndarray().reshape(-1)


def _ignore_invalid_frames(frames):
    """Пропускает битые кадры вместо падения.

    На длинных записях это не теория: у диктофонных m4a регулярно встречается повреждённый кадр
    в середине (обрыв записи, перенос с телефона). Ронять из-за него двухчасовую расшифровку —
    худший из возможных исходов; потеря — доли секунды звука.
    """
    import av

    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue
