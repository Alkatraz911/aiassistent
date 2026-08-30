"""Нарезка длинной записи на блоки — по тишине, а не по линейке.

Зачем вообще резать, если faster-whisper сам умеет длинное аудио: блок — это единица
1) памяти (см. decode.py: иначе мел-спектрограмма всей записи живёт в RAM целиком),
2) прогресса (пользователь видит, где идёт трёхчасовая расшифровка),
3) возобновления (чекпойнт пишется после каждого блока — обрыв стоит одного блока, а не всего).

Резать строго по таймеру нельзя: рез посреди слова портит ДВА блока — конец одного и начало
другого, причём типично именно тем, что модель дофантазирует оборванное слово. Поэтому точный
момент реза ищется в окне вокруг цели: берём самые тихие полсекунды. Пауза в разговоре длиннее
полусекунды находится практически всегда — в живой речи их десятки в минуту.

Тишина ищется по энергии, а не через Silero VAD: VAD нужен, чтобы отличить речь от шума, а нам
нужно лишь «где тише всего» — для этого хватает RMS, и он не требует прогонять модель по всей
записи ради выбора десятка точек реза.
"""
from __future__ import annotations

import numpy as np

# Шаг сетки, на которой считается энергия. 100 мс — заметно короче самой короткой межсловной
# паузы, дальше дробить бессмысленно.
_FRAME_MS = 100
# Сколько соседних кадров усредняем при поиске минимума. Одиночный тихий кадр бывает и ВНУТРИ
# слова (смычка перед взрывным согласным, «п», «т»), а полсекунды подряд — уже пауза.
_SMOOTH_FRAMES = 5


def plan_blocks(
    pcm: np.ndarray,
    sample_rate: int,
    *,
    target_s: float,
    search_s: float | None = None,
    min_tail_s: float | None = None,
) -> list[tuple[int, int]]:
    """Границы блоков (в сэмплах, полуинтервалы) для всей записи `pcm`.

    `target_s`   — желаемая длина блока;
    `search_s`   — насколько далеко от цели разрешено искать тишину (в обе стороны);
    `min_tail_s` — если до конца записи осталось меньше, не режем: короткий хвостовой блок
                   ничего не ускоряет, а на не-речи в конце записи (щелчок остановки диктофона)
                   короткий кусок — самое подходящее место для галлюцинации.

    Оба допуска по умолчанию считаются ОТ длины блока, а не берутся константами: при блоках по
    10 минут это привычные 45с и 90с, но с коротким блоком (тесты, `--block-min`) константы
    просто отменяли бы нарезку — окно поиска оказалось бы шире самого блока.
    """
    if search_s is None:
        search_s = min(45.0, target_s * 0.15)
    if min_tail_s is None:
        min_tail_s = min(90.0, target_s * 0.25)
    total_samples = len(pcm)
    target = int(target_s * sample_rate)
    radius = int(search_s * sample_rate)
    min_tail = int(min_tail_s * sample_rate)
    if target <= 0 or total_samples <= target + min_tail:
        return [(0, total_samples)]

    blocks: list[tuple[int, int]] = []
    pos = 0
    while total_samples - pos > target + min_tail:
        cut = _quiet_cut(pcm, pos + target, radius, sample_rate)
        # Страховка от вырождения: рез обязан двигать нас вперёд.
        cut = max(cut, pos + target // 2)
        cut = min(cut, total_samples)
        blocks.append((pos, cut))
        pos = cut
    blocks.append((pos, total_samples))
    return blocks


def _quiet_cut(pcm: np.ndarray, center: int, radius: int, sample_rate: int) -> int:
    """Сэмпл в окне `center ± radius`, вокруг которого звук тише всего."""
    lo = max(0, center - radius)
    hi = min(len(pcm), center + radius)
    frame = int(_FRAME_MS * sample_rate / 1000)
    n_frames = (hi - lo) // frame
    if n_frames < _SMOOTH_FRAMES * 2:
        return center

    window = np.abs(np.asarray(pcm[lo:lo + n_frames * frame], dtype=np.float32))
    energy = window.reshape(n_frames, frame).mean(axis=1)
    smooth = np.convolve(energy, np.ones(_SMOOTH_FRAMES) / _SMOOTH_FRAMES, mode="valid")
    best = int(np.argmin(smooth))
    return lo + (best + _SMOOTH_FRAMES // 2) * frame + frame // 2
