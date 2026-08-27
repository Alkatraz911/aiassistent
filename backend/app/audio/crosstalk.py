"""Мягкая оценка cross-talk между каналами (Блок 3.7 плана) — не жёсткий gate.

Раньше `CrossTalkGate.is_owner()` — бинарное решение по одному кадру, сравнивая RMS текущего
канала с самым громким ДРУГИМ каналом за последнее окно. Реальный тест на живой записи показал:
при близко расположенных микрофонах разница между каналами часто «мигает» в пределах пары дБ
кадр от кадра (естественная динамика речи), и жёсткий per-frame gate из-за этого не даёт
устойчивого разделения — оба канала то и дело проходят порог и распознаются независимо.

Здесь `CrossTalkScorer.score()` возвращает непрерывную оценку `bleed_score` в [0, 1] и
own_snr_db (для последующей пост-ASR дедупликации, см. `Session._maybe_mark_crosstalk_duplicate`
в `session.py`), а не бинарное решение:
  - подавление (для приоритета в ASR, НЕ для физического удаления — raw всегда пишется,
    Блок 0.4) активируется только после `CROSSTALK_TAKEOVER_MS` устойчивого превышения другим
    каналом, а не с первого кадра — сглаживает дребезг и не режет короткие «да»/«угу»;
  - требуемая маржа растёт с собственным SNR кадра (SNR-aware margin): кадр с высоким self-SNR
    (типично для настоящей речи в свой микрофон) требует бОльшего разрыва от другого канала,
    чтобы быть признанным протечкой, чем кадр, едва превышающий свой собственный пол шума
    (типично для акустической утечки).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .. import config
from .noise import NoiseFloorTracker


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


class CrossTalkScorer:
    def __init__(self) -> None:
        self._floors: dict[int, NoiseFloorTracker] = {}
        self._recent: dict[int, tuple[float, float]] = {}   # channel -> (rms, monotonic_ts)
        self._loud_since: dict[int, float] = {}              # channel -> ts начала устойчивого превышения
        self._margin_lin = 10.0 ** (config.CROSSTALK_MARGIN_DB / 20.0)

    def reset_transient_state(self) -> None:
        """Сбрасывает покадровое состояние (`_recent`/`_loud_since`) между заходами записи —
        `Session` переиспользуется между кликами «начать/остановить» (см. `Session.attach_ws`),
        и без сброса `_loud_since` реальный баг: если последний обработанный кадр перед «стоп»
        случайно попал в состояние «другой канал громче», метка времени остаётся в словаре
        НАВСЕГДА (следующий `pop` для этого канала никогда не наступает, т.к. кадры перестали
        приходить). В новом заходе первый же такой кадр считает разницу «устойчивой уже много
        минут» и канал мгновенно и необратимо давится как протечка. Адаптивный пол шума
        (`_floors`) сознательно НЕ сбрасываем — характеристики шума помещения не меняются
        за паузу между кликами, теряя которые пришлось бы заново «переучиваться»."""
        self._recent.clear()
        self._loud_since.clear()

    def _floor_for(self, channel: int) -> NoiseFloorTracker:
        f = self._floors.get(channel)
        if f is None:
            f = NoiseFloorTracker(config.GATE_NOISE_FLOOR)
            self._floors[channel] = f
        return f

    def score(self, channel: int, frame: np.ndarray) -> tuple[float, float]:
        """Возвращает `(bleed_score, own_snr_db)`. `bleed_score` — 0 (точно свой голос)..1
        (почти наверняка протечка). `own_snr_db` — на сколько дБ кадр громче своего собственного
        (адаптивного) пола шума, для пост-ASR сравнения каналов."""
        r = rms(frame)
        floor = self._floor_for(channel)
        own_snr = floor.snr_db(r)
        floor.update(r)

        now = time.monotonic()
        self._recent[channel] = (r, now)

        if not config.CROSSTALK_ENABLED or r < config.GATE_NOISE_FLOOR:
            self._loud_since.pop(channel, None)
            return 0.0, own_snr

        loudest_other = 0.0
        window = config.CROSSTALK_WINDOW_MS / 1000.0
        for ch, (orms, ots) in self._recent.items():
            if ch == channel:
                continue
            if now - ots <= window:
                loudest_other = max(loudest_other, orms)

        currently_louder = loudest_other > r * self._margin_lin
        if not currently_louder:
            self._loud_since.pop(channel, None)
            return 0.0, own_snr

        self._loud_since.setdefault(channel, now)
        sustained_ms = (now - self._loud_since[channel]) * 1000.0
        if sustained_ms < config.CROSSTALK_TAKEOVER_MS:
            return 0.0, own_snr   # ещё не устойчиво — не считаем протечкой (сглаживание дребезга)

        # SNR-aware margin: чем увереннее кадр выглядит настоящей близкой речью (высокий
        # собственный SNR), тем больший разрыв от другого канала нужен, чтобы признать протечкой.
        ratio_db = 20.0 * math.log10((loudest_other + 1e-9) / (r + 1e-9))
        required_db = config.CROSSTALK_MARGIN_DB + max(0.0, own_snr - 10.0) * 0.5
        if ratio_db < required_db:
            return 0.0, own_snr

        bleed_score = min(1.0, 0.5 + (ratio_db - required_db) / 10.0)
        return bleed_score, own_snr
