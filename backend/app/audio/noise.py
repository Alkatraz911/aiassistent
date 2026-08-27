"""Адаптивный порог шума (Блок 3.6 плана) — без циклической зависимости.

Раньше классификация речь/тишина сравнивалась с фиксированной константой
(`VAD_ENERGY_THRESHOLD`): если реальный шум помещения был выше неё, шумные кадры навсегда
классифицировались как речь. Хуже того, в изначальной формулировке трекер обновлялся бы только
на кадрах, уже признанных тишиной ЭТИМ ЖЕ порогом — замкнутая зависимость, которая никогда не
даёт «переучиться» на новый уровень шума.

Здесь порог — leaky minimum-follower: обновляется на КАЖДОМ кадре независимо от классификации,
асимметрично (медленно растёт, быстро падает), поэтому кратковременная речь не может задрать
порог, а устойчиво повысившийся шум помещения через несколько секунд всё равно «перетягивает» его.
"""
from __future__ import annotations

import math


class NoiseFloorTracker:
    def __init__(self, floor_min: float, rise_alpha: float = 0.02, fall_alpha: float = 0.15) -> None:
        self._floor_min = floor_min
        self._floor = floor_min
        self._rise_alpha = rise_alpha   # медленный подъём порога (речь не должна его поднимать)
        self._fall_alpha = fall_alpha   # быстрое опускание (комната стала тише — быстро замечаем)

    def update(self, rms: float) -> float:
        """Обновляет пол шума текущим кадром (независимо от того, речь это или нет —
        асимметрия alpha сама по себе не даёт речи «испортить» оценку) и возвращает его."""
        alpha = self._fall_alpha if rms < self._floor else self._rise_alpha
        self._floor += (rms - self._floor) * alpha
        self._floor = max(self._floor, self._floor_min)
        return self._floor

    @property
    def floor(self) -> float:
        return self._floor

    def snr_db(self, rms: float) -> float:
        return 20.0 * math.log10((rms + 1e-9) / (self._floor + 1e-9))

    def classify(self, rms: float, margin_db: float = 6.0) -> bool:
        """True — похоже на речь (заметно выше текущего пола шума). Обновляет порог этим же
        кадром ПОСЛЕ классификации, так что оценка всегда основана на предыдущем состоянии, но
        порог не «застревает»."""
        is_speech = self.snr_db(rms) >= margin_db
        self.update(rms)
        return is_speech
