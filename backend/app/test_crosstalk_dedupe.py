"""Проверка пост-ASR дедупликации протёкшего голоса (Блок 3.7, `Session.
_maybe_mark_crosstalk_duplicate`) при 3+ ВЗАИМНО пересекающихся по времени репликах на разных
каналах.

Реальный баг: цикл сравнения `new_seg` с уже существующими репликами не останавливался на первом
совпадении — если новая реплика пересекалась сразу с НЕСКОЛЬКИМИ ещё не помеченными кандидатами,
`bleed_pair_id` переписывался вторым совпадением, и первый партнёр оставался с ОДНОСТОРОННЕЙ
ссылкой (его `bleed_pair_id` указывает на `new_seg`, а `new_seg.bleed_pair_id` уже указывает на
кого-то другого). После удаления одного дубля через `delete_segment` такой «осиротевший» партнёр
не находился через обратную пару и оставался висеть помеченным дублем навсегда.

Запуск: `pytest app/test_crosstalk_dedupe.py -v`
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import tempfile
from pathlib import Path

from . import config
from .models import Segment
from .session import manager


def _make_session(session_id: str):
    s = manager.create(session_id)
    return s


def test_new_segment_pairs_with_only_first_overlapping_candidate() -> None:
    """A и B — один и тот же канал (никогда не сравниваются друг с другом), оба пересекаются по
    времени и почти дословно совпадают с C на ДРУГОМ канале — классический случай протечки на
    общий/близкий микрофон сразу от двух реплик. До фикса C переписывал бы bleed_pair_id при
    втором совпадении (с A на B), оставляя A висеть с обратной ссылкой в никуда."""
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = config.STORAGE_DIR
        config.STORAGE_DIR = Path(tmp)
        try:
            s = _make_session("crosstalk-test-1")
            a = Segment(channel=0, speaker="Интервьюер", start=0.0, end=1.5,
                        text="это тестовая фраза", own_snr_db=20.0)
            b = Segment(channel=0, speaker="Интервьюер", start=0.2, end=1.7,
                        text="это тестовая фраза", own_snr_db=25.0)
            c = Segment(channel=1, speaker="Опрашиваемый", start=0.1, end=1.6,
                        text="это тестовая фраза", own_snr_db=10.0)
            s.protocol.segments.extend([a, b, c])

            s._maybe_mark_crosstalk_duplicate(c)

            print(f"A: likely_bleed={a.likely_bleed} pair={a.bleed_pair_id}")
            print(f"B: likely_bleed={b.likely_bleed} pair={b.bleed_pair_id}")
            print(f"C: likely_bleed={c.likely_bleed} pair={c.bleed_pair_id}")

            # C нашёл себе пару и остановился — B (второй кандидат) не тронут вовсе.
            assert b.likely_bleed is False
            assert b.bleed_pair_id is None
            # A и C — взаимно согласованная пара, без дангл-ссылок.
            assert a.likely_bleed is True and c.likely_bleed is True
            assert a.bleed_pair_id == c.id
            assert c.bleed_pair_id == a.id
        finally:
            config.STORAGE_DIR = original_dir


def test_deleting_one_pair_does_not_disturb_a_different_pair() -> None:
    """Расширенный сценарий: ДВЕ независимые пары дублей (A-C и D-E) сосуществуют одновременно.
    Удаление одной пары (через `delete_segment`, единственный публичный способ убрать дубль)
    обязано снять пометку ровно со своего партнёра (A) и не задеть вторую, никак не связанную
    пару (D-E) — до фикса именно такая путаница пар и была реальным риском."""
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = config.STORAGE_DIR
        config.STORAGE_DIR = Path(tmp)
        try:
            s = _make_session("crosstalk-test-2")
            a = Segment(channel=0, speaker="Интервьюер", start=0.0, end=1.5,
                        text="это тестовая фраза", own_snr_db=20.0)
            b = Segment(channel=0, speaker="Интервьюер", start=0.2, end=1.7,
                        text="это тестовая фраза", own_snr_db=25.0)
            c = Segment(channel=1, speaker="Опрашиваемый", start=0.1, end=1.6,
                        text="это тестовая фраза", own_snr_db=10.0)
            s.protocol.segments.extend([a, b, c])
            s._maybe_mark_crosstalk_duplicate(c)

            d = Segment(channel=2, speaker="Голос-3", start=5.0, end=6.5,
                        text="другая одинаковая фраза", own_snr_db=5.0)
            e = Segment(channel=3, speaker="Голос-4", start=5.1, end=6.6,
                        text="другая одинаковая фраза", own_snr_db=30.0)
            s.protocol.segments.extend([d, e])
            s._maybe_mark_crosstalk_duplicate(e)

            assert d.bleed_pair_id == e.id and e.bleed_pair_id == d.id, \
                "вторая пара должна образоваться независимо от первой"

            result = s.delete_segment(c.id)
            print(f"после удаления C: {result}")

            assert result is not None
            assert all(seg.id != c.id for seg in s.protocol.segments), "C должен быть удалён"
            # Партнёр удалённого C (A) корректно расформирован.
            assert a.likely_bleed is False
            assert a.bleed_pair_id is None
            # Вторая пара (D-E) не должна была пострадать от удаления совершенно другой пары.
            assert d.likely_bleed is True and e.likely_bleed is True
            assert d.bleed_pair_id == e.id
            assert e.bleed_pair_id == d.id
        finally:
            config.STORAGE_DIR = original_dir


def main() -> None:
    test_new_segment_pairs_with_only_first_overlapping_candidate()
    test_deleting_one_pair_does_not_disturb_a_different_pair()
    print("\nTEST_CROSSTALK_DEDUPE OK ✅")


if __name__ == "__main__":
    main()
