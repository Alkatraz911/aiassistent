"""Smoke-тест пайплайна без сети и UI.

Прогоняет синтетическое аудио через сессию: VAD-чанкинг -> ASR -> сегменты,
проверяет маркировку спикеров, правку с аудитом, сценарий анкеты и сохранение.
Запуск:  ASR_PROVIDER=stub  py -3.11 -m app.smoke
"""
from __future__ import annotations

import sys
import wave

import numpy as np

# Консоль Windows может быть cp1251 — выводим протокол/кириллицу в UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .asr.base import ASRWord
from .asr.local_agreement import UtteranceHypothesis
from .session import ChannelStreamState, manager


def _speech(seconds: float, rate: int = 16000, amp: float = 0.1) -> np.ndarray:
    n = int(seconds * rate)
    tone = (amp * np.sin(2 * np.pi * 180 * np.arange(n) / rate))
    return (tone * 32767).astype("<i2")


def _silence(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype="<i2")


def test_crosstalk() -> None:
    """Громкий канал 0 говорит, в канал 1 голос «протекает» тихо -> канал 1 гасится (в ASR;
    фонограмма при этом сохраняется НЕ гейтнутой — Блок 0.4 плана, целостность записи)."""
    s = manager.create("smoke-xtalk")
    frame = 0.1  # сек на кадр
    loud = _speech(frame, amp=0.12)     # хозяин (опрашиваемый говорит в свой микрофон)
    bleed = _speech(frame, amp=0.012)   # протёкший голос в чужом микрофоне (-20 дБ)
    # Интерливим кадры по времени, как в реальном стриме.
    for _ in range(20):                 # ~2 c одновременной речи
        s.ingest_streaming(0, loud)
        s.ingest_streaming(1, bleed)
    s.force_finalize_channel(0)
    s.force_finalize_channel(1)
    assert manager.scheduler.wait_idle(timeout=30.0), "ASR-планировщик не успел обработать задания"
    ch0 = [seg for seg in s.protocol.segments if seg.channel == 0]
    ch1 = [seg for seg in s.protocol.segments if seg.channel == 1]
    print(f"crosstalk: канал0 сегментов={len(ch0)} (ожидаем >0), "
          f"канал1 сегментов={len(ch1)} (ожидаем 0 — подавлен в ASR)")
    assert ch0, "громкий канал должен дать сегмент"
    assert not ch1, "протёкший голос должен быть подавлен гейтом при распознавании"

    # Raw-звук должен остаться в фонограмме нетронутым, даже если гейт подавил его для ASR.
    s.close()
    with wave.open(str(s.channel_wav_path(1)), "rb") as r:
        raw_ch1 = np.frombuffer(r.readframes(r.getnframes()), dtype="<i2")
    assert np.max(np.abs(raw_ch1)) > 0, "фонограмма протёкшего канала не должна обнуляться"
    print("CROSSTALK OK ✅ (raw-звук сохранён, ASR подавлен)")


def test_crosstalk_backchannel() -> None:
    """Канал 0 — громкая непрерывная речь, канал 1 — умеренно тихий (−8 дБ, не −20 дБ) всплеск.
    SNR-aware маржа (Блок 3.7) не должна подавлять это как протечку — иначе реальные тихие
    реплики/бэкчаннелы («да»/«угу») поверх более громкого собеседника будут теряться (именно это
    показал реальный тест с близко расположенными микрофонами)."""
    s = manager.create("smoke-backchannel")
    frame = 0.1
    loud = _speech(frame, amp=0.12)
    moderate = _speech(frame, amp=0.12 * 10 ** (-8 / 20))   # −8 дБ относительно loud
    for _ in range(20):
        s.ingest_streaming(0, loud)
        s.ingest_streaming(1, moderate)
    s.force_finalize_channel(0)
    s.force_finalize_channel(1)
    assert manager.scheduler.wait_idle(timeout=30.0), "ASR-планировщик не успел обработать задания"
    ch1 = [seg for seg in s.protocol.segments if seg.channel == 1]
    print(f"backchannel: канал1 (−8 дБ) сегментов={len(ch1)} (ожидаем >0 — НЕ должен подавляться)")
    assert ch1, "умеренно тихий, но не протёкший голос не должен подавляться SNR-aware гейтом"
    print("CROSSTALK_BACKCHANNEL OK ✅")


def test_repetition_hallucination_filter() -> None:
    """Прямая проверка эвристики `has_repeating_ngram` (Блок 3, продолжение) на реальных
    примерах галлюцинаций, которые вылезли на практике — и на обычной речи, чтобы не резать
    ложно."""
    from .asr.base import has_repeating_ngram

    hallucinations = [
        "Ветка. " * 30,
        "Я не знаю, я не знаю, я не знаю, я не знаю, я не знаю.",
        "Ветка. Ветка. Ветка. Спасибо за просмотр!",   # смесь повтора и "хвоста" — должно ловиться
    ]
    normal = [
        "мне было очень весело",
        "Меня зовут Александр Сергеевич Ковалёв",
        "Там я встретился с Михаилом Петровичем, передал ему документы",
    ]
    for text in hallucinations:
        assert has_repeating_ngram(text), f"должно определяться как галлюцинация: {text!r}"
    for text in normal:
        assert not has_repeating_ngram(text), f"не должно ложно резаться: {text!r}"
    print("REPETITION_HALLUCINATION_FILTER OK ✅")


def test_prompt_echo_filter() -> None:
    """Самый частый вид галлюцинации на реальных записях — модель отдаёт обратно сам
    `WHISPER_PROMPT`, когда распознавать нечего (тишина, шум, обрывок). Порогами уверенности не
    ловится: эхо промпта модель выдаёт УВЕРЕННО. Все «галлюцинации» ниже — дословно из протокола
    живой 107-секундной записи; крупные модели (large-v3) эхают заметно чаще мелких."""
    from .asr.base import looks_like_known_hallucination, looks_like_prompt_echo
    from . import config

    prompt = config.WHISPER_PROMPT
    echoes = [
        "Интервьюер и опрашиваемый. Разговорная речь.",   # промпт целиком (без "Протокол опроса.")
        "Разговорная речь.",                              # хвост промпта
        "Интервьюер.",                                    # одно слово из промпта
        "опросы.",                          # слово промпта в другой форме («опроса»)
        "Интервьюер опрашиваемый. Разговорная речь.",   # эхо с выпавшим словом («и»)
    ]
    credits = [
        "Редактор субтитров И .Бойкова",
        "Субтитры сделал DimaTorzok",
        "Корректор А .Егорова",
        "Продолжение следует...",
    ]
    normal = [
        "Иванов Сергей Петрович, тысяча девятьсот восемьдесят четвёртого года рождения",
        "Моя машинка разноцветная.",
        "Разговаривай в микрофон. Не надо так близко. Просто держи и говори.",
        "Миша, ты что делаешь?",
        # Реплика про сам протокол — из слов промпта, но НЕ в его порядке и с чужими словами.
        "Я прочитал протокол и всё подтверждаю",
        # «Корректор» — не только титры, но и обычная профессия из биографического блока.
        # Голым вхождением маркера такая реплика отбраковывалась: гейта в 8 слов ей мало.
        "Он работал корректором в типографии",
        # Реальный воспроизведённый случай: ответ на анкетный вопрос «На каком языке желаете
        # давать показания» — раньше промпт содержал слово «русская», и однословный ответ ловился
        # по 5-буквенной основе («русск») как эхо промпта, оставляя поле анкеты пустым при КАЖДОЙ
        # попытке (не эпизодически). Проверяем впредь, а не только чиним задним числом.
        "русский",
        "русском.",
        "на русском",
    ]
    for text in echoes:
        assert looks_like_prompt_echo(text, prompt), f"должно ловиться как эхо промпта: {text!r}"
    for text in credits:
        assert looks_like_known_hallucination(text), f"должно ловиться как титры: {text!r}"
    for text in normal:
        assert not looks_like_prompt_echo(text, prompt), f"не должно ложно резаться: {text!r}"
        assert not looks_like_known_hallucination(text), f"не должно ложно резаться: {text!r}"
    # Почему сравнивать надо со СТАТИЧЕСКИМ промптом, а не с тем, что реально уходит в модель:
    # в живой сессии initial_prompt = WHISPER_PROMPT + последние распознанные слова канала
    # (Session._build_prompt). Опрашиваемый постоянно повторяет формулировку вопроса — по
    # динамическому промпту такая настоящая реплика выглядела бы эхом и была бы выброшена.
    dynamic = prompt + " Вы были там пятнадцатого марта вечером"
    repeat_of_question = "там пятнадцатого марта вечером"
    assert looks_like_prompt_echo(repeat_of_question, dynamic), (
        "сценарий: по динамическому промпту повтор вопроса действительно выглядит эхом — "
        "именно поэтому провайдер сравнивает со статическим")
    assert not looks_like_prompt_echo(repeat_of_question, prompt), (
        "по статическому промпту повтор вопроса эхом быть НЕ должен — иначе режем живую речь")
    print("PROMPT_ECHO_FILTER OK ✅")


def test_prompt_has_no_recent_speech() -> None:
    """В `initial_prompt` не должно попадать недавно распознанное.

    Раньше туда безусловно дописывались последние 60 слов канала — «дадим модели контекст». На
    живой записи это давало обратный эффект: на фоновом шуме модель возвращала этот контекст
    ДОСЛОВНО, и в протокол шли повторы предыдущих вопросов с новыми тайм-кодами («Миша, сколько
    тебе лет?» четыре раза подряд). Замер на тихих окнах: с контекстом в промпте модель выдавала
    прошлые реплики, с одним доменным промптом — пустоту. Фильтром по тексту это неотличимо от
    настоящего повтора (в допросе вопрос повторяют постоянно), поэтому проверяем сам промпт."""
    from . import config

    s = manager.create("smoke-prompt")
    s._remember_prompt(0, "Миша, сколько тебе лет?")
    s._remember_prompt(0, "Я вам обещала.")
    prompt = s._build_prompt(0)
    assert prompt == (config.WHISPER_PROMPT or "").strip(), (
        f"промпт должен содержать только доменную подсказку, а содержит: {prompt!r}")
    for phrase in ("Миша", "обещала"):
        assert phrase not in prompt, f"недавняя речь просочилась в промпт: {phrase!r}"
    print("PROMPT_NO_RECENT_SPEECH OK ✅")


def test_recording_started_at_set_once() -> None:
    """Блок 6: `protocol.recording_started_at` — якорь для «время начала/окончания опроса»
    (docgen.AUTO_FIELDS). Реальный баг: если ставить его на КАЖДЫЙ `attach_ws` («стоп» ->
    «начать снова»), несколько реплик, ждавших очереди ASR и финализированных подряд сразу
    после «стоп», получали бы одинаковое время — start и end совпадали. Якорь должен
    фиксироваться один раз, на первый заход."""
    s = manager.create("smoke-recording-anchor")
    assert s.protocol.recording_started_at is None, "до первого attach_ws якоря быть не должно"
    s.attach_ws(None, None)
    first = s.protocol.recording_started_at
    assert first is not None
    s.detach_ws()
    s.attach_ws(None, None)   # второй заход — «стоп» -> «начать снова»
    assert s.protocol.recording_started_at == first, "якорь не должен сбрасываться между заходами"
    print("RECORDING_STARTED_AT_ONCE OK ✅")


def test_get_or_load_restores_session_after_restart() -> None:
    """Блок 7: «открыть прошлый допрос» должно работать и после перезапуска backend, когда
    сессии уже нет в памяти (типичный случай в ежедневной работе — не после каждой записи сразу
    смотрят готовый протокол). Симулируем перезапуск: сохраняем сессию на диск, убираем её из
    `manager.sessions` (то, что при реальном перезапуске стало бы пустым), просим `get_or_load` —
    должен восстановить и сам протокол, и состояние анкеты, СОГЛАСОВАННОЕ с сохранёнными
    ответами (иначе повторное «Сохранить» переписало бы questionnaire пустым)."""
    from .assistant.questionnaire import build_script
    from .models import Template, TemplateStep

    steps = [TemplateStep(key="fio", label="ФИО", kind="field", question="Ваше ФИО?")]

    s = manager.create("smoke-get-or-load")
    s.assistant.load_script(build_script(steps))
    s.protocol.template_id = "t"
    s.protocol.template_name = "Тест"
    s.protocol.template_snapshot = Template(id="t", name="Тест", steps=steps)
    s.assistant.start()
    s.assistant.submit_answer("Иванов Иван Иванович")
    s.save()   # пишет protocol.json — то самое состояние, что должно пережить "перезапуск"

    del manager.sessions["smoke-get-or-load"]   # симуляция: backend перезапущен, память пуста
    assert manager.get("smoke-get-or-load") is None, "get() не должен читать с диска"

    restored = manager.get_or_load("smoke-get-or-load")
    assert restored is not None, "get_or_load должен восстановить сессию из protocol.json"
    assert restored.protocol.questionnaire[0].value == "Иванов Иван Иванович"

    # Повторное «Сохранить» на восстановленной сессии не должно стереть questionnaire пустым.
    restored.save()
    reloaded = manager.get("smoke-get-or-load").protocol
    assert reloaded.questionnaire and reloaded.questionnaire[0].value == "Иванов Иван Иванович", (
        "повторный save() на восстановленной сессии не должен затирать ответы анкеты")
    print("GET_OR_LOAD_RESTORES_SESSION OK ✅")


def test_crosstalk_dedup_prefers_asr_confidence_over_snr() -> None:
    """Блок 3.7 (пересмотр дважды): при близких микрофонах система не может надёжно решить, кто
    реальный автор — помечаются ОБЕ копии дубля одинаково (`likely_bleed`), а не одна. Уверенность
    ASR (word.prob) и SNR не выбрасываются, а становятся подсказкой `bleed_hint` — ориентиром для
    оператора, не решением. При заметном разрыве уверенности подсказку определяет она: протёкший/
    искажённый звук ASR обычно распознаёт менее уверенно, даже когда SNR вводит в заблуждение."""
    from .models import Segment, Word

    s = manager.create("smoke-dedup-confidence")
    confident = [Word(text="слово", start=0.0, end=0.4, prob=0.95)] * 5
    unsure = [Word(text="слово", start=0.0, end=0.4, prob=0.4)] * 5   # искажённый протёкший звук

    seg_a = Segment(channel=0, speaker="Интервьюер", start=0.0, end=2.0,
                     text="Где вы были вчера вечером?", own_snr_db=5.0, words=confident)
    # Ниже SNR, чем у seg_a, — по старому (SNR-only) критерию подсказка указала бы на неё же,
    # хотя распознана она увереннее (похоже на реальный источник), а seg_a — тише, но неувереннее.
    seg_b = Segment(channel=1, speaker="Опрашиваемый", start=0.1, end=2.1,
                     text="Где вы были вчера вечером?", own_snr_db=20.0, words=unsure)
    s.protocol.segments = [seg_a]
    s._maybe_mark_crosstalk_duplicate(seg_b)
    assert seg_a.likely_bleed and seg_b.likely_bleed, "должны быть помечены ОБЕ копии"
    assert seg_b.bleed_hint == "likely_leak" and seg_a.bleed_hint == "likely_original", (
        "уверенность ASR должна была перевесить SNR в подсказке при заметном разрыве")
    print("CROSSTALK_DEDUP_PREFERS_CONFIDENCE OK ✅")


def test_crosstalk_dedup_falls_back_to_snr_when_confidence_is_close() -> None:
    """Когда уверенность ASR у обеих копий практически одинаковая (разрыв меньше
    CROSSTALK_DEDUPE_CONFIDENCE_GAP), подсказка (не решение — обе копии всё равно помечены и
    удаляемы) остаётся за SNR, как и было исходно."""
    from .models import Segment, Word

    s = manager.create("smoke-dedup-snr-fallback")
    words = [Word(text="слово", start=0.0, end=0.4, prob=0.9)] * 5   # одинаковая уверенность
    seg_a = Segment(channel=0, speaker="Интервьюер", start=0.0, end=2.0,
                     text="Где вы были вчера вечером?", own_snr_db=20.0, words=words)
    seg_b = Segment(channel=1, speaker="Опрашиваемый", start=0.1, end=2.1,
                     text="Где вы были вчера вечером?", own_snr_db=5.0, words=words)
    s.protocol.segments = [seg_a]
    s._maybe_mark_crosstalk_duplicate(seg_b)
    assert seg_a.likely_bleed and seg_b.likely_bleed, "должны быть помечены ОБЕ копии"
    assert seg_b.bleed_hint == "likely_leak" and seg_a.bleed_hint == "likely_original", (
        "при равной уверенности подсказка должна указывать на копию с более низким SNR")
    print("CROSSTALK_DEDUP_SNR_FALLBACK OK ✅")


def test_delete_segment_only_allowed_for_bleed_marked() -> None:
    """Удаление реплики (Блок 3.7: кнопка «✕ удалить» у дублей) — не общий способ стирать
    транскрипт: разрешено только для реплик, уже помеченных вероятным дублем, иначе кнопка
    превратилась бы в способ незаметно вычистить настоящие показания без следа."""
    from .models import Segment

    s = manager.create("smoke-delete-segment")
    real = Segment(channel=0, text="настоящая реплика")
    dup = Segment(channel=1, text="дубль", likely_bleed=True)
    s.protocol.segments = [real, dup]

    raised = False
    try:
        s.delete_segment(real.id)
    except PermissionError:
        raised = True
    assert raised, "нельзя удалить реплику, не помеченную дублем"
    assert len(s.protocol.segments) == 2, "неудачная попытка не должна ничего менять"

    result = s.delete_segment(dup.id)
    assert result == {"partner": None}, "дубль без bleed_pair_id — партнёра снимать не с кого"
    assert len(s.protocol.segments) == 1
    assert s.delete_segment("nonexistent") is None
    print("DELETE_SEGMENT_PERMISSION OK ✅")


def test_delete_segment_unflags_remaining_partner() -> None:
    """Реальный случай: пользователь удалил одну копию дубля, а вторая осталась висеть с
    флажком «вероятный дубль» и кнопкой удаления — хотя сравнивать её уже не с чем, это больше
    не дубль. Удаление должно снимать пометку с партнёра по `bleed_pair_id`."""
    from .models import Segment

    s = manager.create("smoke-delete-unflag-partner")
    a = Segment(channel=0, text="Где вы были вчера вечером?", likely_bleed=True,
                bleed_score=0.9, bleed_hint="likely_original")
    b = Segment(channel=1, text="Где вы были вчера вечером?", likely_bleed=True,
                bleed_score=0.9, bleed_hint="likely_leak")
    a.bleed_pair_id, b.bleed_pair_id = b.id, a.id
    s.protocol.segments = [a, b]

    result = s.delete_segment(b.id)   # удаляем протёкшую копию
    assert result is not None and result["partner"] is not None
    assert result["partner"]["id"] == a.id
    assert len(s.protocol.segments) == 1
    remaining = s.protocol.segments[0]
    assert not remaining.likely_bleed and remaining.bleed_hint == "" and not remaining.bleed_pair_id, (
        "у оставшейся реплики должна быть полностью снята пометка дубля")
    print("DELETE_SEGMENT_UNFLAGS_PARTNER OK ✅")


def test_rename_speaker_after_diarization() -> None:
    """После диаризации общего микрофона в ОДНОМ канале лежат разные спикеры, поэтому
    переименование должно идти по метке голоса, а не по каналу.

    Реальный баг: пользователь менял «Голос-2» на «Опрашивающий» и получал «Опрашивающий» у
    ВСЕХ реплик разом, включая чужие, — потому что `set_speaker` переписывает весь канал.
    Для основного режима (микрофон-на-участника) поведение «весь канал» остаётся верным: там
    канал и есть спикер, поэтому обе операции существуют рядом."""
    from .models import Segment

    s = manager.create("smoke-rename")
    for spk, txt in [("Голос-1", "первая"), ("Голос-2", "вторая"),
                     ("Голос-1", "третья"), ("Голос-2", "четвёртая")]:
        s.protocol.segments.append(Segment(channel=0, speaker=spk, speaker_auto=spk,
                                           start=0.0, end=1.0, text=txt, text_original=txt))

    n = s.rename_speaker("Голос-2", "Опрашивающий")
    labels = [sg.speaker for sg in s.protocol.segments]
    print(f"переименован один голос: обновлено {n}, метки {labels}")
    assert n == 2, f"должны смениться ровно две реплики, а не {n}"
    assert labels == ["Голос-1", "Опрашивающий", "Голос-1", "Опрашивающий"], labels

    # Канальное переименование (основной режим) по-прежнему берёт весь канал.
    m = s.set_speaker(0, "Следователь")
    assert m == 4 and all(sg.speaker == "Следователь" for sg in s.protocol.segments), m
    print("RENAME_SPEAKER OK ✅")


def test_crosstalk_state_reset_across_cycles() -> None:
    """Регрессия: `CrossTalkScorer` держал `_loud_since` между заходами записи. Если предыдущий
    цикл заканчивался ровно в момент «канал 0 громче канала 1», метка оставалась в словаре
    навсегда (для этого канала кадры перестали приходить, `pop` никогда не наступал). В новом
    заходе первый же похожий кадр считал разницу «устойчивой уже много минут» и канал мгновенно
    и необратимо давился как протечка — именно это увидел пользователь после 2-3 циклов
    старт/стоп: все реплики стали приписываться только одному каналу."""
    import time as _t

    from .audio.crosstalk import CrossTalkScorer

    def _cycle_then_new_frame(do_reset: bool) -> float:
        gate = CrossTalkScorer()
        loud = _speech(0.05, amp=0.1).astype(np.float32) / 32768.0
        quiet = _speech(0.05, amp=0.02).astype(np.float32) / 32768.0
        gate.score(0, loud)
        gate.score(1, quiet)             # конец "предыдущего цикла" — канал 1 в состоянии "тише"
        _t.sleep(0.3)                    # пауза между заходами записи (клик стоп -> клик старт)
        if do_reset:
            gate.reset_transient_state()  # это должен вызывать Session.attach_ws()
        gate.score(0, loud)
        score1, _ = gate.score(1, quiet)  # первый кадр НОВОГО захода
        return score1

    stale_score = _cycle_then_new_frame(do_reset=False)
    fresh_score = _cycle_then_new_frame(do_reset=True)
    print(f"без reset_transient_state(): bleed_score={stale_score:.2f} "
          f"(баг — мгновенная 'протечка' из-за не сброшенной метки времени)")
    print(f"с reset_transient_state():   bleed_score={fresh_score:.2f} "
          f"(норма — превышение внутри нового захода ещё не успело стать устойчивым)")
    assert stale_score > 0.0, "тест должен воспроизводить баг без reset (иначе тест не показателен)"
    assert fresh_score == 0.0, "reset_transient_state() должен устранять ложную мгновенную протечку"
    print("CROSSTALK_STATE_RESET OK ✅")


def test_stream_epoch_staleness() -> None:
    """Регрессия на реальный баг: `Session` переиспользуется между циклами «начать/остановить»
    записи (тот же session_id). Если ASR финала из СТАРОГО захода завершается уже ПОСЛЕ того,
    как пользователь начал новый заход (декодирование медленнее паузы между кликами — реалистично
    на слабом железе, см. README), текст не должен всплыть в живой ленте нового захода — но и не
    должен потеряться из протокола."""
    s = manager.create("smoke-epoch")
    published: list[dict] = []
    s._publish = lambda msg: published.append(msg)   # без реального asyncio/WS

    s.stream_epoch = 1
    hyp = UtteranceHypothesis(utterance_id="u_old", channel=1)
    words = [ASRWord(text="Нужно", start=0.0, end=0.3, prob=1.0),
             ASRWord(text="пройти", start=0.3, end=0.6, prob=1.0)]
    st = s.stream.setdefault(1, ChannelStreamState())

    # Пользователь нажал «начать запись» заново — новый заход, epoch увеличился (attach_ws)...
    s.stream_epoch = 2
    # ...а финал из СТАРОГО захода (epoch=1) только сейчас "доехал" с воркер-потока планировщика.
    s._finish_utterance(1, st, hyp, tail_words=words, window_start=0, epoch=1)

    assert not published, "финал из старого захода не должен публиковаться в текущую живую ленту"
    assert any(seg.text == "Нужно пройти" for seg in s.protocol.segments), (
        "но текст всё равно должен попасть в протокол — реальные слова нельзя терять")
    print("STREAM_EPOCH_STALENESS OK ✅ (не всплыл в новом заходе, но сохранён в протоколе)")


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
            s.ingest_streaming(ch, part)
        s.force_finalize_channel(ch)

    assert manager.scheduler.wait_idle(timeout=30.0), "ASR-планировщик не успел обработать задания"
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

    print("\n--- кросс-канальный гейтинг ---")
    test_crosstalk()

    print("\n--- бэкчаннел (−8 дБ, SNR-aware маржа) ---")
    test_crosstalk_backchannel()

    print("\n--- фильтр повторяющихся галлюцинаций ---")
    test_repetition_hallucination_filter()
    print("\n--- эхо initial_prompt и заученные титры ---")
    test_prompt_echo_filter()
    test_prompt_has_no_recent_speech()
    test_recording_started_at_set_once()
    test_get_or_load_restores_session_after_restart()
    test_crosstalk_dedup_prefers_asr_confidence_over_snr()
    test_crosstalk_dedup_falls_back_to_snr_when_confidence_is_close()
    test_delete_segment_only_allowed_for_bleed_marked()
    test_delete_segment_unflags_remaining_partner()
    test_rename_speaker_after_diarization()

    print("\n--- сброс cross-talk состояния между заходами записи ---")
    test_crosstalk_state_reset_across_cycles()

    print("\n--- устаревший заход записи (stream_epoch) ---")
    test_stream_epoch_staleness()

    print("\nSMOKE OK ✅")


if __name__ == "__main__":
    main()
