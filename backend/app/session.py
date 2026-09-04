"""Управление сессиями опроса: состояние протокола, запись фонограммы, real-time streaming ASR.

Фонограмма пишется как многоканальный микс (для MVP — суммируем каналы в один
WAV, чтобы плеер на клиенте проигрывал единую дорожку, а тайм-коды совпадали).

Стриминговая часть (Блок 2 плана): каждый канал ведёт `Endpointer` (когда реплика началась/
закончилась) и `RollingBuffer` (что именно распознавать), а сам ASR-инференс идёт через общий
`AsrScheduler` (Блок 0.1/2.2) — ни одна из этих частей не вызывает `ASRProvider.transcribe()`
напрямую. Результаты приходят на **отдельном потоке планировщика** и публикуются клиенту через
`asyncio.Queue`, поэтому все точки, где поток планировщика касается разделяемого состояния сессии
(`self.stream[...]`, `self.protocol.segments`), защищены `self._lock`.
"""
from __future__ import annotations

import asyncio
import bisect
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, telemetry
from .asr import get_asr
from .asr.base import ASRWord, dropped_note, hallucination_reason
from .asr.local_agreement import UtteranceHypothesis
from .asr.model_manager import ModelKey, ModelManager
from .asr.scheduler import AsrJob, AsrScheduler, PRIORITY_FINAL, PRIORITY_ONESHOT, PRIORITY_PARTIAL
from .assistant.questionnaire import AssistantSession, build_script
from .audio.crosstalk import CrossTalkScorer
from .audio.endpointer import Endpointer, EndpointEvent
from .audio.rolling_buffer import RollingBuffer
from .models import Edit, Protocol, QuestionnaireField, Segment, Word


@dataclass
class ChannelStreamState:
    endpointer: Endpointer = field(default_factory=Endpointer)
    rolling: RollingBuffer = field(default_factory=RollingBuffer)
    hypothesis: UtteranceHypothesis | None = None
    last_partial_submit_sample: int = 0
    last_partial_submit_ns: int = 0
    # Экспоненциальное среднее времени submit->result для partial-раундов этого канала.
    # Если планировщик не успевает за реальным временем, каданс новых partial-заданий
    # растягивается пропорционально (Блок 2.2 плана — "адаптивный каданс, backpressure").
    lag_ema_ms: float = 0.0


class Session:
    def __init__(self, session_id: str, scheduler: AsrScheduler, asr_model_key: ModelKey) -> None:
        self.id = session_id
        self.scheduler = scheduler
        self.asr_model_key = asr_model_key   # снапшот модели на момент создания сессии (Блок 4)
        self.dir = config.STORAGE_DIR / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audio_path = self.dir / "phonogram.wav"   # сводный микс для плеера

        self.protocol = Protocol(session_id=session_id, audio_path=str(self.audio_path))
        self.assistant = AssistantSession()
        self.speaker_names: dict[int, str] = {}     # канал -> метка
        self.gate = CrossTalkScorer()                # оценка протёкшего голоса (только для ASR)

        # По-канальные WAV (моно 16 кГц): таймлайн совпадает с тайм-кодами слов,
        # пишем ИСХОДНЫЙ (не гейтнутый) звук — Блок 0.4 плана, целостность фонограммы.
        self._wavs: dict[int, wave.Wave_write] = {}
        self._closed = False           # после сборки микса запись завершена
        self._lock = threading.RLock()

        # --- стриминг (Блок 2) ---
        self.stream: dict[int, ChannelStreamState] = {}
        self._recent_prompt: dict[int, str] = {}     # контекст для initial_prompt по каналу
        self._pending_finals = 0
        self.streaming = False
        self.stream_epoch = 0          # см. attach_ws() — «текущий заход» записи
        self.loop: asyncio.AbstractEventLoop | None = None
        self.out_queue: asyncio.Queue | None = None
        self.telemetry = telemetry.registry.for_session(
            session_id, jsonl_path=self.dir / "telemetry.jsonl")

    def channel_wav_path(self, channel: int) -> Path:
        return self.dir / f"phonogram_ch{channel}.wav"

    def _wav_writer(self, channel: int) -> wave.Wave_write:
        w = self._wavs.get(channel)
        if w is None:
            w = wave.open(str(self.channel_wav_path(channel)), "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(config.SAMPLE_RATE)
            self._wavs[channel] = w
        return w

    # --- спикеры ---------------------------------------------------------
    def speaker_label(self, channel: int) -> str:
        if channel in self.speaker_names:
            return self.speaker_names[channel]
        default = {0: "Интервьюер", 1: "Опрашиваемый"}.get(channel, f"Голос-{channel + 1}")
        return default

    def rename_speaker(self, old_label: str, new_label: str) -> int:
        """Переименовать КОНКРЕТНЫЙ голос по его текущей метке; вернуть число сегментов.

        Отдельно от `set_speaker` (который работает по каналу), потому что после офлайн-
        диаризации общего микрофона в одном канале лежат РАЗНЫЕ спикеры («Голос-1», «Голос-2»).
        Переименование по каналу в этом случае схлопывало весь протокол в одну метку — реальный
        баг: пользователь менял «Голос-2» на «Опрашивающий» и получал «Опрашивающий» у всех
        реплик разом, включая чужие.
        """
        with self._lock:
            n = 0
            for seg in self.protocol.segments:
                if seg.speaker == old_label and seg.speaker != new_label:
                    seg.edits.append(Edit(field="speaker", old=seg.speaker, new=new_label))
                    seg.speaker = new_label
                    n += 1
            # Пер-канальная метка задаёт имя БУДУЩИХ реплик канала, поэтому двигаем её следом.
            # Сегменты канала при этом не пересматриваем, и проверять нечего: запись в
            # `speaker_names` появляется только через `set_speaker` («весь канал — этот
            # голос»), а диаризация эту карту не трогает вовсе — значит канал, разложенный на
            # «Голос-1»/«Голос-2», сюда просто не попадёт.
            for ch, name in list(self.speaker_names.items()):
                if name == old_label:
                    self.speaker_names[ch] = new_label
                    self.protocol.speaker_map[ch] = new_label
            return n

    def set_speaker(self, channel: int, label: str) -> int:
        """Переименовать спикера ЦЕЛОГО канала; вернуть число обновлённых сегментов.
        Основной режим (микрофон-на-участника), где канал и есть спикер. Для протокола после
        диаризации нужен `rename_speaker` — там в канале несколько голосов."""
        with self._lock:
            self.speaker_names[channel] = label
            self.protocol.speaker_map[channel] = label
            n = 0
            for seg in self.protocol.segments:
                if seg.channel == channel and seg.speaker != label:
                    seg.edits.append(Edit(field="speaker", old=seg.speaker, new=label))
                    seg.speaker = label
                    n += 1
            return n

    # --- WS-подключение ---------------------------------------------------
    def attach_ws(self, loop: asyncio.AbstractEventLoop, out_queue: asyncio.Queue) -> None:
        # Сессия (тот же session_id) переживает клики «начать/остановить» — стоп не закрывает
        # Session, только WS-соединение. Если предыдущий цикл записи закрылся, пока ASR ещё не
        # успел обработать хвостовую реплику (задержка decode бывает намного больше пары секунд
        # на слабом железе — см. README «Производительность»), результат может прийти УЖЕ ПОСЛЕ
        # того, как пользователь начал новую запись. `stream_epoch` — метка «текущего захода»:
        # задания планировщика запоминают эпоху, в которой были поставлены, и результат из
        # старой эпохи не публикуется в текущую живую ленту (хотя в протокол всё равно
        # попадает — см. `_finish_utterance`), иначе фраза из прошлого захода всплывала бы в
        # новом (реальный баг, воспроизведённый пользователем).
        self.stream_epoch += 1
        # Кросс-ток гейт (Блок 3.7) держит покадровое состояние («с какого момента канал
        # устойчиво тише другого») между вызовами — без сброса метка из ПРЕДЫДУЩЕГО захода могла
        # остаться навсегда (если запись остановилась ровно в момент «другой канал громче») и в
        # новом заходе мгновенно засчитывалась бы как «устойчиво тише уже много минут», необратимо
        # давя канал как протечку с первого же похожего кадра. Реальный баг, найден на практике.
        self.gate.reset_transient_state()
        self.loop = loop
        self.out_queue = out_queue
        self.streaming = True
        # Якорь для «время начала/окончания опроса» (Блок 6, docgen.AUTO_FIELDS) — только на
        # ПЕРВЫЙ заход: относительный таймлайн сегментов (start/end, секунды от начала записи)
        # непрерывен на всю жизнь Session (RollingBuffer._total_samples не сбрасывается между
        # заходами), так что этот единственный якорь плюс segment.start/.end время реплики даёт
        # верное реальное время даже после «стоп» -> «начать снова».
        if self.protocol.recording_started_at is None:
            self.protocol.recording_started_at = time.time()

    def detach_ws(self) -> None:
        self.loop = None
        self.out_queue = None
        self.streaming = False

    def _publish(self, msg: dict) -> None:
        if self.loop is not None and self.out_queue is not None:
            self.loop.call_soon_threadsafe(self.out_queue.put_nowait, msg)

    def has_pending_finals(self) -> bool:
        with self._lock:
            return self._pending_finals > 0

    # --- контекст для ASR (initial_prompt) --------------------------------
    def _remember_prompt(self, channel: int, text: str) -> None:
        """Копит недавно распознанное для контекста. При ASR_PROMPT_RECENT_WORDS=0 (дефолт)
        не копит ничего — см. `_build_prompt`."""
        keep = config.ASR_PROMPT_RECENT_WORDS
        if keep <= 0:
            return
        words = (self._recent_prompt.get(channel, "") + " " + text).split()
        self._recent_prompt[channel] = " ".join(words[-keep:])

    def _build_prompt(self, channel: int) -> str:
        """Промпт для ASR. По умолчанию — ТОЛЬКО доменная подсказка, без недавних реплик.

        Дописывание недавнего текста выглядит бесплатным улучшением («дадим модели контекст»), но
        на реальной записи оно давало обратный эффект: на фоновом шуме модель возвращала этот
        контекст дословно, и в протокол шли повторы предыдущих вопросов с новыми тайм-кодами
        (замер и цифры — в config.py, ASR_PROMPT_RECENT_WORDS). Отличить такой повтор по тексту
        от настоящего нельзя — в допросе опрашиваемый и правда повторяет вопрос, — поэтому
        единственное честное место для лечения здесь: не подсовывать модели то, что она потом
        выдаст за распознанное."""
        base = config.WHISPER_PROMPT or ""
        recent = self._recent_prompt.get(channel, "")
        return (base + " " + recent).strip() if recent else base.strip()

    # --- real-time streaming ASR (Блок 2) ---------------------------------
    def ingest_streaming(self, channel: int, pcm_int16: np.ndarray) -> None:
        """Принять кусок PCM по каналу. Ничего не возвращает — результаты (partial/update/final)
        приходят асинхронно через `out_queue` (см. `attach_ws`)."""
        # Raw-звук всегда пишется как есть (Блок 0.4) — до применения гейта.
        with self._lock:
            if not self._closed:
                self._wav_writer(channel).writeframes(pcm_int16.tobytes())

        pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
        bleed_score, own_snr_db = self.gate.score(channel, pcm_f32)
        asr_input = np.zeros_like(pcm_f32) if bleed_score >= config.CROSSTALK_SCORE_THRESHOLD else pcm_f32

        with self._lock:
            st = self.stream.setdefault(channel, ChannelStreamState())
            if st.hypothesis is not None:
                st.hypothesis.add_snr_sample(own_snr_db)
            st.rolling.add(asr_input)
            events = st.endpointer.add(asr_input)
            for ev in events:
                if ev.kind == "utterance_start":
                    self._on_utterance_start(channel, st, ev)
                    if st.hypothesis is not None:   # свежесозданная гипотеза — не теряем этот кадр
                        st.hypothesis.add_snr_sample(own_snr_db)
                elif ev.kind == "utterance_end":
                    self._on_utterance_end(channel, st, ev)

            if (st.hypothesis is not None
                    and not st.endpointer.is_closed(st.hypothesis.utterance_id)):
                # Пороги — от устройства ЭТОЙ сессии (asr_model_key), а не от модуль-уровневых
                # констант: те заморожены на импорте и остаются GPU-шными даже после отката
                # преполёта на CPU (см. config.asr_overload_lag_ms).
                if st.lag_ema_ms > config.asr_overload_lag_ms(self.asr_model_key[1]):
                    # Модель фундаментально не успевает за реальным временем на этом железе
                    # (naблюдалось: decode ~10с на 5с аудио на turbo/CPU — это не вопрос
                    # каданса, никакой откат внутри разумных пределов не поможет). В таком
                    # состоянии partial-задания только отнимают ёмкость воркеров у финалов и
                    # устаревают раньше, чем будут обработаны — не ставим новые вовсе, пока не
                    # финал не начнёт ставит их снова после закрытия текущей реплики.
                    pass
                else:
                    new_ms = st.rolling.new_audio_ms_since(st.last_partial_submit_sample)
                    # Адаптивный каданс без искусственного потолка — если лаг реально ~10с,
                    # требуемый интервал должен быть порядка 10с+, а не капаться в 3.6с (это и
                    # была первая, недостаточная версия фикса: потолок глушил сам смысл отката).
                    required_ms = max(config.asr_update_ms(self.asr_model_key[1]),
                                      st.lag_ema_ms * 1.3)
                    if new_ms >= required_ms:
                        self._submit_partial(channel, st)

    def _on_utterance_start(self, channel: int, st: ChannelStreamState, ev: EndpointEvent) -> None:
        st.hypothesis = UtteranceHypothesis(
            utterance_id=ev.utterance_id, channel=channel,
            committed_boundary_sample=ev.start_sample,
        )
        st.last_partial_submit_sample = ev.start_sample
        self.telemetry.mark(ev.utterance_id, channel, speech_start_ts=telemetry.now_ns())

    def _on_utterance_end(self, channel: int, st: ChannelStreamState, ev: EndpointEvent) -> None:
        self.telemetry.mark(ev.utterance_id, channel, speech_end_ts=telemetry.now_ns())
        self._submit_final(channel, st, ev)

    def _submit_partial(self, channel: int, st: ChannelStreamState) -> None:
        hyp = st.hypothesis
        if hyp is None:
            return
        audio, window_start = st.rolling.window_since(hyp.committed_boundary_sample)
        st.last_partial_submit_sample = st.rolling.total_samples
        if audio.size < config.SAMPLE_RATE // 10:   # <~0.1c — decode бессмысленен
            return

        uid = hyp.utterance_id
        gen = st.endpointer.generation_of(uid)
        job = AsrJob(
            priority=PRIORITY_PARTIAL, seq=self.scheduler.next_seq(),
            session_id=self.id, channel=channel, kind="partial",
            utterance_id=uid, generation=gen, epoch=self.stream_epoch, audio=audio,
            initial_prompt=self._build_prompt(channel), provider_key=self.asr_model_key,
            on_start=lambda job, ch=channel: self._on_job_started(job, ch),
            on_result=lambda job, result, ws=window_start: self._on_partial_result(job, result, ws),
            is_current=lambda uid=uid, gen=gen, st=st: (
                st.endpointer.generation_of(uid) == gen and not st.endpointer.is_closed(uid)),
        )
        if self.scheduler.submit_partial(job):
            st.last_partial_submit_ns = telemetry.now_ns()
            self.telemetry.mark(uid, channel, asr_enqueued_ts=st.last_partial_submit_ns)

    def _submit_final(self, channel: int, st: ChannelStreamState, ev: EndpointEvent) -> None:
        hyp = st.hypothesis
        if hyp is None:
            return
        epoch = self.stream_epoch   # захватываем «заход» на момент постановки — см. attach_ws()
        audio, window_start = st.rolling.window_since(hyp.committed_boundary_sample)

        if audio.size < config.SAMPLE_RATE // 20:   # почти ничего не осталось после коммита
            self._finish_utterance(channel, st, hyp, tail_words=[], window_start=window_start,
                                    epoch=epoch)
            return

        self._pending_finals += 1
        job = AsrJob(
            priority=PRIORITY_FINAL, seq=self.scheduler.next_seq(),
            session_id=self.id, channel=channel, kind="final",
            utterance_id=hyp.utterance_id, generation=ev.generation, epoch=epoch, audio=audio,
            initial_prompt=self._build_prompt(channel), provider_key=self.asr_model_key,
            # Гипотеза захватывается напрямую в замыкании, а не через `self.stream[channel]` —
            # к моменту, когда планировщик доберётся до этого задания, канал вполне может уже
            # открыть СЛЕДУЮЩУЮ реплику (новый st.hypothesis), и без явного захвата колбэк
            # склеил бы текст текущей реплики с состоянием уже другой.
            on_start=lambda job, ch=channel: self._on_job_started(job, ch),
            on_result=lambda job, result, ws=window_start, h=hyp: self._on_final_result(job, result, ws, h),
            is_current=lambda: True,   # финал по этой реплике уже не может «устареть»
        )
        self.telemetry.mark(hyp.utterance_id, channel, asr_enqueued_ts=telemetry.now_ns())
        self.scheduler.submit_final(job)

    def force_finalize_channel(self, channel: int) -> None:
        """Принудительно закрыть открытую реплику канала (например, по `{"type":"stop"}`)."""
        with self._lock:
            st = self.stream.get(channel)
            if st is None:
                return
            ev = st.endpointer.force_end()
            if ev is not None:
                # Под тем же `self._lock` (RLock — реентерабелен): `_submit_final` может
                # синхронно дойти до `_finish_utterance` для совсем короткого хвоста, а та
                # мутирует `self.protocol.segments`/`st.hypothesis` — раньше это происходило
                # вне лока и могло гоняться с колбэками планировщика на другом потоке.
                self._submit_final(channel, st, ev)

    # --- колбэки планировщика (выполняются на worker-потоке(-ах)!) --------
    def _on_job_started(self, job: AsrJob, channel: int) -> None:
        self.telemetry.mark(job.utterance_id, channel, asr_started_ts=telemetry.now_ns())

    def _update_lag_ema(self, st: ChannelStreamState) -> None:
        """Вызывать под `self._lock`. Экспоненциальное среднее полного времени
        submit->result partial-раунда — сигнал перегрузки для адаптивного каданса
        (`ingest_streaming`), не зависит от того, актуален ли ещё сам результат."""
        if not st.last_partial_submit_ns:
            return
        elapsed_ms = (telemetry.now_ns() - st.last_partial_submit_ns) / 1e6
        st.lag_ema_ms = elapsed_ms if st.lag_ema_ms == 0 else 0.7 * st.lag_ema_ms + 0.3 * elapsed_ms

    def _on_partial_result(self, job: AsrJob, result, window_start: int) -> None:
        sample_rate = config.SAMPLE_RATE
        msg: dict | None = None
        finished_ts = telemetry.now_ns()
        with self._lock:
            st = self.stream.get(job.channel)
            if st is not None:
                self._update_lag_ema(st)
            if st is None or st.hypothesis is None or st.hypothesis.utterance_id != job.utterance_id:
                return   # реплика уже сменилась — поздний partial отбрасываем
            if st.endpointer.is_closed(job.utterance_id):
                return   # final уже произошёл/происходит — поздний partial не публикуем
            if job.epoch != self.stream_epoch:
                return   # с прошлого «захода» записи — живая лента уже про другой заход

            tail_words = [
                ASRWord(text=w.text, start=window_start / sample_rate + w.start,
                        end=window_start / sample_rate + w.end, prob=w.prob)
                for w in result.words
            ]
            boundary_time = st.hypothesis.committed_boundary_sample / sample_rate
            tail_words = [w for w in tail_words if w.end > boundary_time]
            update = st.hypothesis.apply_partial(tail_words, sample_rate)
            if update["text"]:
                msg = {
                    "type": "asr_update", "channel": job.channel,
                    "utterance_id": job.utterance_id, "text": update["text"],
                    "stable_word_count": update["stable_word_count"],
                }
        self.telemetry.mark(job.utterance_id, job.channel, asr_finished_ts=finished_ts)
        if msg is not None:
            self.telemetry.mark_first_partial(job.utterance_id, job.channel)
            self._publish(msg)

    def _on_final_result(self, job: AsrJob, result, window_start: int,
                          hyp: UtteranceHypothesis) -> None:
        sample_rate = config.SAMPLE_RATE
        self.telemetry.mark(job.utterance_id, job.channel, asr_finished_ts=telemetry.now_ns())
        tail_words = [
            ASRWord(text=w.text, start=window_start / sample_rate + w.start,
                    end=window_start / sample_rate + w.end, prob=w.prob)
            for w in result.words
        ]
        boundary_time = hyp.committed_boundary_sample / sample_rate
        tail_words = [w for w in tail_words if w.end > boundary_time]
        with self._lock:
            st = self.stream.get(job.channel)
            self._finish_utterance(job.channel, st, hyp, tail_words, window_start, job.epoch)
            self._pending_finals = max(0, self._pending_finals - 1)

    def _finish_utterance(self, channel: int, st: ChannelStreamState | None,
                           hyp: UtteranceHypothesis, tail_words: list[ASRWord],
                           window_start: int, epoch: int) -> None:
        """Вызывать под `self._lock`. Собирает финальный `Segment` из committed_words + tail_words
        ИМЕННО захваченной `hyp` (не `st.hypothesis` — см. комментарий в `_submit_final`),
        публикует `asr_final`, сбрасывает состояние реплики канала.

        `epoch` — «заход» записи, в котором это задание было поставлено (см. `attach_ws`). Сегмент
        всегда попадает в протокол (текст реально прозвучал, терять его нельзя), но публикуется в
        живую WS-ленту, только если пользователь всё ещё в ТОМ ЖЕ заходе — иначе фраза из
        предыдущего «начать/остановить» цикла (декодирование которой просто заняло дольше, чем
        пауза между кликами) всплыла бы в уже новой записи, которую видит пользователь сейчас."""
        all_words = hyp.committed_words + tail_words
        text = UtteranceHypothesis.finalize_text(all_words)

        # Последний рубеж перед протоколом. Провайдер фильтрует то, что выдала модель за один
        # проход, а сюда текст приходит СОБРАННЫМ из слов, закоммиченных партиалами, плюс хвост
        # финала, и потом ещё подрезанным по границе — то есть это уже другая строка. Реальный
        # случай с живой записи: партиал выдал «Это опрос.» — не эхо промпта, слова «это» в
        # промпте нет, фильтр провайдера пропустил, — LocalAgreement закоммитил слова, а после
        # подрезки в сегменте осталось голое «опрос.», то есть чистое эхо. Проверяем ровно то,
        # что пойдёт в протокол.
        reason = hallucination_reason(text, config.WHISPER_PROMPT)
        if reason:
            # Отбраковка тут молчала совсем: Segment не создавался, и реплика пропадала из
            # протокола бесследно. Собранный из партиалов текст в лог — единственная
            # возможность потом понять, что именно выбросили и каким правилом.
            print(dropped_note(f"сессия {self.id}/канал {channel}", reason, text))
            text = ""

        is_current_epoch = (epoch == self.stream_epoch)

        msg = {"type": "asr_final", "channel": channel, "utterance_id": hyp.utterance_id, "text": text}
        if text:
            label = self.speaker_label(channel)
            # created_at — РЕАЛЬНЫЙ момент речи (recording_started_at + позиция на аудио-
            # таймлайне), а не момент, когда до этой реплики дошла очередь ASR. Иначе несколько
            # реплик, ждавших очереди и финализированных подряд сразу после «Стоп», получили бы
            # created_at, совпадающий с точностью до секунды, хотя реально прозвучали в разное
            # время — и «время начала»/«время окончания» опроса (Блок 6, docgen.AUTO_FIELDS)
            # совпадали бы (реальный баг, воспроизведённый пользователем).
            recorded_at = (self.protocol.recording_started_at or time.time()) + all_words[0].start
            seg = Segment(
                channel=channel, speaker=label, speaker_auto=f"Голос-{channel + 1}",
                start=all_words[0].start, end=all_words[-1].end, text=text, text_original=text,
                words=[Word(text=w.text, start=w.start, end=w.end, prob=w.prob) for w in all_words],
                own_snr_db=hyp.avg_snr_db, created_at=recorded_at,
            )
            # Вставляем по времени начала, а не в конец: финал из устаревшего захода записи
            # (см. epoch выше) может «доехать» позже, чем сегменты нового захода, которые уже
            # успели попасть в протокол — простой append() расположил бы его не по хронологии.
            bisect.insort(self.protocol.segments, seg, key=lambda s: s.start)
            self._remember_prompt(channel, text)
            self._maybe_mark_crosstalk_duplicate(seg)
            msg.update(seg.to_ws_dict())

        if st is not None and st.hypothesis is hyp:
            st.hypothesis = None
            st.rolling.reset()

        if is_current_epoch:
            self._publish(msg)
        self.telemetry.mark(hyp.utterance_id, channel, final_ts=telemetry.now_ns())

    def _maybe_mark_crosstalk_duplicate(self, new_seg: Segment) -> None:
        """Пост-ASR дедупликация (Блок 3.7). Вызывать под `self._lock` (уже держит
        `_finish_utterance`). Сравнивает уже РАСПОЗНАННЫЙ текст+время с сегментами других каналов
        — надёжнее, чем сырая покадровая энергия (см. `crosstalk.py`): реальный тест показал, что
        при близко расположенных микрофонах разница энергии часто в пределах шума, а вот
        текстовое совпадение между «протёкшей» и «настоящей» репликой — почти всегда почти
        дословное. Ничего не удаляет.

        При близких микрофонах система НЕ МОЖЕТ надёжно решить, кто из двух реальный автор —
        реальный случай: по одному только SNR дублем помечалась именно та копия, которую человек
        произнёс на самом деле. Поэтому помечаются ОБЕ копии одинаково (`likely_bleed`), решение
        и удаление лишней — за оператором (см. `delete_segment`). Уверенность ASR (среднее
        word.prob) и SNR при этом не выбрасываются, а становятся ориентиром `bleed_hint` на
        каждой копии — подсказкой, не решением: протёкший/искажённый звук ASR обычно распознаёт
        менее уверенно, даже когда SNR (ненадёжный при близких микрофонах) вводит в заблуждение;
        SNR — резервный критерий подсказки, когда уверенность у обеих копий почти одинаковая."""
        import difflib

        def avg_word_prob(seg: Segment) -> float:
            if not seg.words:
                return 1.0   # нет пословных данных — не штрафуем, отдаём решение SNR
            return sum(w.prob for w in seg.words) / len(seg.words)

        for other in self.protocol.segments:
            if other is new_seg or other.channel == new_seg.channel:
                continue
            if other.likely_bleed:
                continue   # уже помечен дублем, повторно не сравниваем
            if not (new_seg.start < other.end and other.start < new_seg.end):
                continue   # нет пересечения по времени
            similarity = difflib.SequenceMatcher(
                None, new_seg.text.lower(), other.text.lower()).ratio()
            if similarity < config.CROSSTALK_DEDUPE_MIN_SIMILARITY:
                continue
            is_short = (len(new_seg.words) <= config.CROSSTALK_DEDUPE_SHORT_WORDS
                        or len(other.words) <= config.CROSSTALK_DEDUPE_SHORT_WORDS)
            snr_gap = abs(new_seg.own_snr_db - other.own_snr_db)
            if is_short and snr_gap < config.CROSSTALK_DEDUPE_SHORT_SNR_GAP_DB:
                # Короткие реплики («да»/«угу») совпадают по тексту почти всегда тривиально —
                # без явного разрыва по SNR это не доказательство протечки (см. Блок 3.7 плана).
                continue

            new_prob, other_prob = avg_word_prob(new_seg), avg_word_prob(other)
            if abs(new_prob - other_prob) >= config.CROSSTALK_DEDUPE_CONFIDENCE_GAP:
                weaker, stronger = ((new_seg, other) if new_prob < other_prob else (other, new_seg))
            else:
                weaker, stronger = ((new_seg, other) if new_seg.own_snr_db < other.own_snr_db
                                     else (other, new_seg))
            new_seg.likely_bleed = True
            new_seg.bleed_score = round(similarity, 3)
            other.likely_bleed = True
            other.bleed_score = round(similarity, 3)
            new_seg.bleed_pair_id = other.id
            other.bleed_pair_id = new_seg.id
            weaker.bleed_hint = "likely_leak"
            stronger.bleed_hint = "likely_original"
            # `other` уже был отправлен клиенту раньше как обычный — досылаем обновление, иначе
            # живой UI не узнает о пометке без перезагрузки протокола. `new_seg` досылать не
            # нужно — вызывающий код (`_finish_utterance`) сам публикует его чуть позже, уже
            # с актуальными полями.
            self._publish({"type": "segment_update", **other.to_ws_dict()})

    def delete_segment(self, seg_id: str) -> dict | None:
        """Удаляет сегмент — ТОЛЬКО помеченный вероятным дублем протёкшего голоса (Блок 3.7).
        Не общая функция удаления реплик: в этом приложении текст иначе никогда не стирается
        молча, только правится поверх с аудитом (см. `edit_segment`/`Edit`) — здесь же речь о
        конкретном инструменте разбора дублей, когда у одной и той же фразы есть заведомо лишняя
        копия на другом канале. Возвращает `None`, если сегмент не найден. Поднимает
        `PermissionError`, если сегмент не помечен дублем — иначе кнопка в клиенте превратилась
        бы в способ незаметно вычистить любую настоящую реплику из протокола.

        Если у удалённого сегмента была пара (`bleed_pair_id`) — снимает пометку с неё: без
        партнёра сравнивать больше не с чем, оставшаяся реплика уже не дубль (реальный случай:
        удалили одну копию, а вторая осталась висеть с флажком и кнопкой удаления). Возвращает
        `{"partner": dict | None}` — партнёра нужно вернуть В ОТВЕТЕ, а не только толкнуть через
        WS: кнопка удаления в клиенте нужна именно при разборе уже ЗАВЕРШЁННОЙ сессии, когда
        live-соединения обычно уже нет и `_publish` никуда не доходит."""
        with self._lock:
            for i, seg in enumerate(self.protocol.segments):
                if seg.id == seg_id:
                    if not seg.likely_bleed:
                        raise PermissionError(
                            "удалить можно только реплику, помеченную вероятным дублем")
                    del self.protocol.segments[i]
                    partner_dict = None
                    if seg.bleed_pair_id:
                        for other in self.protocol.segments:
                            if other.id == seg.bleed_pair_id:
                                other.likely_bleed = False
                                other.bleed_score = 0.0
                                other.bleed_hint = ""
                                other.bleed_pair_id = None
                                partner_dict = other.to_ws_dict()
                                self._publish({"type": "segment_update", **partner_dict})
                                break
                    return {"partner": partner_dict}
            return None

    # --- правки ----------------------------------------------------------
    def edit_segment(self, seg_id: str, new_text: str) -> bool:
        with self._lock:
            for seg in self.protocol.segments:
                if seg.id == seg_id:
                    if seg.text != new_text:
                        seg.edits.append(Edit(field="text", old=seg.text, new=new_text))
                        seg.text = new_text
                        seg.edited = True
                    return True
            return False

    # --- завершение / сохранение ----------------------------------------
    def save(self) -> Path:
        self.protocol.questionnaire = [
            QuestionnaireField(**f) for f in self.assistant.to_fields()
        ]
        self.protocol.asr_model, self.protocol.asr_device, self.protocol.asr_compute = self.asr_model_key
        path = self.dir / "protocol.json"
        path.write_text(self.protocol.model_dump_json(indent=2), encoding="utf-8")
        return path

    def close(self) -> None:
        with self._lock:
            for w in self._wavs.values():
                try:
                    w.close()
                except Exception:
                    pass
            self._wavs.clear()
            self._closed = True

    def finalize_alignment(self, aligner) -> int:
        """Уточняет тайм-коды слов по фонограмме (forced alignment). Возвращает
        число обработанных сегментов."""
        self.close()  # дописать WAV-заголовки
        pad = 0.2
        done = 0
        by_channel: dict[int, list[Segment]] = {}
        for seg in self.protocol.segments:
            by_channel.setdefault(seg.channel, []).append(seg)

        for channel, segs in by_channel.items():
            p = self.channel_wav_path(channel)
            if not p.exists():
                continue
            with wave.open(str(p), "rb") as r:
                audio = np.frombuffer(r.readframes(r.getnframes()), dtype="<i2")
            audio_f32 = audio.astype(np.float32) / 32768.0
            total = len(audio_f32) / config.SAMPLE_RATE
            for seg in segs:
                if not seg.text.strip():
                    continue
                s0 = max(0.0, seg.start - pad)
                s1 = min(total, seg.end + pad)
                a = audio_f32[int(s0 * config.SAMPLE_RATE):int(s1 * config.SAMPLE_RATE)]
                words = aligner.align(a, seg.text, config.SAMPLE_RATE)
                if not words:
                    continue
                seg.words = [Word(text=w["text"], start=s0 + w["start"],
                                  end=s0 + w["end"], prob=w["prob"]) for w in words]
                seg.start = seg.words[0].start
                seg.end = seg.words[-1].end
                seg.aligned = True
                done += 1
        return done

    def diarize_single_mic(self, diarizer, channel: int = 0,
                           num_speakers: int | None = None) -> int:
        """Разводит сегменты одного общего канала по голосам (ECAPA + кластеризация).
        Возвращает число распознанных голосов."""
        self.close()
        p = self.channel_wav_path(channel)
        if not p.exists():
            return 0
        with wave.open(str(p), "rb") as r:
            audio = np.frombuffer(r.readframes(r.getnframes()), dtype="<i2")
        audio_f32 = audio.astype(np.float32) / 32768.0
        sr = config.SAMPLE_RATE

        segs = [s for s in self.protocol.segments if s.channel == channel]
        embeddings, idx = [], []
        for i, seg in enumerate(segs):
            a = audio_f32[int(seg.start * sr):int(seg.end * sr)]
            emb = diarizer.embed(a)
            if emb is not None:
                embeddings.append(emb)
                idx.append(i)
        if not embeddings:
            return 0

        labels = diarizer.cluster(embeddings, num_speakers)
        for pos, lab in zip(idx, labels):
            seg = segs[pos]
            name = f"Голос-{lab + 1}"
            if seg.speaker != name:
                seg.edits.append(Edit(field="speaker", old=seg.speaker, new=name))
                seg.speaker = name
            seg.speaker_auto = name
        return len(set(labels))

    def build_mix(self) -> Path:
        """Сводит по-канальные WAV в один моно-микс для плеера (таймлайн сохраняется)."""
        self.close()
        channels = sorted(self.protocol.speaker_map.keys()) or \
            [int(p.stem.split("ch")[-1]) for p in self.dir.glob("phonogram_ch*.wav")]
        tracks = []
        for ch in channels:
            p = self.channel_wav_path(ch)
            if not p.exists():
                continue
            with wave.open(str(p), "rb") as r:
                tracks.append(np.frombuffer(r.readframes(r.getnframes()), dtype="<i2"))
        if not tracks:
            return self.audio_path
        n = max(len(t) for t in tracks)
        mix = np.zeros(n, dtype=np.float32)
        for t in tracks:
            mix[: len(t)] += t.astype(np.float32)
        mix = np.clip(mix, -32768, 32767).astype("<i2")
        with wave.open(str(self.audio_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(config.SAMPLE_RATE)
            w.writeframes(mix.tobytes())
        return self.audio_path


class SessionManager:
    def __init__(self) -> None:
        self.models = ModelManager()
        self.scheduler = AsrScheduler(self.models)
        self._active_key: ModelKey = self.models.default_key()
        self._aligner = None
        self._diarizer = None
        self.sessions: dict[str, Session] = {}

    @property
    def active_model(self) -> str:
        return self._active_key[0]

    @property
    def active_key(self) -> ModelKey:
        """Полный ключ (модель, устройство, compute) — нужен диагностике `/api/health`, чтобы
        показать РЕАЛЬНОЕ устройство, а не то, что просили: при провале GPU преполёт (main.py)
        откатывает активный ключ на CPU."""
        return self._active_key

    def set_active_model(self, key: ModelKey) -> None:
        """Меняет модель для НОВЫХ сессий (Блок 4) — уже созданные сессии хранят свой снапшот
        `asr_model_key` и не затрагиваются (см. `Session.__init__`)."""
        self._active_key = key

    def any_streaming(self) -> bool:
        return any(s.streaming for s in self.sessions.values())

    def transcribe_oneshot(self, pcm_int16: np.ndarray, initial_prompt: str | None = None) -> str:
        """Разовая транскрипция вне контекста сессии (анкета, `/api/transcribe`) — тоже идёт через
        общий планировщик (Блок 0.1), не напрямую в модель, чтобы не гоняться с live-стримингом
        за один и тот же инстанс. Синхронный (блокирующий) вызов — вызывающий REST-хендлер сам
        оборачивает его в `asyncio.to_thread`.

        `initial_prompt` — оверрайд на конкретный вызов (Блок 6: контекст анкеты конкретной
        сессии, см. main.py::transcribe_oneshot); `None` — обычное поведение, статичный
        `config.WHISPER_PROMPT`."""
        pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
        done = threading.Event()
        holder: dict = {}

        def on_result(job, result):
            holder["text"] = result.text
            done.set()

        job = AsrJob(
            priority=PRIORITY_ONESHOT, seq=self.scheduler.next_seq(),
            session_id="_oneshot", channel=-1, kind="oneshot",
            audio=pcm_f32,
            initial_prompt=initial_prompt if initial_prompt is not None else config.WHISPER_PROMPT,
            provider_key=self._active_key, on_result=on_result,
        )
        self.scheduler.submit_oneshot(job)
        done.wait(timeout=60.0)
        return holder.get("text", "")

    @property
    def aligner(self):
        if self._aligner is None:
            from .finalize.aligner import WordAligner
            self._aligner = WordAligner()
        return self._aligner

    @property
    def diarizer(self):
        if self._diarizer is None:
            from .finalize.diarizer import Diarizer
            self._diarizer = Diarizer()
        return self._diarizer

    def create(self, session_id: str) -> Session:
        s = Session(session_id, self.scheduler, self._active_key)
        self.sessions[session_id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def get_or_load(self, session_id: str) -> Session | None:
        """Как `get()`, но если сессии нет в памяти (типично — после перезапуска backend), а на
        диске есть `protocol.json` (Блок 7: «открыть прошлый допрос повторно») — восстанавливает
        `Session` из него. Без этого просмотр/повторная генерация .docx старого допроса работали
        бы только пока backend ни разу не перезапускали с момента записи — то есть почти никогда
        в реальном ежедневном использовании."""
        s = self.sessions.get(session_id)
        if s is not None:
            return s
        path = config.STORAGE_DIR / session_id / "protocol.json"
        if not path.exists():
            return None
        try:
            protocol = Protocol.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        s = Session(session_id, self.scheduler, self._active_key)
        s.protocol = protocol
        if protocol.template_snapshot is not None:
            # Восстанавливаем состояние анкеты, СОГЛАСОВАННОЕ с уже загруженным protocol.
            # questionnaire — иначе повторное «Сохранить» на реоткрытой сессии переписало бы его
            # пустым (AssistantSession() по умолчанию — пустой сценарий, to_fields() тогда вернул
            # бы []). Анкета считается пройденной целиком, а не «на середине».
            s.assistant.load_script(build_script(protocol.template_snapshot.steps))
            s.assistant.answers = {f.key: f.value for f in protocol.questionnaire}
            s.assistant.idx = len(s.assistant.script)
            s.assistant.finished = True
        self.sessions[session_id] = s
        return s


manager = SessionManager()
