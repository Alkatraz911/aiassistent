"""Управление сессиями опроса: состояние протокола, запись фонограммы, ASR.

Фонограмма пишется как многоканальный микс (для MVP — суммируем каналы в один
WAV, чтобы плеер на клиенте проигрывал единую дорожку, а тайм-коды совпадали).
"""
from __future__ import annotations

import json
import threading
import wave
from pathlib import Path

import numpy as np

from . import config
from .asr import get_asr
from .assistant.questionnaire import AssistantSession
from .audio.buffer import ChunkBuffer
from .models import Edit, Protocol, QuestionnaireField, Segment, Word


class Session:
    def __init__(self, session_id: str, asr) -> None:
        self.id = session_id
        self.asr = asr
        self.dir = config.STORAGE_DIR / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audio_path = self.dir / "phonogram.wav"

        self.protocol = Protocol(session_id=session_id, audio_path=str(self.audio_path))
        self.assistant = AssistantSession()
        self.buffers: dict[int, ChunkBuffer] = {}
        self.speaker_names: dict[int, str] = {}     # канал -> метка

        # WAV-писатель (моно 16 кГц, сумма каналов)
        self._wav = wave.open(str(self.audio_path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)                   # int16
        self._wav.setframerate(config.SAMPLE_RATE)
        self._lock = threading.Lock()

    # --- спикеры ---------------------------------------------------------
    def speaker_label(self, channel: int) -> str:
        if channel in self.speaker_names:
            return self.speaker_names[channel]
        # дефолтные роли для первых двух микрофонов
        default = {0: "Интервьюер", 1: "Опрашиваемый"}.get(channel, f"Голос-{channel + 1}")
        return default

    def set_speaker(self, channel: int, label: str) -> int:
        """Переименовать спикера; вернуть число обновлённых сегментов."""
        self.speaker_names[channel] = label
        self.protocol.speaker_map[channel] = label
        n = 0
        for seg in self.protocol.segments:
            if seg.channel == channel:
                if seg.speaker != label:
                    seg.edits.append(Edit(field="speaker", old=seg.speaker, new=label))
                    seg.speaker = label
                    n += 1
        return n

    # --- аудио + ASR -----------------------------------------------------
    def ingest(self, channel: int, pcm_int16: np.ndarray) -> list[Segment]:
        """Принять кусок PCM по каналу, вернуть новые готовые сегменты."""
        # запись в фонограмму
        with self._lock:
            self._wav.writeframes(pcm_int16.tobytes())

        pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
        buf = self.buffers.setdefault(channel, ChunkBuffer())
        new_segments: list[Segment] = []
        for audio, offset in buf.add(pcm_f32):
            seg = self._transcribe_chunk(channel, audio, offset)
            if seg is not None:
                new_segments.append(seg)
        return new_segments

    def _transcribe_chunk(self, channel: int, audio: np.ndarray, offset: float) -> Segment | None:
        result = self.asr.transcribe(audio, config.SAMPLE_RATE)
        if not result.text.strip():
            return None
        words = [Word(text=w.text, start=offset + w.start, end=offset + w.end, prob=w.prob)
                 for w in result.words]
        start = words[0].start if words else offset
        end = words[-1].end if words else offset + len(audio) / config.SAMPLE_RATE
        label = self.speaker_label(channel)
        seg = Segment(
            channel=channel, speaker=label, speaker_auto=f"Голос-{channel + 1}",
            start=start, end=end, text=result.text, text_original=result.text, words=words,
        )
        self.protocol.segments.append(seg)
        return seg

    # --- правки ----------------------------------------------------------
    def edit_segment(self, seg_id: str, new_text: str) -> bool:
        for seg in self.protocol.segments:
            if seg.id == seg_id:
                if seg.text != new_text:
                    seg.edits.append(Edit(field="text", old=seg.text, new=new_text))
                    seg.text = new_text
                    seg.edited = True
                return True
        return False

    # --- завершение / сохранение ----------------------------------------
    def finalize_channel(self, channel: int) -> list[Segment]:
        buf = self.buffers.get(channel)
        if not buf:
            return []
        tail = buf.finalize()
        if tail is None:
            return []
        audio, offset = tail
        seg = self._transcribe_chunk(channel, audio, offset)
        return [seg] if seg else []

    def save(self) -> Path:
        self.protocol.questionnaire = [
            QuestionnaireField(**f) for f in self.assistant.to_fields()
        ]
        path = self.dir / "protocol.json"
        path.write_text(self.protocol.model_dump_json(indent=2), encoding="utf-8")
        return path

    def close(self) -> None:
        with self._lock:
            try:
                self._wav.close()
            except Exception:
                pass


class SessionManager:
    def __init__(self) -> None:
        self._asr = None
        self.sessions: dict[str, Session] = {}

    @property
    def asr(self):
        if self._asr is None:
            self._asr = get_asr()
            self._asr.warmup()
        return self._asr

    def create(self, session_id: str) -> Session:
        s = Session(session_id, self.asr)
        self.sessions[session_id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)


manager = SessionManager()
