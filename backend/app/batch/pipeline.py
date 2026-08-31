"""Пакетная расшифровка одной записи: декодирование -> блоки -> ASR -> голоса.

Отличия от live-режима не косметические, и каждое — следствие того, что здесь нет требования
успевать за реальным временем:

  beam_size                   5 (как у финалов live), а не 1 — считаем один раз, качеством
                              разбрасываться незачем;
  condition_on_previous_text  False. Это главная развилка длинных записей: контекст предыдущего
                              окна улучшает связность и пунктуацию, но именно он превращает
                              одну галлюцинацию в самоподдерживающуюся петлю на десятки минут
                              (модель продолжает то, что сама же выдумала). В протоколе опроса
                              выдуманный текст дороже пропущенной запятой — см. `--context`,
                              если материал чистый и связность важнее;
  word_timestamps             True — по ним точнее границы реплик, а значит и нарезка окон под
                              диаризацию, и тайм-коды в .srt/.json;
  фильтры галлюцинаций        те же три, что в live (`app/asr/base.py`), но применяются
                              ПОСЕГМЕНТНО. Это принципиально: `has_repeating_ngram` на тексте
                              целого блока в 10 минут сработал бы от любого настоящего повтора
                              где-нибудь в середине и выбросил бы весь блок.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config
from ..asr.base import dropped_note, hallucination_reason
from .blocks import plan_blocks
from .decode import (close_pcm, decode_to_pcm, open_pcm, probe_duration,
                     to_float32)
from .progress import Reporter, hms

CHECKPOINT_VERSION = 1
# Максимальное окно, с которого берётся ECAPA-эмбеддинг реплики. Больше не нужно: голос
# опознаётся по нескольким секундам, а на длинной реплике лишний звук только замедляет.
DIARIZE_WINDOW_S = 10.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text,
                "speaker": self.speaker, "words": self.words}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(start=d["start"], end=d["end"], text=d["text"],
                   speaker=d.get("speaker"), words=d.get("words") or [])


@dataclass
class Transcript:
    source: Path
    duration: float
    language: str
    model: str
    device: str
    compute: str
    segments: list[Segment]
    speakers: int = 0
    dropped: int = 0
    elapsed: float = 0.0
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


@dataclass
class Options:
    model: str
    device: str
    compute: str
    language: str = config.WHISPER_LANGUAGE
    prompt: str | None = config.WHISPER_PROMPT
    beam_size: int = config.WHISPER_BEAM_SIZE
    condition_on_previous_text: bool = False
    batch: bool = False
    batch_size: int = config.ASR_BATCH_SIZE
    block_s: float = 600.0
    word_timestamps: bool = True
    diarize: bool = True
    speakers: int | None = None

    def signature(self) -> str:
        """Отпечаток настроек, ВЛИЯЮЩИХ НА ТЕКСТ. Чекпойнт с другим отпечатком не подхватывается:
        доклеить к расшифровке `small` хвост от `large-v3` — получить документ, который нигде не
        соответствует сам себе. Диаризация сюда не входит намеренно: она применяется поверх
        готовых сегментов, и повтор с другим `--speakers` не должен заново гонять ASR."""
        payload = json.dumps({
            "model": self.model, "device": self.device, "compute": self.compute,
            "language": self.language, "prompt": self.prompt, "beam": self.beam_size,
            # В батчевом режиме `--context` на текст не влияет ВООБЩЕ: BatchedInferencePipeline
            # выставляет condition_on_previous_text=False жёстко (faster_whisper 1.1.0,
            # transcribe.py:505). В подписи он тогда только вредит — снятие или добавление
            # флага обесценивало бы валидный чекпойнт трёхчасовой работы без единого различия
            # в результате.
            "context": self.condition_on_previous_text and not self.batch,
            "batch": self.batch,
            "block_s": self.block_s, "words": self.word_timestamps,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def work_key(src: Path) -> str:
    """Идентификатор исходника для имён рабочих файлов: путь + размер + время правки.
    Изменился файл — изменился ключ, и старый кэш/чекпойнт не подхватится."""
    st = src.stat()
    raw = f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def work_paths(work_dir: Path, src: Path) -> tuple[Path, Path]:
    """(файл декодированного PCM, файл чекпойнта) для записи `src`."""
    key = work_key(src)
    stem = _safe_stem(src.stem)
    return work_dir / f"{stem}.{key}.pcm", work_dir / f"{stem}.{key}.json"


def cleanup_work(work_dir: Path, src: Path) -> None:
    """Убирает кэш PCM и чекпойнт после успешной записи результата."""
    for p in work_paths(work_dir, src):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        next(work_dir.iterdir())
    except StopIteration:
        work_dir.rmdir()            # каталог пуст — не оставляем мусор рядом с записями
    except OSError:
        pass


def _safe_stem(stem: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in stem)
    return keep.strip()[:60] or "audio"


class BatchTranscriber:
    """Модель (и диаризатор) живут между файлами: загрузка large-v3 — это секунды и гигабайты,
    платить ими за каждый файл в пакете незачем."""

    def __init__(self, opts: Options, reporter: Reporter | None = None) -> None:
        self.opts = opts
        self.reporter = reporter or Reporter()
        self._provider = None
        self._diarizer = None

    # --- модель ---------------------------------------------------------------

    @property
    def provider(self):
        if self._provider is None:
            from ..asr.faster_whisper_provider import FasterWhisperASR
            self._provider = FasterWhisperASR(
                self.opts.model, self.opts.device, self.opts.compute)
            # Прогрев здесь не про скорость, а про то, ЧТОБЫ ОТКАЗ GPU ВЫЯСНИЛСЯ СЕЙЧАС.
            # Типовой отказ (не найдены cuBLAS/cuDNN) проявляется не при загрузке весов, а при
            # первом декодировании — то есть иначе он вылезет после того, как мы час потратим
            # на декодирование звука.
            self._provider.warmup()
        return self._provider

    def _decoder(self):
        if not self.opts.batch:
            return self.provider.model
        # Обёртку берём у провайдера, а не собираем свою: там та же ленивая инициализация плюс
        # предупреждение о том, что в батчевом режиме отключаются
        # hallucination_silence_threshold и лестница температур.
        return self.provider._batched_pipeline()

    # --- один файл ------------------------------------------------------------

    def transcribe_file(self, src: Path, work_dir: Path, *, resume: bool = True) -> Transcript:
        pcm_path, ck_path = work_paths(work_dir, src)
        started = time.monotonic()
        state = _load_checkpoint(ck_path, src, self.opts) if resume else None

        self._ensure_pcm(src, pcm_path)
        pcm = open_pcm(pcm_path)
        sr = config.SAMPLE_RATE
        try:
            total_s = len(pcm) / sr
            if state is not None and state.get("samples") != len(pcm):
                state = None        # кэш и чекпойнт разошлись — пересчитать дешевле, чем спорить
            blocks = ([tuple(b) for b in state["blocks"]] if state
                      else plan_blocks(pcm, sr, target_s=self.opts.block_s))
            segments = ([Segment.from_dict(d) for d in state["segments"]] if state else [])
            stats = {"dropped": state["dropped"] if state else 0}
            done = state["done"] if state else 0
            resume_s = blocks[done][0] / sr if done < len(blocks) else total_s
            if 0 < done < len(blocks):
                self.reporter.note(f"продолжаю с {hms(resume_s)} "
                                   f"(готово блоков {done} из {len(blocks)})")

            self.reporter.stage("расшифровка", done_s=resume_s)
            self.reporter.progress(resume_s, total_s)
            for i in range(done, len(blocks)):
                a, b = blocks[i]
                audio = to_float32(pcm[a:b])
                for seg in self._decode_block(audio, a / sr, stats):
                    segments.append(seg)
                    self.reporter.progress(seg.end, total_s)
                del audio
                self.reporter.progress(b / sr, total_s)
                _save_checkpoint(ck_path, src, self.opts, samples=len(pcm), blocks=blocks,
                                 done=i + 1, segments=segments, dropped=stats["dropped"])
            self.reporter.progress(total_s, total_s)

            if not segments:
                self.reporter.note("речь не распознана — проверьте язык (--language) и то, "
                                   "что в записи действительно есть голос")
            speakers = 0
            if self.opts.diarize and segments:
                speakers = self._diarize(segments, pcm, sr)
                _save_checkpoint(ck_path, src, self.opts, samples=len(pcm), blocks=blocks,
                                 done=len(blocks), segments=segments, dropped=stats["dropped"])
            if not speakers:
                # Метки могли прийти из чекпойнта (диаризация прошла в прошлый раз) — тогда
                # шапка .txt должна говорить о голосах, даже если сейчас мы их не считали.
                speakers = len({s.speaker for s in segments if s.speaker})
        finally:
            close_pcm(pcm)

        return Transcript(
            source=src, duration=total_s, language=self.opts.language,
            model=self.opts.model, device=self.opts.device, compute=self.opts.compute,
            segments=segments, speakers=speakers, dropped=stats["dropped"],
            elapsed=time.monotonic() - started,
        )

    def _ensure_pcm(self, src: Path, pcm_path: Path) -> int:
        if pcm_path.exists() and pcm_path.stat().st_size > 0:
            return pcm_path.stat().st_size // 2
        hint = probe_duration(src) or 0.0
        self.reporter.stage("декодирование")
        samples = decode_to_pcm(
            src, pcm_path,
            on_progress=(lambda s: self.reporter.progress(s, hint)) if hint else None)
        if samples <= 0:
            raise ValueError(f"не удалось декодировать звук: {src.name}")
        return samples

    def _decode_block(self, audio: np.ndarray, offset: float, stats: dict):
        """Один блок -> сегменты с абсолютными (по всей записи) тайм-кодами."""
        kwargs = dict(
            language=self.opts.language,
            beam_size=self.opts.beam_size,
            word_timestamps=self.opts.word_timestamps,
            condition_on_previous_text=self.opts.condition_on_previous_text,
            initial_prompt=self.opts.prompt,
            # VAD с ДЕФОЛТНЫМИ порогами библиотеки, а не с live-настройкой 300 мс: там короткий
            # порог нужен, чтобы не ждать конца паузы, здесь ждать некого, а склейка речи через
            # каждую полусекундную паузу только рвёт интонацию на стыках.
            vad_filter=True,
            temperature=config.WHISPER_TEMPERATURE_FALLBACK,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=config.WHISPER_HALLUCINATION_SILENCE_S,
        )
        decoder = self._decoder()
        if self.opts.batch:
            segments, _info = decoder.transcribe(
                audio, batch_size=self.opts.batch_size, **kwargs)
        else:
            segments, _info = decoder.transcribe(audio, **kwargs)

        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            reason = hallucination_reason(text, self.opts.prompt)
            if reason:
                stats["dropped"] += 1
                # Счётчик говорит СКОЛЬКО, но не что именно: на трёхчасовой записи проверить
                # отбраковку по одному числу нельзя. Через reporter, а не print — иначе строка
                # затрёт живой прогресс.
                self.reporter.note(dropped_note(hms(offset + float(seg.start)), reason, text))
                continue
            yield Segment(
                start=offset + float(seg.start),
                end=offset + float(seg.end),
                text=text,
                words=[{"text": w.word, "start": offset + float(w.start),
                        "end": offset + float(w.end), "prob": float(w.probability or 1.0)}
                       for w in (seg.words or [])],
            )

    # --- голоса ---------------------------------------------------------------

    def _diarize(self, segments: list[Segment], pcm: np.ndarray, sr: int) -> int:
        """Расставляет `speaker` по сегментам (ECAPA + кластеризация). Возвращает число голосов.

        Отказ диаризации НЕ ДОЛЖЕН стоить расшифровки: torch/speechbrain — тяжёлые зависимости,
        модель ECAPA тянется из сети при первом запуске, и если что-то из этого не сложилось,
        правильный исход — текст без разметки по голосам и предупреждение, а не пустой результат
        после часа работы.
        """
        try:
            if self._diarizer is None:
                from ..finalize.diarizer import Diarizer
                self._diarizer = Diarizer()
            self.reporter.stage("голоса")
            total_s = segments[-1].end
            embeddings, idx = [], []
            for i, seg in enumerate(segments):
                a, b = _embed_window(seg, len(pcm), sr)
                emb = self._diarizer.embed(to_float32(pcm[a:b]))
                if emb is not None:
                    embeddings.append(emb)
                    idx.append(i)
                self.reporter.progress(seg.end, total_s)
            if not embeddings:
                return 0
            labels = self._diarizer.cluster(embeddings, self.opts.speakers)
        except Exception as exc:
            self.reporter.note(f"голоса развести не удалось ({exc}); текст будет без разметки")
            return 0

        order: dict[int, int] = {}          # нумеруем голоса по первому появлению, а не по
        for lab in labels:                  # произвольному номеру кластера
            order.setdefault(int(lab), len(order) + 1)
        for pos, lab in zip(idx, labels):
            segments[pos].speaker = f"Голос-{order[int(lab)]}"
        _fill_unlabeled(segments)
        return len(order)


def _embed_window(seg: Segment, total_samples: int, sr: int) -> tuple[int, int]:
    """Окно под эмбеддинг: центр реплики, не длиннее DIARIZE_WINDOW_S."""
    start, end = seg.start, seg.end
    if end - start > DIARIZE_WINDOW_S:
        mid = (start + end) / 2
        start, end = mid - DIARIZE_WINDOW_S / 2, mid + DIARIZE_WINDOW_S / 2
    a = max(0, int(start * sr))
    b = min(total_samples, int(end * sr))
    return a, b


def _fill_unlabeled(segments: list[Segment]) -> None:
    """Короткие реплики («да», «угу») эмбеддинга не дают — ECAPA на 0.3 с ненадёжен. Отдаём их
    ближайшему по времени размеченному соседу: в тексте лучше пусть будет вероятный говорящий,
    чем реплика без говорящего посреди диалога."""
    known = [i for i, s in enumerate(segments) if s.speaker]
    if not known:
        return
    for i, seg in enumerate(segments):
        if seg.speaker:
            continue
        nearest = min(known, key=lambda j: abs(segments[j].start - seg.start))
        seg.speaker = segments[nearest].speaker


# --- чекпойнт -----------------------------------------------------------------

def _save_checkpoint(path: Path, src: Path, opts: Options, *, samples: int,
                     blocks: list[tuple[int, int]], done: int, segments: list[Segment],
                     dropped: int) -> None:
    st = src.stat()
    payload = {
        "version": CHECKPOINT_VERSION,
        "source": str(src.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
        "signature": opts.signature(), "samples": samples,
        "blocks": [list(b) for b in blocks], "done": done, "dropped": dropped,
        "segments": [s.as_dict() for s in segments],
    }
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)           # атомарно: оборванная запись не должна портить прогресс


def _load_checkpoint(path: Path, src: Path, opts: Options) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    st = src.stat()
    ok = (data.get("version") == CHECKPOINT_VERSION
          and data.get("signature") == opts.signature()
          and data.get("size") == st.st_size
          and data.get("mtime_ns") == st.st_mtime_ns
          and data.get("blocks") and 0 < data.get("done", 0) <= len(data["blocks"]))
    return data if ok else None
