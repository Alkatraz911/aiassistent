"""Проверка пакетной расшифровки (`app/transcribe_files.py`) без сети и без ASR-модели.

Проверяется то, что ломается молча и дорого — то есть выясняется только на трёхчасовой записи
и уже после часа работы:

  • декодирование m4a (сжатый вход, а не WAV — именно им пользуются),
  • точки реза блоков попадают в тишину, а не в середину слова,
  • чекпойнт: возобновление подхватывается, а под ЧУЖИМИ настройками — нет,
  • сборка .txt: реплики склеиваются, длинный монолог всё равно получает тайм-коды,
  • раскрытие путей: маска `*.m4a` (оболочка Windows её не раскрывает) и обход папки.

Сама модель здесь не запускается — это отдельный прогон на реальном файле:
    py -3.11 -m app.transcribe_files запись.m4a --model tiny --no-diarize

Запуск:  py -3.11 -m app.test_batch
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import config
from .batch import writers
from .batch.blocks import plan_blocks
from .batch.decode import (close_pcm, decode_to_pcm, open_pcm, probe_duration,
                          to_float32)
from .batch.pipeline import (BatchTranscriber, Options, Segment, Transcript,
                             _load_checkpoint, _save_checkpoint, work_paths)
from .transcribe_files import _expand_inputs

SR = config.SAMPLE_RATE
# «Речь» и паузы: четыре всплеска с заведомо известными границами тишины между ними.
SPEECH_SPANS = [(1, 5), (8, 13), (17, 22), (25, 29)]
TOTAL_S = 30


def _signal() -> np.ndarray:
    t = np.arange(TOTAL_S * SR) / SR
    sig = np.zeros(TOTAL_S * SR, dtype=np.float32)
    for a, b in SPEECH_SPANS:
        sl = slice(int(a * SR), int(b * SR))
        sig[sl] = 0.3 * np.sin(2 * np.pi * 180 * t[sl]) + 0.15 * np.sin(2 * np.pi * 360 * t[sl])
    return (sig * 32767).astype(np.int16)


def _encode_m4a(pcm: np.ndarray, path: Path) -> None:
    """Кодирует int16-моно в m4a (AAC) средствами PyAV — тем же ffmpeg, что и на чтении."""
    import av

    with av.open(str(path), "w") as out:
        stream = out.add_stream("aac", rate=SR, layout="mono")
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=SR)
        fifo = av.audio.fifo.AudioFifo()
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = SR
        frame.pts = 0
        for resampled in resampler.resample(frame):
            fifo.write(resampled)
        size = stream.codec_context.frame_size or 1024   # AAC кодирует кадрами по 1024
        pts = 0
        while True:
            chunk = fifo.read(size)
            if chunk is None:
                break
            chunk.pts = pts
            pts += chunk.samples
            for packet in stream.encode(chunk):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def test_decode(tmp: Path) -> Path:
    src = tmp / "проба.m4a"
    _encode_m4a(_signal(), src)
    assert src.stat().st_size > 0

    hint = probe_duration(src)
    assert hint is not None and abs(hint - TOTAL_S) < 1.0, f"длительность из контейнера: {hint}"

    dst = tmp / "проба.pcm"
    seen: list[float] = []
    n = decode_to_pcm(src, dst, on_progress=seen.append)
    assert abs(n / SR - TOTAL_S) < 1.0, f"декодировано {n / SR:.2f}с вместо {TOTAL_S}"
    # Прогресс отдаётся по группам кадров (~31с звука), поэтому на 30-секундной пробе он
    # приходит один раз — важно, что последнее значение совпадает с реально записанным.
    assert seen and abs(seen[-1] - n / SR) < 0.05, f"прогресс декодирования: {seen}"

    pcm = open_pcm(dst)
    assert pcm.dtype == np.dtype("<i2") and len(pcm) == n
    # Структура сигнала пережила кодирование: во всплесках громко, в паузах тихо.
    loud = np.abs(to_float32(pcm[int(2 * SR):int(4 * SR)])).mean()
    quiet = np.abs(to_float32(pcm[int(6 * SR):int(7 * SR)])).mean()
    assert loud > 10 * quiet, f"всплеск {loud:.4f} против паузы {quiet:.4f}"
    close_pcm(pcm)
    print(f"OK  декодирование m4a: {n / SR:.2f}с, всплеск {loud:.4f} против паузы {quiet:.5f}")
    return dst


def test_blocks(pcm_path: Path) -> None:
    pcm = open_pcm(pcm_path)
    # `search_s` задан явно: по умолчанию окно поиска считается от длины блока (0.15), и на
    # игрушечных блоках по 10с это 1.5с — уже самой паузы. На боевых блоках по 10 минут
    # умолчание даёт 45с, чего с запасом хватает на любую живую речь.
    blocks = plan_blocks(pcm, SR, target_s=10.0, search_s=4.0)
    close_pcm(pcm)
    assert len(blocks) >= 3, f"ожидались блоки, получен {blocks}"
    assert blocks[0][0] == 0 and blocks[-1][1] == pcm_path.stat().st_size // 2,         "блоки не покрывают запись"
    for (_, end), (start, _) in zip(blocks, blocks[1:]):
        assert end == start, "между блоками образовалась дыра"

    silences = [(SPEECH_SPANS[i][1], SPEECH_SPANS[i + 1][0]) for i in range(len(SPEECH_SPANS) - 1)]
    for _, cut in blocks[:-1]:
        at = cut / SR
        assert any(a <= at <= b for a, b in silences), \
            f"рез на {at:.2f}с пришёлся не в паузу (паузы: {silences})"
    print(f"OK  нарезка: {len(blocks)} блока, все резы в паузах "
          f"({', '.join(f'{c / SR:.1f}с' for _, c in blocks[:-1])})")


def test_checkpoint(tmp: Path, pcm_path: Path) -> None:
    src = tmp / "проба.m4a"
    opts = Options(model="small", device="cpu", compute="int8")
    _, ck = work_paths(tmp, src)
    segs = [Segment(start=0.0, end=1.0, text="раз"), Segment(start=1.5, end=2.0, text="два")]
    _save_checkpoint(ck, src, opts, samples=1000, blocks=[(0, 500), (500, 1000)],
                     done=1, segments=segs, dropped=3)

    state = _load_checkpoint(ck, src, opts)
    assert state is not None and state["done"] == 1 and state["dropped"] == 3
    assert [s["text"] for s in state["segments"]] == ["раз", "два"]

    # Другая модель — другой отпечаток: доклеивать хвост от другой модели нельзя.
    other = Options(model="large-v3", device="cpu", compute="int8")
    assert _load_checkpoint(ck, src, other) is None, "чекпойнт подхватился под чужими настройками"

    # Испорченный файл не должен ронять расшифровку — просто считаем заново.
    ck.write_text("{не json", encoding="utf-8")
    assert _load_checkpoint(ck, src, opts) is None
    print("OK  чекпойнт: возобновление, отсечка по настройкам, устойчивость к битому файлу")


def test_writers(tmp: Path) -> None:
    segs = [
        Segment(start=1.0, end=3.0, text="Представьтесь, пожалуйста.", speaker="Голос-1"),
        Segment(start=3.2, end=4.0, text="Назовите дату рождения.", speaker="Голос-1"),
        Segment(start=4.5, end=6.0, text="Иванов Иван Иванович.", speaker="Голос-2"),
        Segment(start=400.0, end=402.0, text="Продолжим.", speaker="Голос-2"),
    ]
    doc = Transcript(source=tmp / "проба.m4a", duration=420.0, language="ru", model="small",
                     device="cpu", compute="int8", segments=segs, speakers=2)

    turns = writers.group_turns(segs)
    assert [t.speaker for t in turns] == ["Голос-1", "Голос-2", "Голос-2"], \
        "реплики одного голоса должны склеиваться, а разрыв в минуты — разрывать"
    assert turns[0].lines == ["Представьтесь, пожалуйста.", "Назовите дату рождения."]

    txt = tmp / "проба.txt"
    writers.write_txt(txt, doc)
    body = txt.read_text(encoding="utf-8-sig")
    assert body.startswith("проба.m4a"), "нет шапки"
    assert "голосов: 2" in body
    assert "[00:00:01] Голос-1" in body and "[00:06:40] Голос-2" in body, body
    assert txt.read_bytes().startswith(b"\xef\xbb\xbf"), "нет BOM — Word покажет кракозябры"

    writers.write_txt(txt, doc, timestamps=False, header=False)
    body = txt.read_text(encoding="utf-8-sig")
    assert body.startswith("Голос-1") and "[00:" not in body

    srt = tmp / "проба.srt"
    writers.write_srt(srt, doc)
    lines = srt.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "00:00:01,000 --> 00:00:03,000", lines[:3]
    assert lines[2].startswith("Голос-1: ")

    js = tmp / "проба.json"
    writers.write_json(js, doc)
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["speakers"] == 2 and len(data["segments"]) == 4
    print("OK  вывод: склейка реплик, тайм-коды, BOM в .txt, .srt и .json")


def test_long_turn_is_split() -> None:
    """Монолог без пауз не должен превращаться в один абзац без тайм-кодов."""
    segs = [Segment(start=i * 20.0, end=i * 20.0 + 19.0, text=f"фраза {i}", speaker="Голос-1")
            for i in range(30)]                      # 10 минут подряд одним голосом
    turns = writers.group_turns(segs)
    assert len(turns) >= 3, f"монолог остался одной репликой ({len(turns)})"
    assert all(t.end - t.start <= writers.TURN_MAX_S + 20 for t in turns)
    print(f"OK  длинный монолог разбит на {len(turns)} реплики с тайм-кодами")


class _FakeDiarizer:
    """Подмена ECAPA: та же форма ответа, но без torch.

    Нужна, чтобы проверить именно НАШУ часть диаризации — нумерацию голосов и раздачу меток
    коротким репликам. Настоящий `Diarizer` тянет torch, speechbrain и модель из сети, и
    прогонять его ради проверки нумерации незачем; сам он проверяется `app/test_diarize.py`.
    """

    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def embed(self, audio: np.ndarray):
        # Как в настоящем: слишком короткий кусок эмбеддинга не даёт.
        return None if audio.size < SR // 3 else np.zeros(4, dtype=np.float32)

    def cluster(self, embeddings, num_speakers):
        assert len(embeddings) == len(self.labels)
        return self.labels


def test_diarize_labels() -> None:
    segs = [
        Segment(start=0.0, end=4.0, text="Представьтесь."),
        Segment(start=4.2, end=4.4, text="Да."),            # коротко -> без эмбеддинга
        Segment(start=5.0, end=9.0, text="Иванов Иван."),
        Segment(start=9.0, end=13.0, text="Год рождения — 1980."),
    ]
    pcm = np.zeros(20 * SR, dtype=np.int16)
    t = BatchTranscriber(Options(model="tiny", device="cpu", compute="int8"))
    t._diarizer = _FakeDiarizer([1, 0, 0])          # первым говорит кластер 1

    speakers = t._diarize(segs, pcm, SR)
    assert speakers == 2, speakers
    # Голоса нумеруются по первому появлению, а не по номеру кластера.
    assert [s.speaker for s in segs] == ["Голос-1", "Голос-2", "Голос-2", "Голос-2"],         [s.speaker for s in segs]
    print("OK  голоса: нумерация по первому появлению, короткая реплика получила соседа")


def test_diarize_failure_keeps_text() -> None:
    """Отказ диаризации (нет torch, не скачалась модель) не должен стоить расшифровки."""
    class _Broken:
        def embed(self, audio):
            raise RuntimeError("No module named 'speechbrain'")

    segs = [Segment(start=0.0, end=4.0, text="Текст на месте.")]
    t = BatchTranscriber(Options(model="tiny", device="cpu", compute="int8"))
    t._diarizer = _Broken()
    assert t._diarize(segs, np.zeros(5 * SR, dtype=np.int16), SR) == 0
    assert segs[0].text == "Текст на месте." and segs[0].speaker is None
    print("OK  отказ диаризации: текст сохранён, разметки нет")


def test_expand_inputs(tmp: Path) -> None:
    folder = tmp / "записи"
    (folder / "вложенная").mkdir(parents=True)
    for name in ("a.m4a", "b.mp3", "заметка.txt"):
        (folder / name).write_bytes(b"x")
    (folder / "вложенная" / "c.m4a").write_bytes(b"x")

    exts = ["m4a", "mp3"]
    flat = _expand_inputs([str(folder)], exts, recursive=False)
    assert [p.name for p in flat] == ["a.m4a", "b.mp3"], flat

    deep = _expand_inputs([str(folder)], exts, recursive=True)
    assert {p.name for p in deep} == {"a.m4a", "b.mp3", "c.m4a"}

    # Маску раскрываем сами: PowerShell и cmd передают её программе как есть.
    masked = _expand_inputs([str(folder / "*.m4a")], exts, recursive=False)
    assert [p.name for p in masked] == ["a.m4a"], masked

    # Явно указанный файл берётся с любым расширением, повтор не дублируется.
    named = _expand_inputs([str(folder / "заметка.txt"), str(folder / "a.m4a"),
                            str(folder / "a.m4a")], exts, recursive=False)
    assert [p.name for p in named] == ["заметка.txt", "a.m4a"], named
    print("OK  пути: папка, рекурсия, маска, явный файл без дублей")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="batch-test-") as raw:
        tmp = Path(raw)
        pcm_path = test_decode(tmp)
        test_blocks(pcm_path)
        test_checkpoint(tmp, pcm_path)
        test_writers(tmp)
        test_long_turn_is_split()
        test_diarize_labels()
        test_diarize_failure_keeps_text()
        test_expand_inputs(tmp)
    print("\nвсе проверки пройдены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
