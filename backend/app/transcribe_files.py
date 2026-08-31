"""Расшифровка готовых аудиозаписей в текстовые файлы (пакетный офлайн-режим).

Отдельный клиент, не связанный с сервером и Electron-интерфейсом: сервер нужен для живой записи
опроса, а здесь на входе уже готовые файлы — длинные (часы) и, как правило, m4a с диктофона.
Формат входа значения не имеет: PyAV несёт в себе ffmpeg, поэтому m4a/aac, mp3, opus, wav,
дорожка из mp4 читаются одинаково и ничего доустанавливать не нужно.

    py -3.11 -m app.transcribe_files E:\\rec\\допрос.m4a
    py -3.11 -m app.transcribe_files E:\\rec\\*.m4a -o E:\\txt
    py -3.11 -m app.transcribe_files E:\\rec -r --speakers 2       # вся папка, ровно два голоса
    py -3.11 -m app.transcribe_files запись.m4a --format txt,srt,json

Рядом с результатом создаётся служебный каталог `.transcribe-work`: в нём лежат декодированный
звук и чекпойнт. Прерванная на середине трёхчасовая запись при повторном запуске продолжится
с того же места, а не начнётся заново; после успешной записи результата каталог убирается сам.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

# Консоль Windows может быть cp1251 — иначе кириллица в прогрессе превратится в исключение.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import config, device as device_mod
from .batch.pipeline import BatchTranscriber, Options, cleanup_work
from .batch.progress import ConsoleReporter, hms, human
from .batch import writers

# Что считать записью при обходе папки. Явно указанный файл берётся с любым расширением.
DEFAULT_EXTS = ("m4a", "mp3", "wav", "mp4", "m4b", "aac", "ogg", "opus", "flac", "wma", "webm")
# На CPU эти модели декодируют МЕДЛЕННЕЕ реального времени (0.5-0.9x, замер в backend/README.md):
# двухчасовая запись займёт больше двух часов. Не запрещаем — предупреждаем.
CPU_SLOW_MODELS = ("medium", "turbo", "large-v1", "large-v2", "large-v3", "large")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    sources = _expand_inputs(args.paths, args.ext.split(","), args.recursive)
    if not sources:
        print("нечего расшифровывать: не найдено ни одного файла", file=sys.stderr)
        return 2

    device = args.device if args.device != "auto" else device_mod.resolve_device("auto")
    model, compute = _defaults_for(device)
    opts = Options(
        model=args.model or model,
        device=device,
        compute=args.compute or compute,
        language=args.language,
        prompt=None if args.no_prompt else (args.prompt or config.WHISPER_PROMPT),
        beam_size=args.beam,
        condition_on_previous_text=args.context,
        batch=args.batch,
        block_s=args.block_min * 60.0,
        diarize=not args.no_diarize,
        speakers=args.speakers,
    )
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    unknown = [f for f in formats if f not in ("txt", "srt", "json")]
    if unknown:
        print(f"неизвестный формат: {', '.join(unknown)}", file=sys.stderr)
        return 2

    reporter = ConsoleReporter()
    print(f"устройство {opts.device}/{opts.compute}, модель {opts.model}, "
          f"файлов {len(sources)}", file=sys.stderr)
    if opts.device == "cpu" and any(opts.model.startswith(m) for m in CPU_SLOW_MODELS):
        print(f"  ВНИМАНИЕ: {opts.model} на CPU идёт медленнее реального времени — расшифровка "
              f"займёт больше, чем длится запись. Быстрее: --model small (или GPU).",
              file=sys.stderr)
    if opts.diarize and not _diarize_available():
        print("  ВНИМАНИЕ: разбивка по голосам недоступна — не установлены torch / speechbrain / "
              "scikit-learn (pip install -r requirements.txt). Текст будет без разметки по "
              "голосам; --no-diarize убирает это предупреждение.", file=sys.stderr)
    if opts.batch:
        print("  ВНИМАНИЕ: --batch ускоряет вдвое, но отключает лестницу температур и "
              "hallucination_silence_threshold (ограничение faster-whisper) — защита от "
              "галлюцинаций слабее", file=sys.stderr)
        if opts.condition_on_previous_text:
            print("  ВНИМАНИЕ: --context вместе с --batch не действует — батчевый проход "
                  "выставляет condition_on_previous_text=False жёстко", file=sys.stderr)

    transcriber = BatchTranscriber(opts, reporter)
    taken: set[Path] = set()
    done = skipped = failed = 0
    audio_s = 0.0
    started = time.monotonic()

    for i, src in enumerate(sources, start=1):
        out_dir = Path(args.out) if args.out else src.parent
        targets = _targets(out_dir, src, formats, taken)
        if not args.overwrite and all(p.exists() for p in targets.values()):
            print(f"[{i}/{len(sources)}] {src.name}: уже расшифрован, пропускаю "
                  f"(--overwrite — перезаписать)", file=sys.stderr)
            skipped += 1
            continue

        work_dir = Path(args.work_dir) if args.work_dir else out_dir / ".transcribe-work"
        reporter.file(i, len(sources), src.name, _duration_hint(src))
        try:
            doc = transcriber.transcribe_file(src, work_dir, resume=not args.no_resume)
        except KeyboardInterrupt:
            print("\nпрервано. Прогресс сохранён — повторный запуск продолжит с того же места.",
                  file=sys.stderr)
            return 130
        except Exception as exc:
            reporter.note(f"ОШИБКА: {exc}")
            failed += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        _write(doc, targets, timestamps=not args.no_timestamps, header=not args.no_header)
        if not args.keep_work:
            cleanup_work(work_dir, src)

        done += 1
        audio_s += doc.duration
        speed = doc.duration / max(1e-6, doc.elapsed)
        extra = f", голосов {doc.speakers}" if doc.speakers else ""
        extra += f", отброшено галлюцинаций {doc.dropped}" if doc.dropped else ""
        reporter.done(f"готово за {human(doc.elapsed)} ({speed:.1f}x), "
                      f"реплик {len(doc.segments)}{extra}")
        for path in targets.values():
            print(path, flush=True)

    elapsed = time.monotonic() - started
    print(f"итого: расшифровано {done}, пропущено {skipped}, с ошибкой {failed}; "
          f"звука {hms(audio_s)} за {human(elapsed)}", file=sys.stderr)
    return 1 if failed else 0


def _write(doc, targets: dict[str, Path], *, timestamps: bool, header: bool) -> None:
    for fmt, path in targets.items():
        if fmt == "txt":
            writers.write_txt(path, doc, timestamps=timestamps, header=header)
        elif fmt == "srt":
            writers.write_srt(path, doc)
        else:
            writers.write_json(path, doc)


def _targets(out_dir: Path, src: Path, formats: list[str], taken: set[Path]) -> dict[str, Path]:
    """Пути результатов. Если в один каталог собираются одноимённые записи из разных папок,
    второй получает суффикс — молча затирать чужую расшифровку нельзя."""
    stem = src.stem
    suffix = 1
    while any((out_dir / f"{stem}.{fmt}") in taken for fmt in formats):
        suffix += 1
        stem = f"{src.stem}_{suffix}"
    targets = {fmt: out_dir / f"{stem}.{fmt}" for fmt in formats}
    taken.update(targets.values())
    return targets


def _diarize_available() -> bool:
    """Проверка зависимостей диаризации ДО начала работы.

    Сама проверка тривиальна, но момент важен: диаризация идёт последним шагом, и без неё
    пользователь узнал бы об отсутствии torch через час после запуска — по завершении
    расшифровки. `find_spec` вместо импорта: импорт torch стоит несколько секунд.
    """
    from importlib.util import find_spec
    try:
        return all(find_spec(m) is not None for m in ("torch", "speechbrain", "sklearn"))
    except (ImportError, ValueError):
        return False


def _duration_hint(src: Path) -> float:
    from .batch.decode import probe_duration
    return probe_duration(src) or 0.0


def _defaults_for(device: str) -> tuple[str, str]:
    """Дефолты модели и точности для УКАЗАННОГО устройства.

    `config` считает их для того устройства, которое определилось при импорте, — а здесь его
    можно задать флагом. Без этой поправки `--device cuda` на машине, где автоопределение дало
    cpu, тихо взяло бы `small`/`int8`: работать будет, но это не тот режим, за которым идут на GPU.
    """
    if device.startswith("cuda"):
        return (config.WHISPER_MODEL if config.IS_GPU else "large-v3",
                config.WHISPER_COMPUTE if config.IS_GPU else "float16")
    return (config.WHISPER_MODEL if not config.IS_GPU else config.WHISPER_MODEL_CPU_FALLBACK,
            config.WHISPER_COMPUTE if not config.IS_GPU else "int8")


def _expand_inputs(paths: list[str], exts: list[str], recursive: bool) -> list[Path]:
    """Файлы, папки и маски -> список записей.

    Маски раскрываются здесь, а не оболочкой: PowerShell и cmd передают `*.m4a` программе как
    есть, поэтому без этого самый очевидный вызов из README не работал бы вовсе. `**` в маске
    раскрывается вглубь только с `-r` — рекурсию просят флагом, а не написанием маски.
    """
    exts = {("." + e.strip().lower().lstrip(".")) for e in exts if e.strip()}
    found: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen and p.is_file():
            seen.add(rp)
            found.append(p)

    for raw in paths:
        p = Path(raw)
        # Существующий путь — всегда файл/папка, даже если в имени есть `[` или `*`. Иначе
        # `E:\rec\[2024-03] допрос.m4a` (обычное имя с диктофона) уходил в glob, `[2024-03]`
        # читался как класс символов, совпадений не находилось — и запись исчезала МОЛЧА,
        # потому что ветка glob уходила по `continue` мимо сообщения «не найдено».
        if not p.exists() and any(ch in raw for ch in "*?["):
            hits = sorted(glob.glob(raw, recursive=recursive))
            if not hits:
                print(f"не найдено по маске: {raw}", file=sys.stderr)
            for hit in hits:
                add(Path(hit))
            continue
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for hit in sorted(it):
                if hit.is_file() and hit.suffix.lower() in exts:
                    add(hit)
        elif p.is_file():
            add(p)                    # явно указанный файл берём с любым расширением
        else:
            print(f"не найдено: {raw}", file=sys.stderr)
    return found


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="py -3.11 -m app.transcribe_files",
        description="Расшифровка аудиозаписей в текстовые файлы (офлайн, локально).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Прерванная расшифровка продолжается при повторном запуске с того же места.",
    )
    p.add_argument("paths", nargs="+", metavar="ЗАПИСЬ",
                   help="файлы, папки или маски (E:\\rec\\*.m4a)")
    p.add_argument("-o", "--out", metavar="ПАПКА",
                   help="куда класть результат (по умолчанию — рядом с записью)")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="обходить вложенные папки")
    p.add_argument("--ext", default=",".join(DEFAULT_EXTS), metavar="СПИСОК",
                   help="расширения при обходе папки (по умолчанию: %(default)s)")
    p.add_argument("--format", default="txt", metavar="СПИСОК",
                   help="что писать: txt, srt, json через запятую (по умолчанию: txt)")

    g = p.add_argument_group("распознавание")
    g.add_argument("--model", help="модель Whisper (по умолчанию: large-v3 на GPU, small на CPU)")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="устройство (по умолчанию: auto)")
    g.add_argument("--compute", help="точность: float16 / int8 / int8_float16")
    g.add_argument("--language", default=config.WHISPER_LANGUAGE, help="язык записи")
    g.add_argument("--beam", type=int, default=config.WHISPER_BEAM_SIZE,
                   help="ширина луча (по умолчанию: %(default)s)")
    g.add_argument("--prompt", help="подсказка домена (терминология)")
    g.add_argument("--no-prompt", action="store_true",
                   help="без подсказки домена — если её эхо мешает больше, чем помогает")
    g.add_argument("--context", action="store_true",
                   help="передавать модели контекст предыдущего окна: связнее текст, но выше "
                        "риск самоподдерживающихся галлюцинаций на длинной записи")
    g.add_argument("--batch", action="store_true",
                   help="батчевый проход: вдвое быстрее ценой ослабленной защиты от галлюцинаций")
    g.add_argument("--block-min", type=float, default=10.0, metavar="МИН",
                   help="длина блока обработки в минутах (по умолчанию: %(default)s)")

    g = p.add_argument_group("голоса")
    g.add_argument("--speakers", type=int, metavar="N",
                   help="сколько голосов на записи (по умолчанию — определить автоматически)")
    g.add_argument("--no-diarize", action="store_true",
                   help="не разводить по голосам (быстрее, текст без разметки)")

    g = p.add_argument_group("оформление и служебное")
    g.add_argument("--no-timestamps", action="store_true", help="без тайм-кодов в .txt")
    g.add_argument("--no-header", action="store_true", help="без шапки в .txt")
    g.add_argument("--overwrite", action="store_true", help="перезаписывать готовые расшифровки")
    g.add_argument("--no-resume", action="store_true",
                   help="не подхватывать прерванный прогресс, считать заново")
    g.add_argument("--work-dir", metavar="ПАПКА",
                   help="где держать кэш звука и чекпойнт (по умолчанию: .transcribe-work "
                        "рядом с результатом)")
    g.add_argument("--keep-work", action="store_true",
                   help="не удалять служебные файлы после расшифровки")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
