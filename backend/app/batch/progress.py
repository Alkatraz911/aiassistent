"""Отображение хода расшифровки в консоли.

Держится отдельно от пайплайна намеренно: расшифровку многочасовой записи нельзя запускать
«вслепую», но и печатать её ход в лог сервера или в файл — не то же самое, что рисовать
живую строку в терминале. Пайплайн знает только про интерфейс `Reporter`.
"""
from __future__ import annotations

import sys
import time


def hms(seconds: float) -> str:
    """Секунды -> ЧЧ:ММ:СС (для длинных записей часы нужны всегда)."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def human(seconds: float) -> str:
    """Короткая оценка оставшегося времени: «6м12с», «1ч04м»."""
    seconds = max(0, int(round(seconds)))
    if seconds >= 3600:
        return f"{seconds // 3600}ч{seconds % 3600 // 60:02d}м"
    if seconds >= 60:
        return f"{seconds // 60}м{seconds % 60:02d}с"
    return f"{seconds}с"


class Reporter:
    """Пустая реализация: пайплайн работает молча (например, при вызове из кода)."""

    def file(self, index: int, total: int, name: str, duration: float) -> None: ...
    def stage(self, name: str, done_s: float = 0.0) -> None: ...
    def progress(self, done_s: float, total_s: float) -> None: ...
    def note(self, text: str) -> None: ...
    def done(self, text: str) -> None: ...


class ConsoleReporter(Reporter):
    """Однострочный прогресс в stderr (stdout остаётся чистым — его можно перенаправить).

    В не-терминал (перенаправленный вывод, CI, запуск из планировщика) `\r`-строка превратилась
    бы в мусор из тысяч строк, поэтому там печатаются только редкие вехи.
    """

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._prefix = ""
        self._stage = ""
        self._base = 0.0
        self._started = 0.0
        self._last_draw = 0.0
        self._line_len = 0
        self._last_decile = -1

    def file(self, index: int, total: int, name: str, duration: float) -> None:
        self._prefix = f"[{index}/{total}] {name}"
        self._clear()
        dur = f"  {hms(duration)}" if duration else ""
        print(f"{self._prefix}{dur}", file=self.stream, flush=True)

    def stage(self, name: str, done_s: float = 0.0) -> None:
        """`done_s` — сколько секунд записи уже готово ДО этого запуска (возобновление):
        скорость и оценка остатка должны считаться по работе ЭТОГО прогона, иначе после
        возобновления с середины они показывают фантастические числа."""
        self._clear()
        self._stage = name
        self._base = done_s
        self._started = time.monotonic()
        self._last_draw = 0.0
        self._last_decile = -1

    def progress(self, done_s: float, total_s: float) -> None:
        now = time.monotonic()
        if self.tty and now - self._last_draw < 0.25:
            return                      # чаще 4 раз в секунду перерисовывать нечего
        self._last_draw = now
        elapsed = max(1e-6, now - self._started)
        speed = max(0.0, done_s - self._base) / elapsed
        share = (done_s / total_s) if total_s > 0 else 0.0
        eta = (total_s - done_s) / speed if speed > 0 else 0.0
        # Скорость выше сотен «x» бывает только там, где мерить нечего (мгновенное
        # декодирование короткого файла) — печатать «19968000.0x» незачем.
        rate = f"{speed:.1f}x" if speed < 999 else ">999x"
        text = (f"  {self._stage}: {share * 100:5.1f}%  {hms(done_s)} / {hms(total_s)}"
                f"   {rate}  ост. ~{human(eta)}")
        if self.tty:
            self._write_line(text)
            return
        decile = int(share * 10)
        if decile > self._last_decile:      # не-терминал: одна строка на каждые 10%
            self._last_decile = decile
            print(text, file=self.stream, flush=True)

    def note(self, text: str) -> None:
        self._clear()
        print(f"  {text}", file=self.stream, flush=True)

    def done(self, text: str) -> None:
        self._clear()
        print(f"  {text}", file=self.stream, flush=True)

    def _write_line(self, text: str) -> None:
        pad = max(0, self._line_len - len(text))
        self.stream.write("\r" + text + " " * pad)
        self.stream.flush()
        self._line_len = len(text)

    def _clear(self) -> None:
        if self.tty and self._line_len:
            self.stream.write("\r" + " " * self._line_len + "\r")
            self.stream.flush()
            self._line_len = 0
