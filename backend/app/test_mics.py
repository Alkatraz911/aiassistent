"""Разведены ли два микрофона по каналам одного USB-входа?

Зачем. Основной режим («канал = спикер») держится на том, что каждый участник пишется своим
микрофоном. Когда оба микрофона воткнуты в ОДИН мини-USB адаптер, Windows показывает одно
устройство — и дальше возможны два принципиально разных случая, неотличимых на глаз:

  1. Адаптер разводит микрофоны по стереоканалам: левый = микрофон 1, правый = микрофон 2.
     Тогда всё в порядке — клиенту достаточно расщепить стерео на два канала конвейера.
  2. Адаптер СУММИРУЕТ оба микрофона в один сигнал и дублирует его в оба канала. Тогда
     разделить голоса нельзя никаким софтом: информации о том, кто говорил, в записи просто
     нет. Помогут либо два отдельных USB-входа, либо режим общего микрофона с офлайн-
     диаризацией (`/api/finalize`, `diarize=true`).

Отличить их можно только звуком в КОНКРЕТНЫЙ микрофон — поэтому тест интерактивный: он просит
говорить сначала в один, потом в другой, и смотрит, что происходит с уровнями каналов.

Запуск:
    py -3.11 -m app.test_mics                 # устройство выбирается из списка
    py -3.11 -m app.test_mics "Микрофон (USBAudio1.0)"
"""
from __future__ import annotations

import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PHASE_S = 5.0
DEFAULT_NAME = "Микрофон (USBAudio1.0)"


def _db(x: np.ndarray) -> float:
    return 20.0 * float(np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def list_devices() -> list[str]:
    """Имена dshow-входов. ffmpeg печатает их в stderr при list_devices — ловим из исключения."""
    import av

    names: list[str] = []
    try:
        av.open("dummy", format="dshow", options={"list_devices": "true"})
    except Exception as exc:
        for line in str(exc).splitlines():
            if "(audio)" in line and '"' in line:
                names.append(line.split('"')[1])
    return names


def record(name: str, seconds: float) -> tuple[np.ndarray, int]:
    """Пишет `seconds` секунд с устройства в родном формате. Возвращает (каналы x сэмплы, rate)."""
    import av

    container = av.open(f"audio={name}", format="dshow")
    chunks: list[np.ndarray] = []
    rate = 0
    channels = 0
    collected = 0
    for frame in container.decode(audio=0):
        rate = frame.sample_rate
        channels = len(frame.layout.channels)
        arr = frame.to_ndarray()
        # PyAV отдаёт либо (каналы, сэмплы), либо (1, каналы*сэмплы) с чередованием.
        if arr.shape[0] == 1 and channels > 1:
            arr = arr.reshape(-1, channels).T
        chunks.append(arr)
        collected += arr.shape[-1]
        if collected >= seconds * rate:
            break
    container.close()
    return np.concatenate(chunks, axis=-1), rate


def _countdown(text: str) -> None:
    print(f"\n{text}")
    for i in (3, 2, 1):
        print(f"   {i}...", end="\r", flush=True)
        time.sleep(1)
    print("   ГОВОРИТЕ!    ", flush=True)


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name is None:
        devices = list_devices()
        if devices:
            print("Найденные аудиовходы:")
            for i, d in enumerate(devices):
                print(f"   [{i}] {d}")
            name = next((d for d in devices if "USBAudio" in d), devices[0])
        else:
            name = DEFAULT_NAME
        print(f"\nПроверяем: {name!r}  (другое — передайте имя аргументом)")

    probe, rate = record(name, 0.5)
    n_ch = probe.shape[0]
    print(f"\nустройство отдаёт каналов: {n_ch}, частота: {rate} Гц")
    if n_ch < 2:
        print("\nВЕРДИКТ: вход МОНО — разделять нечего. Нужны два отдельных устройства записи "
              "либо режим общего микрофона с офлайн-диаризацией.")
        return 1

    print("\nДальше — две фазы по 5 секунд. Говорите ТОЛЬКО в тот микрофон, который назван;\n"
          "второй в это время не трогайте и держите подальше.")

    _countdown("ФАЗА 1 из 2 — говорите в ПЕРВЫЙ микрофон:")
    a, _ = record(name, PHASE_S)
    _countdown("ФАЗА 2 из 2 — говорите во ВТОРОЙ микрофон:")
    b, _ = record(name, PHASE_S)

    l1, r1 = _db(a[0]), _db(a[1])
    l2, r2 = _db(b[0]), _db(b[1])
    print(f"\n{'':10s}{'левый канал':>14s}{'правый канал':>15s}{'разница':>11s}")
    print(f"{'фаза 1':10s}{l1:>12.1f} дБ{r1:>13.1f} дБ{l1 - r1:>9.1f} дБ")
    print(f"{'фаза 2':10s}{l2:>12.1f} дБ{r2:>13.1f} дБ{l2 - r2:>9.1f} дБ")

    corr = float(np.corrcoef(a[0].astype(np.float64), a[1].astype(np.float64))[0, 1])
    print(f"\nкорреляция каналов в фазе 1: {corr:.4f}")

    # Разведены — если «свой» канал в каждой фазе заметно громче чужого, И перевес меняет сторону.
    swing = (l1 - r1) - (l2 - r2)
    print(f"перекладка баланса между фазами: {swing:.1f} дБ")
    if swing > 6.0:
        print("\nВЕРДИКТ: микрофоны РАЗВЕДЕНЫ по каналам ✅ — левый и правый пишут разных людей.\n"
              "Включите в клиенте режим «стерео-вход: L и R — разные участники».")
        return 0
    if corr > 0.99 and abs(swing) < 3.0:
        print("\nВЕРДИКТ: каналы несут ОДИН И ТОТ ЖЕ сигнал ❌ — адаптер суммирует микрофоны.\n"
              "Разделить голоса программно невозможно: в записи нет информации о том, кто\n"
              "говорил. Варианты: (1) два отдельных USB-входа — по адаптеру на микрофон;\n"
              "(2) режим «Один общий микрофон (диаризация)» с указанием числа голосов.")
        return 2
    print("\nВЕРДИКТ: неоднозначно. Возможно, второй микрофон слышал первого (стояли рядом)\n"
          "или в фазе было слишком тихо. Повторите, разнеся микрофоны и говоря громче.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
