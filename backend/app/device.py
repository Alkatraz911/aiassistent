"""Выбор и диагностика вычислительного устройства для ASR (CPU / CUDA).

Три задачи, каждая — про реальные грабли GPU-запуска на Windows, а не про красоту:

1. `ensure_cuda_dlls()` — Windows не находит cuBLAS/cuDNN, поставленные как pip-пакеты
   (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`): их DLL лежат в `site-packages/nvidia/*/bin`,
   которого нет в PATH. torch дописывает эти каталоги сам при импорте, а CTranslate2 (движок
   faster-whisper) — нет, поэтому при живом драйвере и корректно установленных пакетах он всё
   равно падает с «Library cublas64_12.dll is not found». Регистрируем каталоги через
   `os.add_dll_directory()` ДО первого импорта ctranslate2.

2. `probe_cuda()` — сколько CUDA-устройств реально видит CTranslate2. Это НЕ гарантия, что
   инференс поедет: счётчик устройств отдаёт CUDA-runtime, а cuBLAS/cuDNN подгружаются лениво
   при первом декодировании. Поэтому окончательная проверка — реальный warmup (см.
   `main.py::_startup`), а не этот счётчик.

3. `resolve_device()` — разбор `WHISPER_DEVICE=auto`: cuda, если она есть, иначе cpu.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

# Подкаталоги site-packages/nvidia/*, в которых pip-пакеты NVIDIA держат Windows-DLL.
_NVIDIA_DLL_SUBDIRS = ("bin", "lib")


@lru_cache(maxsize=1)
def ensure_cuda_dlls() -> list[str]:
    """Делает DLL из pip-пакетов `nvidia-*-cu12` видимыми для CTranslate2 (только Windows).

    Возвращает список зарегистрированных каталогов (пустой — если регистрировать нечего или
    платформа не Windows). Идемпотентна: `lru_cache` не даёт добавить одни и те же каталоги
    дважды при повторных вызовах из разных модулей.
    """
    if sys.platform != "win32":
        return []   # на Linux пакеты кладут .so туда, где их находит ld.so/RPATH

    registered: list[str] = []
    for site_dir in _site_packages_dirs():
        nvidia_root = os.path.join(site_dir, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in sorted(os.listdir(nvidia_root)):
            for sub in _NVIDIA_DLL_SUBDIRS:
                cand = os.path.join(nvidia_root, pkg, sub)
                if not os.path.isdir(cand):
                    continue
                if not any(f.lower().endswith(".dll") for f in os.listdir(cand)):
                    continue
                try:
                    os.add_dll_directory(cand)
                except OSError:
                    continue
                # PATH — для дочерних процессов и загрузчиков, которые не смотрят на
                # add_dll_directory (некоторые нативные расширения ищут по старинке).
                os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
                registered.append(cand)
    return registered


def _site_packages_dirs() -> list[str]:
    dirs: list[str] = []
    try:
        import site
        dirs.extend(site.getsitepackages())
        user = site.getusersitepackages()
        if isinstance(user, str):
            dirs.append(user)
    except Exception:
        pass
    # В venv `site.getsitepackages()` иногда не отдаёт Lib/site-packages — добавим руками.
    dirs.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    seen: set[str] = set()
    return [d for d in dirs if d and os.path.isdir(d) and not (d in seen or seen.add(d))]


@lru_cache(maxsize=1)
def probe_cuda() -> tuple[int, str]:
    """(число видимых CUDA-устройств, причина недоступности). Импортирует ctranslate2."""
    ensure_cuda_dlls()
    try:
        import ctranslate2
    except Exception as exc:                       # ctranslate2 не установлен (например, stub-режим)
        return 0, f"ctranslate2 недоступен: {exc}"
    try:
        n = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:
        return 0, f"CUDA-runtime не отвечает: {exc}"
    if n <= 0:
        return 0, "CUDA-устройств не найдено (нет GPU/драйвера или сборка ctranslate2 без CUDA)"
    return n, ""


def resolve_device(requested: str) -> str:
    """`auto` -> cuda/cpu; явные `cuda`/`cpu` возвращаются как есть.

    Явный `cuda` здесь намеренно не проверяется и не понижается: угадывать за пользователя
    нечего, а отказ всё равно всплывёт при реальном декодировании. Разбирается с ним преполёт
    (`main.py::_preflight`) — он откатывается на CPU и пишет причину в `/api/health`.
    Подбор устройства делает только `auto`.
    """
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        return requested
    n, _ = probe_cuda()
    return "cuda" if n > 0 else "cpu"


def describe() -> str:
    """Однострочная диагностика для логов старта."""
    n, why = probe_cuda()
    if n > 0:
        return f"CUDA доступна: устройств {n}"
    return f"CUDA недоступна ({why})"


def resolve_torch_device(requested: str) -> str:
    """То же, что `resolve_device`, но для torch (офлайн-финализация: alignment + диаризация).

    Отдельная функция, потому что это независимый стек: CUDA-сборка torch ставится отдельно от
    CUDA-библиотек CTranslate2, и «ASR на GPU, финализация на CPU» — нормальная рабочая
    конфигурация (requirements.txt по умолчанию тянет именно CPU-сборку torch).
    """
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
