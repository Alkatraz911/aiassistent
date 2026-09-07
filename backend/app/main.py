"""FastAPI backend: WebSocket real-time streaming ASR + REST для протокола.

Протокол WebSocket (бинарный + JSON):
  client -> server:
    JSON {"type":"start"}                       не обрабатывается — сессия создаётся при
                                                  подключении к сокету; см. Блок housekeeping плана
    BINARY  [int16 LE] с префиксом-каналом       кусок PCM (см. ниже)
    JSON {"type":"stop"}                          завершить, дослать хвосты
  Кадр PCM: первые 4 байта = номер канала (int32 LE), далее int16-сэмплы 16кГц.

  server -> client (Блок 2 плана — partial/stable/final):
    JSON {"type":"asr_partial", channel, utterance_id, text}
    JSON {"type":"asr_update",  channel, utterance_id, text, stable_word_count}
    JSON {"type":"asr_final",   channel, utterance_id, text, id, speaker, start, end, words}
    JSON {"type":"stopped"}                      все хвосты дослали, можно финализировать/сохранять
"""
from __future__ import annotations

import asyncio
import io
import json
import struct
import threading
import uuid

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config, device, telemetry
from .assistant import docgen, templates as templates_store
from .assistant import profile as profile_store
from .assistant import projects as projects_store
from .assistant.docx_import import normalize_docx, uncovered_placeholders
from .assistant.questionnaire import build_script
from .assistant.sessions_index import list_project_sessions
from .models import Template, TemplateStep, normalize_placeholder_token
from .session import manager

app = FastAPI(title="Протокол-ассистент MVP")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# --- преполёт: проверить устройство и прогреть модель до первой записи ---------

STARTUP_WARNING: str | None = None      # непустая строка -> GPU просили, но не поднялся

# Взведён, когда преполёт закончил и `manager.active_key` наконец означает РЕАЛЬНОЕ устройство,
# а не намерение. До этого момента ключ ещё может смениться (откат на CPU), поэтому сессию
# создавать нельзя: `Session.__init__` снимает снимок ключа на всю свою жизнь, и сессия,
# созданная в окне загрузки, навсегда осталась бы привязанной к устройству, которое так и не
# поднялось, — каждая её partial/final-задача падала бы. Ждём готовности вместо того, чтобы
# разрешать ключ позднее: ждать несколько секунд один раз на старте дешевле и понятнее, чем
# разносить «когда именно зафиксирован ключ» по всему пути постановки заданий.
PREFLIGHT_DONE = threading.Event()

CUDA_HELP = (
    "Проверьте: (1) `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` в этом venv "
    "(CTranslate2 не тянет их сам); (2) свежий драйвер NVIDIA; (3) `nvidia-smi` видит карту."
)


def _preflight() -> None:
    """Загружает и прогревает активную модель на старте, а не при первой реплике.

    Смысл ровно в двух вещах, и обе — про то, что пользователь не должен ловить их в бою:
    1. Ошибка GPU-конфигурации всплывает СЕЙЧАС, с внятным текстом, а не как «запись не
       распознаётся» посреди допроса. При провале уезжаем на CPU — но не молча: причина
       остаётся в `STARTUP_WARNING` и в `/api/health`, откуда её показывает бейдж клиента.
    2. Первая реплика не оплачивает загрузку весов (large-v3 — секунды) и ленивую
       инициализацию CUDA-модулей: `ModelManager.acquire` делает и загрузку, и warmup.
    """
    global STARTUP_WARNING
    if config.ASR_PROVIDER == "stub":
        print("[asr] stub-режим: модель не загружается")
        return

    key = manager.models.default_key()
    print(f"[asr] устройство={key[1]} compute={key[2]} модель={key[0]} "
          f"(запрошено WHISPER_DEVICE={config.WHISPER_DEVICE_REQUESTED}); {device.describe()}")
    try:
        manager.models.acquire(key)
        manager.models.release(key)
        print(f"[asr] модель {key[0]} загружена и прогрета на {key[1]}")
        return
    except Exception as exc:
        if not key[1].startswith("cuda"):
            STARTUP_WARNING = f"не удалось загрузить модель {key[0]}: {exc}"
            print(f"[asr] ОШИБКА: {STARTUP_WARNING}")
            return
        STARTUP_WARNING = (f"GPU запрошен ({key[1]}/{key[2]}), но не поднялся: {exc}. "
                           f"{CUDA_HELP} Работаем на CPU — распознавание будет медленнее.")
        print(f"[asr] ОШИБКА: {STARTUP_WARNING}")

    # Понижение до CPU: не молча (см. STARTUP_WARNING и /api/health), но и не с падением
    # сервера — сорванная запись хуже медленной.
    cpu_key = (config.WHISPER_MODEL_CPU_FALLBACK, "cpu", "int8")
    try:
        manager.models.acquire(cpu_key)
        manager.models.release(cpu_key)
        manager.set_active_model(cpu_key)
        print(f"[asr] откат на CPU: модель {cpu_key[0]}")
    except Exception as exc:
        print(f"[asr] откат на CPU тоже не удался: {exc}")


def _preflight_and_signal() -> None:
    """Преполёт + взведение `PREFLIGHT_DONE` при ЛЮБОМ исходе — иначе ожидающие зависли бы
    навсегда на непредвиденной ошибке внутри `_preflight`."""
    try:
        _preflight()
    finally:
        PREFLIGHT_DONE.set()


def _wait_preflight() -> None:
    """Блокирующее ожидание конца преполёта — звать только из потока, не из event loop.

    Без таймаута намеренно: событие взводится в `finally`, то есть ожидание кончается при любом
    исходе, а обрывать по таймауту первую загрузку весов (large-v3 качается минутами на первом
    запуске) значило бы ломать самый обычный сценарий ради несуществующего дедлока.
    """
    PREFLIGHT_DONE.wait()


@app.on_event("startup")
async def _on_startup() -> None:
    # В отдельном потоке: загрузка модели — секунды, а event loop должен принимать соединения.
    # Но принимать — не значит обслуживать вслепую: точки, создающие сессию, ждут PREFLIGHT_DONE.
    asyncio.get_running_loop().run_in_executor(None, _preflight_and_signal)


@app.get("/api/health")
def health() -> dict:
    cuda_devices, cuda_why = device.probe_cuda()
    loading = not PREFLIGHT_DONE.is_set()
    return {
        # Пока преполёт не кончился, `ok: true` был бы обещанием, которого мы не давали:
        # `device` в этот момент — ещё НАМЕРЕНИЕ, и при провале GPU оно сменится на cpu, а
        # STARTUP_WARNING ещё пуст. Раньше клиент в это окно рисовал зелёный бейдж «GPU».
        "ok": STARTUP_WARNING is None and not loading,
        "loading": loading,
        "asr": config.ASR_PROVIDER,
        "model": manager.active_model,
        "device": manager.active_key[1],
        "compute": manager.active_key[2],
        "cuda_devices": cuda_devices,
        "cuda_unavailable_reason": cuda_why or None,
        # Не-None и на время загрузки: клиент, не знающий про `loading`, всё равно не покажет
        # зелёное «всё хорошо» там, где итог ещё не известен.
        "warning": STARTUP_WARNING or ("модель ещё загружается" if loading else None),
    }


# --- AI-ассистент: сценарий анкеты --------------------------------------

class SessionReq(BaseModel):
    session_id: str
    template_id: str | None = None   # Блок 5: какой шаблон анкеты использовать


class AnswerReq(BaseModel):
    session_id: str
    answer: str


def _step_payload(step) -> dict:
    return {"step": step.key, "label": step.label, "kind": step.kind,
            "prompt": step.prompt, "statement": step.statement, "question": step.question,
            "needs_answer": step.kind != "info", "source": step.source}


@app.post("/api/assistant/start")
def assistant_start(req: SessionReq) -> dict:
    # Сессия фиксирует ключ модели на всю жизнь — создавать её до конца преполёта нельзя
    # (см. PREFLIGHT_DONE). Хендлер синхронный, FastAPI держит его в threadpool, так что
    # ожидание здесь не блокирует event loop.
    _wait_preflight()
    s = manager.get(req.session_id) or manager.create(req.session_id)
    if req.template_id:
        tmpl = templates_store.store.get(req.template_id)
        if tmpl is None:
            return JSONResponse({"error": "шаблон не найден"}, status_code=404)
        s.assistant.load_script(build_script(tmpl.steps))
        s.protocol.template_id = tmpl.id
        s.protocol.template_name = tmpl.name
        s.protocol.template_version = tmpl.version
        s.protocol.template_snapshot = tmpl
    step = s.assistant.start()
    return _step_payload(step)


# --- шаблоны анкеты (Блок 5) ---------------------------------------------

class TemplateSaveReq(BaseModel):
    name: str
    description: str = ""
    steps: list[TemplateStep]
    # Черновик из /api/templates/import_docx несёт свой id и путь к уже сохранённому докс-файлу —
    # без явной передачи id при сохранении получил бы НОВЫЙ id, и docx/<старый_id>.docx осиротел
    # бы (см. комментарий в TemplateStore.create). id значим только для POST (создание);
    # PUT игнорирует его — id шаблона в апдейте берётся из URL.
    id: str | None = None
    docx_filename: str | None = None
    qa_placeholder: str | None = None


@app.get("/api/templates")
def list_templates() -> list[dict]:
    return [t.model_dump() for t in templates_store.store.list()]


@app.get("/api/templates/{template_id}")
def get_template(template_id: str) -> dict:
    tmpl = templates_store.store.get(template_id)
    if tmpl is None:
        return JSONResponse({"error": "шаблон не найден"}, status_code=404)
    return tmpl.model_dump()


@app.post("/api/templates")
def create_template(req: TemplateSaveReq) -> dict:
    try:
        tmpl = templates_store.store.create(
            req.name, req.description, req.steps,
            id=req.id, docx_filename=req.docx_filename, qa_placeholder=req.qa_placeholder,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    return tmpl.model_dump()


@app.put("/api/templates/{template_id}")
def update_template(template_id: str, req: TemplateSaveReq) -> dict:
    try:
        tmpl = templates_store.store.update(
            template_id, req.name, req.description, req.steps,
            docx_filename=req.docx_filename, qa_placeholder=req.qa_placeholder,
        )
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    if tmpl is None:
        return JSONResponse({"error": "шаблон не найден"}, status_code=404)
    return tmpl.model_dump()


@app.delete("/api/templates/{template_id}")
def delete_template(template_id: str) -> dict:
    try:
        ok = templates_store.store.delete(template_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    if not ok:
        return JSONResponse({"error": "шаблон не найден"}, status_code=404)
    return {"ok": True}


@app.post("/api/templates/import_docx")
async def import_docx_template(file: UploadFile = File(...)) -> dict:
    """Импорт реального .docx-бланка протокола: находит `#{NS.FIELD}`-токены, нормализует их в
    Jinja-плейсхолдеры и возвращает ЧЕРНОВИК шаблона (по одному шагу на найденный токен) —
    оператор донастраивает формулировки/порядок в редакторе и сохраняет через POST /api/templates
    (передавая тот же id и docx_filename из этого ответа, см. TemplateSaveReq). Ничего не пишет
    в TemplateStore здесь — черновик до подтверждения оператором не считается сохранённым
    шаблоном."""
    raw = await file.read()
    new_id = uuid.uuid4().hex[:12]
    docx_filename = f"{new_id}.docx"
    dst = config.TEMPLATES_DIR / "docx" / docx_filename
    try:
        placeholders = normalize_docx(io.BytesIO(raw), dst)
    except Exception as e:
        return JSONResponse({"error": f"не удалось разобрать .docx: {e}"}, status_code=422)

    steps = [
        TemplateStep(key=token, label=token, kind="field", source="manual", placeholder=token)
        for token in placeholders
    ]
    tmpl = Template(id=new_id, name="", docx_filename=docx_filename, steps=steps)
    return tmpl.model_dump()


@app.get("/api/templates/{template_id}/coverage")
def template_docx_coverage(template_id: str) -> dict:
    """Плейсхолдеры, реально встречающиеся в привязанном .docx-бланке, но не покрытые НИ ОДНИМ
    шагом (и не выбранные как qa_placeholder) — Блок 6, страховка от того, что при ручном
    редактировании шаблона шаг удалили или переименовали плейсхолдер с опечаткой. Такой
    плейсхолдер в итоговом документе тихо остаётся пустым (см. docgen.render), а голосовая
    анкета вообще не спросит соответствующую ей информацию — оператор должен узнать об этом
    сразу при сохранении шаблона, а не постфактум по готовому документу."""
    tmpl = templates_store.store.get(template_id)
    if tmpl is None:
        return JSONResponse({"error": "шаблон не найден"}, status_code=404)
    if not tmpl.docx_filename:
        return {"missing": []}
    docx_path = config.TEMPLATES_DIR / "docx" / tmpl.docx_filename
    if not docx_path.exists():
        return {"missing": []}
    return {"missing": uncovered_placeholders(tmpl, docx_path)}


# --- профиль оператора (Блок 6) -------------------------------------------
# Значения полей с source="profile" (напр. «автор документа») — общие для всех сессий этого
# пользователя программы, заполняются один раз, а не в каждой анкете (см. assistant/profile.py).

@app.get("/api/profile/fields")
def list_profile_fields() -> list[dict]:
    """Все различные профильные токены (`placeholder or key`, где `source == "profile"`) из ВСЕХ
    сохранённых шаблонов — чтобы клиент мог показать оператору одну общую форму настройки,
    а не заставлять его открывать каждый шаблон отдельно. Дубли (разные шаблоны используют один
    и тот же токен) схлопываются — берём подпись первого встреченного."""
    seen: dict[str, str] = {}
    for summary in templates_store.store.list():
        tmpl = templates_store.store.get(summary.id)
        if tmpl is None:
            continue
        for step in tmpl.steps:
            if step.source != "profile":
                continue
            token = step.placeholder or step.key
            if token and token not in seen:
                seen[token] = step.label or token
    return [{"placeholder": token, "label": label} for token, label in sorted(seen.items())]


@app.get("/api/profile")
def get_profile() -> dict[str, str]:
    return profile_store.store.get_all()


class ProfileSaveReq(BaseModel):
    values: dict[str, str]


@app.post("/api/profile")
def save_profile(req: ProfileSaveReq) -> dict[str, str]:
    # Ключи — токены плейсхолдеров (`TemplateStep.placeholder or .key`), которые клиент теперь
    # нормализует на вводе (см. app.js::normalizePlaceholderToken), а `TemplateStep`/`Template` —
    # на сохранении шаблона (см. models.normalize_placeholder_token). Страховка и здесь: запрос
    # мог прийти не из штатного редактора, а токены в /api/profile/fields — из шаблона,
    # сохранённого ДО этого фикса (реальный случай: "T1.DOC_AUTHOR_FULL_INFO" и
    # "T1.DOC_AUTHOR_FULL_INFO}" осели в profile.json как две разные записи).
    values = {normalize_placeholder_token(k): v for k, v in req.values.items()}
    return profile_store.store.update(values)


# --- проекты/дела и история допросов (Блок 7) ------------------------------

@app.get("/api/projects")
def list_projects() -> list[dict]:
    return [p.model_dump() for p in projects_store.store.list()]


class ProjectCreateReq(BaseModel):
    name: str


@app.post("/api/projects")
def create_project(req: ProjectCreateReq) -> dict:
    name = req.name.strip()
    if not name:
        return JSONResponse({"error": "укажите название проекта"}, status_code=422)
    return projects_store.store.create(name).model_dump()


@app.get("/api/projects/{project_id}/sessions")
def project_sessions(project_id: str) -> list[dict]:
    if projects_store.store.get(project_id) is None:
        return JSONResponse({"error": "проект не найден"}, status_code=404)
    return [s.model_dump() for s in list_project_sessions(project_id)]


class SessionInitReq(BaseModel):
    session_id: str
    project_id: str


@app.post("/api/session/init")
def session_init(req: SessionInitReq) -> dict:
    """Создаёт (если ещё не существует) сессию и сразу привязывает её к проекту — до начала
    анкеты/записи, чтобы допрос попал в список проекта независимо от того, с чего реально
    начнётся работа (анкета или сразу запись)."""
    if projects_store.store.get(req.project_id) is None:
        return JSONResponse({"error": "проект не найден"}, status_code=404)
    s = manager.get(req.session_id) or manager.create(req.session_id)
    s.protocol.project_id = req.project_id
    return {"ok": True}


@app.post("/api/protocol/{session_id}/docx")
def generate_protocol_docx(session_id: str):
    """Заполняет докс-шаблон, привязанный к пройденному шаблону анкеты сессии, и отдаёт готовый
    .docx на скачивание. Требует, чтобы сессия проходила анкету с шаблоном, у которого задан
    docx_filename (см. docgen.render)."""
    s = manager.get_or_load(session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    snapshot = s.protocol.template_snapshot
    if snapshot is None or not snapshot.docx_filename:
        return JSONResponse(
            {"error": "у шаблона этой сессии нет привязанного .docx-бланка"}, status_code=422)
    # `protocol.questionnaire` — не живые данные, а снапшот, который синхронизируется из
    # `assistant.answers` только внутри `Session.save()`. Без этого вызова здесь генерация
    # молча уходила бы в MISSING_MARKER по всем полям анкеты, даже если оператор их надиктовал —
    # реальный баг, воспроизведённый на живой сессии. `save()` заодно кладёт protocol.json на
    # диск в состоянии, согласованном с только что сформированным .docx.
    s.save()
    try:
        out_path = docgen.render(
            s.protocol,
            docx_template_path=config.TEMPLATES_DIR / "docx" / snapshot.docx_filename,
            out_path=s.dir / "protocol.docx",
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    return FileResponse(
        str(out_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"protocol_{session_id}.docx",
    )


@app.post("/api/transcribe")
async def transcribe_oneshot(request: Request, session_id: str | None = None) -> dict:
    """Разовая транскрипция куска (raw int16 LE, 16кГц моно) — для ответов анкеты. Идёт через
    общий ASR-планировщик (Блок 0.1), не напрямую в модель — не гонится с live-стримингом.

    `session_id` (Блок 6, опционально) — если передан и сессия существует, в initial_prompt
    подмешивается `assistant.recent_context` (подтверждённые ответы этой же анкеты). Раньше
    ответы анкеты распознавались с ГОЛЫМ статичным WHISPER_PROMPT без какого-либо контекста —
    заметно хуже, чем реплики потокового распознавания, у которого initial_prompt всегда несёт
    хвост уже распознанной речи того же канала (см. Session._build_prompt)."""
    raw = await request.body()
    if len(raw) < 2:
        return {"text": ""}
    pcm_int16 = np.frombuffer(raw, dtype="<i2")
    prompt = None
    if session_id:
        s = manager.get(session_id)
        if s is not None and s.assistant.recent_context:
            prompt = f"{config.WHISPER_PROMPT} {s.assistant.recent_context}".strip()
    text = await asyncio.to_thread(manager.transcribe_oneshot, pcm_int16, initial_prompt=prompt)
    return {"text": text}


@app.post("/api/assistant/answer")
def assistant_answer(req: AnswerReq) -> dict:
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    nxt = s.assistant.submit_answer(req.answer)
    return {"next": nxt, "fields": s.assistant.to_fields()}


@app.post("/api/assistant/next")
def assistant_next(req: SessionReq) -> dict:
    """Перейти к следующему шагу без ответа (для info-шагов)."""
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    nxt = s.assistant.advance()
    return {"next": nxt, "fields": s.assistant.to_fields()}


class FieldEditReq(BaseModel):
    session_id: str
    key: str
    value: str


@app.post("/api/assistant/field")
def assistant_field_edit(req: FieldEditReq) -> dict:
    """Прямая правка/заполнение поля анкеты по ключу, в обход последовательного хода анкеты
    (Блок 6) — таблица «Поля анкеты» в клиенте редактируема, см. `AssistantSession.set_answer`."""
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    ok = s.assistant.set_answer(req.key, req.value)
    return {"ok": ok, "fields": s.assistant.to_fields()}


# --- маркировка спикеров и правки ---------------------------------------

class SpeakerReq(BaseModel):
    session_id: str
    channel: int
    label: str
    # Текущая метка переименовываемого голоса. Задана — переименовываем именно его; не задана —
    # прежнее поведение «весь канал» (микрофон-на-участника, где канал и есть спикер).
    speaker: str | None = None


@app.post("/api/speaker")
def set_speaker(req: SpeakerReq) -> dict:
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    if req.speaker:
        n = s.rename_speaker(req.speaker, req.label)
    else:
        n = s.set_speaker(req.channel, req.label)
    return {"updated": n, "label": req.label}


class EditReq(BaseModel):
    session_id: str
    segment_id: str
    text: str


@app.post("/api/segment/edit")
def edit_segment(req: EditReq) -> dict:
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    ok = s.edit_segment(req.segment_id, req.text)
    return {"ok": ok}


class SegmentDeleteReq(BaseModel):
    session_id: str
    segment_id: str


@app.post("/api/segment/delete")
def delete_segment(req: SegmentDeleteReq) -> dict:
    """Удалить реплику, помеченную вероятным дублем протёкшего голоса (Блок 3.7) — при близко
    расположенных микрофонах автоматика не может надёжно решить, кто реальный автор, поэтому
    помечает ОБЕ копии (см. Session._maybe_mark_crosstalk_duplicate), а лишнюю удаляет оператор."""
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    try:
        result = s.delete_segment(req.segment_id)
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    if result is None:
        return JSONResponse({"error": "сегмент не найден"}, status_code=404)
    # partner — сегмент, с которого автоматически снята пометка дубля (если она была): без
    # партнёра сравнивать больше не с чем, второй копии в паре уже нет. Клиент обновляет его на
    # месте по этому полю ответа — WS-обновление могло никуда не дойти (см. delete_segment).
    return {"ok": True, "partner": result["partner"]}


@app.post("/api/segment/unflag")
def unflag_segment(req: SegmentDeleteReq) -> dict:
    """Снять пометку «вероятный дубль», не удаляя реплику (Блок 3.7) — для ложных совпадений
    или для уже сохранённых сессий, где партнёра удалили ДО того, как delete_segment научился
    чистить пару автоматически, и флаг остался висеть сиротой."""
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    result = s.clear_bleed_flag(req.segment_id)
    if result is None:
        return JSONResponse({"error": "сегмент не найден"}, status_code=404)
    return {"ok": True, "partner": result["partner"]}


class FinalizeReq(BaseModel):
    session_id: str
    align: bool = True                 # уточнить тайм-коды (forced alignment)
    diarize: bool = False              # развести голоса (режим общего микрофона)
    channel: int = 0                   # канал общего микрофона
    num_speakers: int | None = None    # ожидаемое число голосов (None = авто)


@app.post("/api/finalize")
def finalize(req: FinalizeReq) -> dict:
    """Офлайн-финализация: точные тайм-коды (alignment) и опц. диаризация общего
    микрофона. Всё локально, без токенов."""
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    result: dict = {}
    try:
        if req.diarize:
            result["speakers"] = s.diarize_single_mic(
                manager.diarizer, channel=req.channel, num_speakers=req.num_speakers)
        if req.align:
            result["aligned_segments"] = s.finalize_alignment(manager.aligner)
    except Exception as e:
        return JSONResponse({"error": f"finalize failed: {e}"}, status_code=500)
    result["protocol"] = s.protocol.model_dump()
    return result


@app.post("/api/save")
def save(req: SessionReq) -> dict:
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    path = s.save()
    return {"saved": str(path)}


@app.get("/api/protocol/{session_id}")
def get_protocol(session_id: str) -> dict:
    # get_or_load, не get — открыть сохранённый допрос (Блок 7) должно работать и после
    # перезапуска backend, когда сессии уже нет в памяти.
    s = manager.get_or_load(session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    return s.protocol.model_dump()


@app.get("/api/audio/{session_id}")
def get_audio(session_id: str):
    s = manager.get_or_load(session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    path = s.build_mix()  # свести по-канальные WAV в моно-микс для плеера
    if not path.exists():
        return JSONResponse({"error": "no audio"}, status_code=404)
    return FileResponse(str(path), media_type="audio/wav")


# --- телеметрия (Блок 1, диагностика) ------------------------------------

@app.get("/api/telemetry/{session_id}")
def get_telemetry(session_id: str) -> dict:
    """Диагностический эндпоинт для тюнинга задержки — не часть основного UI. Отдаёт перцентили
    (p50/p95/p99) по 'first partial after speech start' и 'final after speech end' + сырые события."""
    t = telemetry.registry.get(session_id)
    if t is None:
        return JSONResponse({"error": "no telemetry for session"}, status_code=404)
    return {"summary": t.summary(), "events": t.raw_events()}


# --- выбор модели ASR (Блок 4) -------------------------------------------

class ModelSwitchReq(BaseModel):
    model: str
    device: str | None = None
    compute: str | None = None


CURATED_MODELS = ["small", "medium", "turbo", "large-v3"]


@app.get("/api/models")
def list_models() -> dict:
    key = manager.active_key
    return {
        "models": CURATED_MODELS,
        "active": key[0],
        "loaded": [k[0] for k in manager.models.loaded_keys()],
        # Устройство отдаём вместе со списком, чтобы UI мог показать, на чём реально считаем:
        # выбор «large-v3» означает совершенно разные вещи на GPU и на CPU (на CPU он не тянет
        # live вовсе), и пользователь должен видеть это до начала записи, а не по задержке.
        "device": key[1],
        "compute": key[2],
        "warning": STARTUP_WARNING,
    }


@app.post("/api/models/switch")
async def switch_model(req: ModelSwitchReq):
    """Переключение модели по требованию (Блок 4): первый раз — с задержкой загрузки (в
    threadpool, не блокирует event loop), повторно — из кэша ModelManager, мгновенно. Запрещено
    во время активной записи — переключение модели на середине потока не имеет смысла и рискует
    гонкой; клиент дублирует эту блокировку на своей стороне, но источник истины — сервер."""
    if manager.any_streaming():
        return JSONResponse(
            {"error": "нельзя менять модель во время активной записи"}, status_code=409)
    import time
    # Устройство берём от АКТИВНОГО ключа, а не из config: если преполёт откатил нас на CPU
    # (GPU не поднялся), переключение модели из UI не должно молча возвращать нас на cuda.
    active = manager.active_key
    key = (req.model, req.device or active[1], req.compute or active[2])
    t0 = time.monotonic()
    try:
        await asyncio.to_thread(manager.models.acquire, key)
    except RuntimeError as exc:
        # Типовой отказ именно на GPU: две тяжёлые модели в кэше (ASR_MAX_CACHED_MODELS=2) не
        # влезают в VRAM. Сообщение CTranslate2 про out of memory ничего не говорит оператору о
        # том, что делать, — переводим его в действие.
        if "out of memory" in str(exc).lower():
            return JSONResponse(
                {"error": f"не хватает видеопамяти для модели {req.model}. Уменьшите "
                          f"ASR_MAX_CACHED_MODELS до 1, возьмите модель полегче или "
                          f"WHISPER_COMPUTE=int8_float16"},
                status_code=507)
        raise
    manager.models.release(key)   # acquire() держит рефкаунт только на время использования джобом
    manager.set_active_model(key)
    took_ms = int((time.monotonic() - t0) * 1000)
    return {"active": key[0], "loaded": [k[0] for k in manager.models.loaded_keys()], "took_ms": took_ms}


# --- WebSocket стриминг --------------------------------------------------

async def _drain_outbox(ws: WebSocket, out_queue: asyncio.Queue) -> None:
    while True:
        msg = await out_queue.get()
        await ws.send_json(msg)


@app.websocket("/ws/stream/{session_id}")
async def ws_stream(ws: WebSocket, session_id: str) -> None:
    # Ждём преполёт ДО accept, а не после: клиент начинает захват и шлёт PCM по `onopen`, и
    # принять сокет раньше готовности значило бы либо копить кадры, либо гнать их в сессию с
    # ещё не устоявшимся ключом модели (см. PREFLIGHT_DONE).
    await asyncio.to_thread(_wait_preflight)
    await ws.accept()
    s = manager.get(session_id) or manager.create(session_id)
    loop = asyncio.get_running_loop()
    out_queue: asyncio.Queue = asyncio.Queue()
    s.attach_ws(loop, out_queue)
    drain_task = asyncio.create_task(_drain_outbox(ws, out_queue))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if (data := msg.get("bytes")) is not None:
                if len(data) < 4:
                    continue
                channel = struct.unpack_from("<i", data, 0)[0]
                pcm = np.frombuffer(data, dtype="<i2", offset=4)
                # Инференс идёт на отдельном потоке планировщика (Блок 0.1) — эта строка сама
                # по себе не блокирует event loop, она только копит буферы и ставит задание в очередь.
                s.ingest_streaming(channel, pcm)

            elif (text := msg.get("text")) is not None:
                evt = json.loads(text)
                if evt.get("type") == "stop":
                    for ch in list(s.stream.keys()):
                        s.force_finalize_channel(ch)
                    deadline = loop.time() + 5.0
                    while s.has_pending_finals() and loop.time() < deadline:
                        await asyncio.sleep(0.05)
                    # дать драйн-задаче шанс переслать последние сообщения из очереди
                    await asyncio.sleep(0.05)
                    await ws.send_json({"type": "stopped"})
    except WebSocketDisconnect:
        pass
    finally:
        drain_task.cancel()
        s.detach_ws()
