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
import json
import struct

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config, telemetry
from .assistant import templates as templates_store
from .assistant.questionnaire import build_script
from .models import TemplateStep
from .session import manager

app = FastAPI(title="Протокол-ассистент MVP")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "asr": config.ASR_PROVIDER, "model": manager.active_model}


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
            "needs_answer": step.kind != "info"}


@app.post("/api/assistant/start")
def assistant_start(req: SessionReq) -> dict:
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
        tmpl = templates_store.store.create(req.name, req.description, req.steps)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    return tmpl.model_dump()


@app.put("/api/templates/{template_id}")
def update_template(template_id: str, req: TemplateSaveReq) -> dict:
    try:
        tmpl = templates_store.store.update(template_id, req.name, req.description, req.steps)
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
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    if not ok:
        return JSONResponse({"error": "шаблон не найден"}, status_code=404)
    return {"ok": True}


@app.post("/api/transcribe")
async def transcribe_oneshot(request: Request) -> dict:
    """Разовая транскрипция куска (raw int16 LE, 16кГц моно) — для ответов анкеты. Идёт через
    общий ASR-планировщик (Блок 0.1), не напрямую в модель — не гонится с live-стримингом."""
    raw = await request.body()
    if len(raw) < 2:
        return {"text": ""}
    pcm_int16 = np.frombuffer(raw, dtype="<i2")
    text = await asyncio.to_thread(manager.transcribe_oneshot, pcm_int16)
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


# --- маркировка спикеров и правки ---------------------------------------

class SpeakerReq(BaseModel):
    session_id: str
    channel: int
    label: str


@app.post("/api/speaker")
def set_speaker(req: SpeakerReq) -> dict:
    s = manager.get(req.session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
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
    s = manager.get(session_id)
    if not s:
        return JSONResponse({"error": "no session"}, status_code=404)
    return s.protocol.model_dump()


@app.get("/api/audio/{session_id}")
def get_audio(session_id: str):
    s = manager.get(session_id)
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
    return {
        "models": CURATED_MODELS,
        "active": manager.models.default_key()[0],
        "loaded": [k[0] for k in manager.models.loaded_keys()],
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
    key = (req.model, req.device or config.WHISPER_DEVICE, req.compute or config.WHISPER_COMPUTE)
    t0 = time.monotonic()
    await asyncio.to_thread(manager.models.acquire, key)
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
