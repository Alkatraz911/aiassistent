"""FastAPI backend: WebSocket-стриминг аудио + REST для протокола.

Протокол WebSocket (бинарный + JSON):
  client -> server:
    JSON {"type":"start"}                       начать сессию записи
    BINARY  [int16 LE] с префиксом-каналом       кусок PCM (см. ниже)
    JSON {"type":"stop"}                          завершить, дослать хвосты
  Кадр PCM: первые 4 байта = номер канала (int32 LE), далее int16-сэмплы 16кГц.

  server -> client:
    JSON {"type":"segment", ...}                 новый/обновлённый сегмент
"""
from __future__ import annotations

import struct

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .session import manager

app = FastAPI(title="Протокол-ассистент MVP")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    from . import config
    return {"ok": True, "asr": config.ASR_PROVIDER, "model": config.WHISPER_MODEL}


# --- AI-ассистент: сценарий анкеты --------------------------------------

class SessionReq(BaseModel):
    session_id: str


class AnswerReq(BaseModel):
    session_id: str
    answer: str


@app.post("/api/assistant/start")
def assistant_start(req: SessionReq) -> dict:
    s = manager.get(req.session_id) or manager.create(req.session_id)
    step = s.assistant.start()
    return {"step": step.key, "label": step.label, "kind": step.kind,
            "prompt": step.prompt, "needs_answer": step.kind != "info"}


@app.post("/api/transcribe")
async def transcribe_oneshot(request: Request) -> dict:
    """Разовая транскрипция куска (raw int16 LE, 16кГц моно) — для ответов анкеты."""
    raw = await request.body()
    if len(raw) < 2:
        return {"text": ""}
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    from . import config
    result = manager.asr.transcribe(pcm, config.SAMPLE_RATE)
    return {"text": result.text}


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
    if not s or not s.audio_path.exists():
        return JSONResponse({"error": "no audio"}, status_code=404)
    s.close()  # дописать WAV-заголовок перед отдачей
    return FileResponse(str(s.audio_path), media_type="audio/wav")


# --- WebSocket стриминг --------------------------------------------------

def _segment_payload(seg) -> dict:
    return {
        "type": "segment",
        "id": seg.id, "channel": seg.channel, "speaker": seg.speaker,
        "speaker_auto": seg.speaker_auto, "start": seg.start, "end": seg.end,
        "text": seg.text,
        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in seg.words],
    }


@app.websocket("/ws/stream/{session_id}")
async def ws_stream(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    s = manager.get(session_id) or manager.create(session_id)
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
                for seg in s.ingest(channel, pcm):
                    await ws.send_json(_segment_payload(seg))

            elif (text := msg.get("text")) is not None:
                import json
                evt = json.loads(text)
                if evt.get("type") == "stop":
                    for ch in list(s.buffers.keys()):
                        for seg in s.finalize_channel(ch):
                            await ws.send_json(_segment_payload(seg))
                    await ws.send_json({"type": "stopped"})
    except WebSocketDisconnect:
        pass
