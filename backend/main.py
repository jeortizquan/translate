"""
Live Translate — FastAPI Application
=====================================
HTTP routes:
  GET  /                           → redirect to /client
  GET  /client                     → audience portal (HTML)
  GET  /operator                   → operator panel (HTML)
  GET  /api/status                 → server status JSON
  GET  /api/microphones            → list audio input devices
  GET  /api/voices                 → all available voices (keyed by lang)
  GET  /api/voices/{lang}          → available voices for one language
  POST /api/operator/settings      → update runtime settings live
  POST /api/operator/start         → start whisper-stream STT
  POST /api/operator/stop          → stop whisper-stream STT

WebSocket endpoints:
  WS   /ws/listen/{lang}           → audience: binary WAV + JSON subtitles
  WS   /ws/operator                → operator: status events
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .broadcast import manager
from .config import (
    LANGUAGES, WHISPER_MODELS, SOURCE_LANGUAGES,
    settings, VOICES_DIR, MODELS_DIR, WHISPER_STREAM_BIN, WHISPER_DIR, FRONTEND_DIR,
)
from .pipeline import pipeline
from .stt import WhisperStreamManager, list_microphones
from .tts import make_tts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Globals ───────────────────────────────────────────────────────────────────
whisper = WhisperStreamManager()

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Live Translate starting ===")
    await pipeline.start()
    yield
    logger.info("=== Live Translate shutting down ===")
    await whisper.stop()
    await pipeline.stop()

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Live Translate", version="1.0.0", lifespan=lifespan)

_frontend = Path(FRONTEND_DIR)
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

# ─── Pages ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/client")

@app.get("/client", response_class=HTMLResponse)
async def client_page():
    return HTMLResponse((_frontend / "client.html").read_text(encoding="utf-8"))

@app.get("/operator", response_class=HTMLResponse)
async def operator_page():
    return HTMLResponse((_frontend / "operator.html").read_text(encoding="utf-8"))

# ─── REST API ─────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    return {
        "running":             whisper.is_running,
        "translation_engine":  settings.translation_engine,
        "tts_engine":          settings.tts_engine,
        "whisper_model":       settings.whisper_model,
        "source_language":     settings.source_language,
        "tts_speed":           settings.tts_speed,
        "total_connections":   manager.total(),
        "connections_by_lang": manager.counts(),
        "supported_languages": {
            code: {"name": v["name"], "flag": v["flag"]}
            for code, v in LANGUAGES.items()
        },
        "source_languages":    SOURCE_LANGUAGES,
        "whisper_models":      WHISPER_MODELS,
        "voice_overrides":     settings.voice_overrides,
    }

@app.get("/api/microphones")
async def api_microphones():
    return {"microphones": list_microphones()}

@app.get("/api/voices")
async def api_all_voices():
    tts = make_tts(settings.tts_engine, VOICES_DIR)
    return {lang: tts.available_voices(lang) for lang in LANGUAGES}

@app.get("/api/voices/{lang}")
async def api_voices(lang: str):
    if lang not in LANGUAGES:
        raise HTTPException(404, f"Unknown language: {lang}")
    tts = make_tts(settings.tts_engine, VOICES_DIR)
    return {"lang": lang, "voices": tts.available_voices(lang)}

# ─── Operator Settings ────────────────────────────────────────────────────────
class SettingsPayload(BaseModel):
    whisper_model:      Optional[str]   = None
    source_language:    Optional[str]   = None
    microphone_index:   Optional[int]   = None
    translation_engine: Optional[str]   = None
    tts_engine:         Optional[str]   = None
    tts_speed:          Optional[float] = None
    voice_overrides:    Optional[dict]  = None
    gemma_host:         Optional[str]   = None

@app.post("/api/operator/settings")
async def update_settings(payload: SettingsPayload):
    engine_changed = False

    if payload.whisper_model and payload.whisper_model in WHISPER_MODELS:
        settings.whisper_model = payload.whisper_model
    if payload.source_language and payload.source_language in SOURCE_LANGUAGES:
        settings.source_language = payload.source_language
    if payload.microphone_index is not None:
        settings.microphone_index = payload.microphone_index
    if payload.tts_speed is not None:
        settings.tts_speed = max(0.5, min(2.0, payload.tts_speed))
    if payload.voice_overrides is not None:
        settings.voice_overrides.update(payload.voice_overrides)
    if payload.gemma_host:
        settings.gemma_host = payload.gemma_host
    if payload.translation_engine and payload.translation_engine != settings.translation_engine:
        settings.translation_engine = payload.translation_engine
        engine_changed = True
    if payload.tts_engine and payload.tts_engine != settings.tts_engine:
        settings.tts_engine = payload.tts_engine
        engine_changed = True

    if engine_changed:
        await pipeline.restart_engines()

    await manager.broadcast_text("op", {
        "type": "settings_updated",
        "settings": {
            "whisper_model":      settings.whisper_model,
            "source_language":    settings.source_language,
            "translation_engine": settings.translation_engine,
            "tts_engine":         settings.tts_engine,
            "tts_speed":          settings.tts_speed,
        },
    })
    return {"ok": True}

# ─── STT Start / Stop ────────────────────────────────────────────────────────
class StartPayload(BaseModel):
    whisper_binary: Optional[str] = None

@app.post("/api/operator/start")
async def start_stt(payload: StartPayload):
    if whisper.is_running:
        return {"ok": True, "message": "Already running"}

    binary     = payload.whisper_binary or WHISPER_STREAM_BIN
    model_file = _resolve_whisper_model(settings.whisper_model)
    if not model_file:
        raise HTTPException(
            400,
            f"Whisper model '{settings.whisper_model}' not found. "
            f"Download the .bin file into: {MODELS_DIR}",
        )

    await whisper.start(
        binary=binary,
        model_path=model_file,
        language=settings.source_language,
        mic_index=settings.microphone_index,
        on_segment=pipeline.on_segment,
    )
    await manager.broadcast_text("op", {"type": "stt_started", "model": settings.whisper_model})
    return {"ok": True, "model": settings.whisper_model}

@app.post("/api/operator/stop")
async def stop_stt():
    await whisper.stop()
    await manager.broadcast_text("op", {"type": "stt_stopped"})
    return {"ok": True}

def _resolve_whisper_model(model_name: str) -> Optional[str]:
    candidates = [
        Path(MODELS_DIR) / f"ggml-{model_name}.bin",
        Path(MODELS_DIR) / f"{model_name}.bin",
        Path(MODELS_DIR) / model_name,
        Path(WHISPER_DIR) / "models" / f"ggml-{model_name}.bin",
        Path(WHISPER_DIR) / "models" / f"{model_name}.bin",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None

# ─── WebSocket: Audience listener ────────────────────────────────────────────
@app.websocket("/ws/listen/{lang}")
async def ws_listen(ws: WebSocket, lang: str):
    if lang not in LANGUAGES:
        await ws.close(code=4000, reason=f"Unknown language: {lang}")
        return

    await manager.connect(ws, lang)
    await manager.broadcast_text("op", {
        "type":        "listener_joined",
        "lang":        lang,
        "connections": manager.counts(),
    })
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws, lang)
        await manager.broadcast_text("op", {
            "type":        "listener_left",
            "lang":        lang,
            "connections": manager.counts(),
        })

# ─── WebSocket: Operator ──────────────────────────────────────────────────────
@app.websocket("/ws/operator")
async def ws_operator(ws: WebSocket):
    await manager.connect(ws, "op")
    # Send full current state immediately
    await ws.send_text(json.dumps({
        "type":    "init",
        "running": whisper.is_running,
        "settings": {
            "whisper_model":      settings.whisper_model,
            "source_language":    settings.source_language,
            "translation_engine": settings.translation_engine,
            "tts_engine":         settings.tts_engine,
            "tts_speed":          settings.tts_speed,
        },
        "connections": manager.counts(),
    }))
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({
                    "type":        "pong",
                    "connections": manager.counts(),
                }))
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws, "op")

# ─── Entry point ──────────────────────────────────────────────────────────────
def run():
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )

if __name__ == "__main__":
    run()
