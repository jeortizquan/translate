"""
Live Translate — FastAPI Application

HTTP routes:
  GET  /                           → redirect to /client
  GET  /client                     → audience portal (HTML)
  GET  /operator                   → operator panel (HTML)
  GET  /api/status                 → server status JSON
  GET  /api/microphones            → list audio input devices
  GET  /api/voices                 → all available voices (keyed by lang)
  GET  /api/voices/{lang}          → available voices for one language
  POST /api/operator/settings      → update runtime settings live
  POST /api/operator/start         → start STT (whisper.cpp or faster-whisper)
  POST /api/operator/stop          → stop STT
  GET  /api/filter                 → get current word lists
  POST /api/filter/add             → add a custom word
  POST /api/filter/remove          → remove a custom word
  PUT  /api/filter/custom          → replace full custom word list

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
from .filter import profanity_filter
from .config import (
    LANGUAGES, WHISPER_MODELS, SOURCE_LANGUAGES, STT_ENGINES,
    settings, VOICES_DIR, MODELS_DIR, WHISPER_STREAM_BIN, WHISPER_DIR, FRONTEND_DIR,
)
from .pipeline import pipeline
from .stt import WhisperStreamManager, FasterWhisperManager, list_microphones
from .tts import make_tts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── STT engine instances ──────────────────────────────────────────────────────
_whisper_cpp    = WhisperStreamManager()
_faster_whisper = FasterWhisperManager()


def _active_stt():
    """Return whichever STT manager is currently selected."""
    if settings.stt_engine == "faster-whisper":
        return _faster_whisper
    return _whisper_cpp


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Live Translate starting ===")
    await pipeline.start()
    yield
    logger.info("=== Live Translate shutting down ===")
    await _whisper_cpp.stop()
    await _faster_whisper.stop()
    await pipeline.stop()

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
    stt = _active_stt()
    return {
        "running":             stt.is_running,
        "stt_engine":          settings.stt_engine,
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
        "stt_engines":         STT_ENGINES,
        "voice_overrides":     settings.voice_overrides,
        "fw_device":           settings.fw_device,
        "fw_compute_type":     settings.fw_compute_type,
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
    stt_engine:         Optional[str]   = None
    whisper_model:      Optional[str]   = None
    source_language:    Optional[str]   = None
    microphone_index:   Optional[int]   = None
    translation_engine: Optional[str]   = None
    tts_engine:         Optional[str]   = None
    tts_speed:          Optional[float] = None
    voice_overrides:    Optional[dict]  = None
    gemma_host:         Optional[str]   = None
    fw_device:          Optional[str]   = None
    fw_compute_type:    Optional[str]   = None
    fw_vad_threshold:   Optional[float] = None
    fw_silence_ms:      Optional[int]   = None

@app.post("/api/operator/settings")
async def update_settings(payload: SettingsPayload):
    engine_changed = False

    if payload.stt_engine and payload.stt_engine in STT_ENGINES:
        settings.stt_engine = payload.stt_engine
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
    if payload.fw_device:
        settings.fw_device = payload.fw_device
    if payload.fw_compute_type:
        settings.fw_compute_type = payload.fw_compute_type
    if payload.fw_vad_threshold is not None:
        settings.fw_vad_threshold = max(0.001, min(0.5, payload.fw_vad_threshold))
    if payload.fw_silence_ms is not None:
        settings.fw_silence_ms = max(200, min(3000, payload.fw_silence_ms))

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
        "settings": _current_settings(),
    })
    return {"ok": True}

def _current_settings() -> dict:
    return {
        "stt_engine":          settings.stt_engine,
        "whisper_model":       settings.whisper_model,
        "source_language":     settings.source_language,
        "translation_engine":  settings.translation_engine,
        "tts_engine":          settings.tts_engine,
        "tts_speed":           settings.tts_speed,
        "fw_device":           settings.fw_device,
        "fw_compute_type":     settings.fw_compute_type,
        "fw_vad_threshold":    settings.fw_vad_threshold,
        "fw_silence_ms":       settings.fw_silence_ms,
    }

# ─── STT Start / Stop ────────────────────────────────────────────────────────
class StartPayload(BaseModel):
    whisper_binary: Optional[str] = None

@app.post("/api/operator/start")
async def start_stt(payload: StartPayload):
    stt = _active_stt()
    if stt.is_running:
        return {"ok": True, "message": "Already running"}

    if settings.stt_engine == "faster-whisper":
        # Configure VAD parameters from settings
        _faster_whisper.vad_energy_threshold = settings.fw_vad_threshold
        _faster_whisper.vad_silence_ms       = settings.fw_silence_ms

        await _faster_whisper.start(
            model_name=settings.whisper_model,
            language=settings.source_language,
            mic_index=settings.microphone_index,
            on_segment=pipeline.on_segment,
        )
        await manager.broadcast_text("op", {
            "type": "stt_started",
            "engine": "faster-whisper",
            "model": settings.whisper_model,
        })
        return {"ok": True, "engine": "faster-whisper", "model": settings.whisper_model}

    else:
        # whisper.cpp / whisper-stream
        binary     = payload.whisper_binary or WHISPER_STREAM_BIN
        model_file = _resolve_whisper_model(settings.whisper_model)
        if not model_file:
            raise HTTPException(
                400,
                f"Whisper model '{settings.whisper_model}' not found. "
                f"Download the .bin file into: {MODELS_DIR}",
            )
        await _whisper_cpp.start(
            binary=binary,
            model_path=model_file,
            language=settings.source_language,
            mic_index=settings.microphone_index,
            on_segment=pipeline.on_segment,
        )
        await manager.broadcast_text("op", {
            "type": "stt_started",
            "engine": "whisper.cpp",
            "model": settings.whisper_model,
        })
        return {"ok": True, "engine": "whisper.cpp", "model": settings.whisper_model}

@app.post("/api/operator/stop")
async def stop_stt():
    await _whisper_cpp.stop()
    await _faster_whisper.stop()
    await pipeline._coalescer.flush_now()
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

# ─── Filter API ───────────────────────────────────────────────────────────────
@app.get("/api/filter")
async def api_filter_get():
    return profanity_filter.list_words()

class FilterWordPayload(BaseModel):
    word: str

class FilterListPayload(BaseModel):
    words: list[str]

@app.post("/api/filter/add")
async def api_filter_add(payload: FilterWordPayload):
    added = profanity_filter.add_custom_word(payload.word)
    return {"ok": True, "added": added, "word": payload.word.strip().lower()}

@app.post("/api/filter/remove")
async def api_filter_remove(payload: FilterWordPayload):
    removed = profanity_filter.remove_custom_word(payload.word)
    return {"ok": True, "removed": removed, "word": payload.word.strip().lower()}

@app.put("/api/filter/custom")
async def api_filter_save(payload: FilterListPayload):
    profanity_filter.save_custom_words(payload.words)
    return {"ok": True, "count": len(payload.words)}

# ─── WebSocket: Audience listener ────────────────────────────────────────────
@app.websocket("/ws/listen/{lang}")
async def ws_listen(ws: WebSocket, lang: str):
    if lang not in LANGUAGES:
        await ws.close(code=4000, reason=f"Unknown language: {lang}")
        return
    await manager.connect(ws, lang)
    await manager.broadcast_text("op", {
        "type": "listener_joined", "lang": lang, "connections": manager.counts(),
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
            "type": "listener_left", "lang": lang, "connections": manager.counts(),
        })

# ─── WebSocket: Operator ──────────────────────────────────────────────────────
@app.websocket("/ws/operator")
async def ws_operator(ws: WebSocket):
    await manager.connect(ws, "op")
    stt = _active_stt()
    await ws.send_text(json.dumps({
        "type":        "init",
        "running":     stt.is_running,
        "settings":    _current_settings(),
        "connections": manager.counts(),
    }))
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({
                    "type": "pong", "connections": manager.counts(),
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
