"""
Live Translate — Configuration
Central config for languages, models, TTS engines, and translation engines.
"""
from dataclasses import dataclass, field
import os

# ─── Supported Languages ───────────────────────────────────────────────────────
LANGUAGES = {
    "en": {"name": "English",    "flag": "🇬🇧", "piper_voice": "en_US-lessac-medium"},
    "nl": {"name": "Dutch",      "flag": "🇳🇱", "piper_voice": "nl_NL-mls-medium"},
    "es": {"name": "Spanish",    "flag": "🇪🇸", "piper_voice": "es_ES-mls-medium"},
    "pt": {"name": "Portuguese", "flag": "🇵🇹", "piper_voice": "pt_BR-edresson-low"},
    "fr": {"name": "French",     "flag": "🇫🇷", "piper_voice": "fr_FR-mls-medium"},
    "de": {"name": "German",     "flag": "🇩🇪", "piper_voice": "de_DE-thorsten-medium"},
    "it": {"name": "Italian",    "flag": "🇮🇹", "piper_voice": "it_IT-riccardo-x_low"},
    "uk": {"name": "Ukrainian",  "flag": "🇺🇦", "piper_voice": "uk_UA-ukrainian_tts-medium"},
}

# Languages supported natively by Supertonic TTS (falls back to Piper for others)
SUPERTONIC_LANGUAGES = {"es", "pt", "fr"}

# ─── Source Language Options ───────────────────────────────────────────────────
SOURCE_LANGUAGES = ["en", "nl", "es"]   # en=default, nl=secondary, es=third

# ─── Whisper Models ────────────────────────────────────────────────────────────
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR    = os.path.join(BASE_DIR, "voices")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
WHISPER_DIR   = os.path.join(BASE_DIR, "whisper.cpp")
FRONTEND_DIR  = os.path.join(BASE_DIR, "frontend")

PIPER_BIN     = os.environ.get("PIPER_BIN", "piper")
PARKIET_BIN   = os.environ.get("PARKIET_BIN", "parkiet")
WHISPER_STREAM_BIN = os.environ.get(
    "WHISPER_STREAM_BIN",
    os.path.join(WHISPER_DIR, "build", "bin", "whisper-stream"),
)

# ─── Runtime Settings (mutated live by the operator API) ──────────────────────
@dataclass
class AppSettings:
    # STT
    whisper_model:      str   = "base"
    source_language:    str   = "en"
    microphone_index:   int   = 0

    # Translation
    translation_engine: str   = "argos"            # "argos" | "gemma4b" | "gemma12b"
    gemma_host:         str   = "http://localhost:11434"

    # TTS
    tts_engine:         str   = "piper"            # "piper" | "parkiet" | "supertonic"
    tts_speed:          float = 1.0                # 0.5 – 2.0

    # Per-language voice overrides  lang_code -> filename inside voices/<lang>/
    voice_overrides:    dict  = field(default_factory=dict)

    # Server
    host:               str   = "0.0.0.0"
    port:               int   = 8765
    max_connections:    int   = 1000


# Singleton shared by the entire app
settings = AppSettings()
