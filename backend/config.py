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

# Languages supported natively by Supertonic TTS
SUPERTONIC_LANGUAGES = {"es", "pt", "fr"}

# ─── Source Language Options ───────────────────────────────────────────────────
SOURCE_LANGUAGES = ["en", "nl", "es"]

# ─── STT Engines ──────────────────────────────────────────────────────────────
STT_ENGINES = [
    "whisper.cpp",      # subprocess-based, uses whisper-stream binary + Metal GPU
    "faster-whisper",   # Python library, CPU INT8 on macOS / CUDA on Linux
]

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

# ─── Runtime Settings ──────────────────────────────────────────────────────────
@dataclass
class AppSettings:
    # STT
    stt_engine:         str   = "whisper.cpp"  # "whisper.cpp" | "faster-whisper"
    whisper_model:      str   = "base"
    source_language:    str   = "en"
    microphone_index:   int   = 0

    # faster-whisper specific
    fw_device:          str   = "auto"         # "auto" | "cpu" | "cuda"
    fw_compute_type:    str   = "int8"         # "int8" | "float16" | "auto"
    fw_vad_threshold:   float = 0.01           # energy VAD threshold (0.0–1.0)
    fw_silence_ms:      int   = 700            # ms of silence to end speech segment

    # Translation
    translation_engine: str   = "argos"
    gemma_host:         str   = "http://localhost:11434"

    # TTS
    tts_engine:         str   = "piper"
    tts_speed:          float = 1.0
    voice_overrides:    dict  = field(default_factory=dict)

    # Server
    host:               str   = "0.0.0.0"
    port:               int   = 8765
    max_connections:    int   = 1000


settings = AppSettings()
