"""
TTS Engine — Supertonic 1.1.2
==============================
pip install supertonic

Confirmed API (supertonic 1.1.2):
  tts = TTS(model='supertonic-2')
  style = tts.get_voice_style_from_path("voices/supertonic/preset_voice_m.json")
  wav, _ = tts.synthesize(text, voice_style=style, speed=1.05, lang='en')
  # wav is a numpy float32/int16 array → convert to int16 PCM → wrap in WAV

Voice layout:
  voices/
  └── supertonic/
      ├── preset_voice_m.json   ← male voice style
      ├── preset_voice_f.json   ← female voice style
      └── ...                   ← any additional .json styles

JSON files are voice STYLES (not model weights). They work with any language.
The model itself is downloaded automatically by supertonic on first use.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from .base import BaseTTS
from .piper_engine import PiperTTS
from .utils import pcm_to_wav
from ..config import VOICES_DIR, LANGUAGES

logger = logging.getLogger(__name__)

# Folder containing supertonic .json voice style files
_ST_SUBDIR = "supertonic"

# Sample rate used by supertonic-2 model (22050 Hz standard)
_SAMPLE_RATE = 22050

# Languages supertonic supports — from AVAILABLE_LANGUAGES constant
# Covers all 8 of our target languages
_SUPERTONIC_LANGS = {"en", "nl", "es", "pt", "fr", "de", "it", "uk"}


# ── Module-level singletons ───────────────────────────────────────────────────
# One TTS instance shared across all calls (model is large, load once)
_tts_instance   = None
_tts_lock       = None   # asyncio.Lock, created lazily

# Style cache: json_path -> Style object
_style_cache: dict[str, object] = {}


def _get_lock():
    global _tts_lock
    if _tts_lock is None:
        _tts_lock = asyncio.Lock()
    return _tts_lock


# ── Supertonic initialisation ─────────────────────────────────────────────────

def _init_tts_sync(model: str = "supertonic-2", model_dir: Optional[str] = None):
    """Load the TTS model (blocking — runs in thread). Returns TTS instance."""
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance

    from supertonic import TTS
    logger.info("Supertonic: loading model '%s' …", model)
    kwargs = {"model": model}
    if model_dir:
        kwargs["model_dir"] = model_dir
    _tts_instance = TTS(**kwargs)
    logger.info("Supertonic: model ready")
    return _tts_instance


def _load_style_sync(json_path: str):
    """Load a voice Style from a JSON file (blocking). Returns Style object."""
    if json_path in _style_cache:
        return _style_cache[json_path]

    tts = _tts_instance
    if tts is None:
        raise RuntimeError("Supertonic TTS not initialised — call _init_tts_sync first")

    logger.info("Supertonic: loading voice style %s …", Path(json_path).name)
    style = tts.get_voice_style_from_path(json_path)
    _style_cache[json_path] = style
    logger.info("Supertonic: style cached — %s", Path(json_path).name)
    return style


# ── Core synthesis ────────────────────────────────────────────────────────────

def _synthesize_sync(
        json_path: str,
        text: str,
        lang: str,
        speed: float,
        total_steps: int,
        silence_duration: float,
) -> bytes:
    """Blocking synthesis — runs in asyncio.to_thread()."""
    tts   = _tts_instance
    style = _load_style_sync(json_path)

    # synthesize() returns (wav: np.ndarray, something_else: np.ndarray)
    # wav is the audio — float32 in range [-1, 1] or already int16
    wav, _ = tts.synthesize(
        text            = text,
        voice_style     = style,
        speed           = speed,
        lang            = lang,
        total_steps     = total_steps,
        silence_duration= silence_duration,
        verbose         = False,
    )

    # Convert numpy array → int16 PCM bytes
    import numpy as np
    if wav.dtype != np.int16:
        # Float32 normalised [-1, 1] → int16
        wav = (wav * 32767).clip(-32768, 32767).astype(np.int16)

    # Flatten in case it's (samples, 1) shaped
    pcm = wav.flatten().tobytes()

    # Try to read sample rate from the TTS instance
    sr = _SAMPLE_RATE
    for attr in ("sample_rate", "sampling_rate", "sr"):
        val = getattr(tts, attr, None)
        if isinstance(val, int) and val > 0:
            sr = val
            break

    wav_bytes = pcm_to_wav(pcm, sample_rate=sr, channels=1, sample_width=2)
    logger.debug(
        "Supertonic: %d PCM bytes → %d WAV bytes @ %d Hz [%s] %r",
        len(pcm), len(wav_bytes), sr, lang, text[:40],
    )
    return wav_bytes


# ── SupertonicTTS class ───────────────────────────────────────────────────────

class SupertonicTTS(BaseTTS):
    name = "supertonic"

    def __init__(
            self,
            voices_dir: str = VOICES_DIR,
            model: str = "supertonic-2",
            model_dir: Optional[str] = None,
            total_steps: int = 5,
            silence_duration: float = 0.3,
    ):
        self._voices_dir      = Path(voices_dir)
        self._st_dir          = self._voices_dir / _ST_SUBDIR
        self._model           = model
        self._model_dir       = model_dir
        self._total_steps     = total_steps
        self._silence_duration= silence_duration
        self._fallback        = PiperTTS(voices_dir=voices_dir)
        self._default_json:   Optional[str] = None

    # ── Voice style discovery ─────────────────────────────────────────────────

    def _find_json_files(self) -> list[Path]:
        """Return all .json voice style files in voices/supertonic/."""
        if not self._st_dir.exists():
            return []
        return sorted(self._st_dir.glob("*.json"))

    def _resolve_json(self, voice_override: Optional[str] = None) -> Optional[str]:
        """
        Priority:
          1. voice_override (filename inside voices/supertonic/ or absolute path)
          2. Cached default (first JSON found on setup)
          3. First JSON found now
        """
        if voice_override:
            p = Path(voice_override)
            if p.is_absolute() and p.exists():
                return str(p)
            candidate = self._st_dir / voice_override
            if candidate.exists():
                return str(candidate)

        if self._default_json:
            return self._default_json

        jsons = self._find_json_files()
        if jsons:
            self._default_json = str(jsons[0])
            return self._default_json

        logger.warning(
            "Supertonic: no JSON voice files found in %s. "
            "Place preset_voice_m.json / preset_voice_f.json there.",
            self._st_dir,
        )
        return None

    def available_voices(self, lang: str = "") -> list:
        """Return names of all discovered JSON voice style files."""
        return [f.name for f in self._find_json_files()]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Pre-load the supertonic model and default voice style."""
        # Always set up Piper fallback too
        await self._fallback.setup()

        try:
            from supertonic import TTS, AVAILABLE_LANGUAGES  # noqa: F401
        except ImportError as exc:
            logger.error(
                "Supertonic not installed — will use Piper for all languages. %s", exc
            )
            return

        jsons = self._find_json_files()
        if not jsons:
            logger.warning(
                "Supertonic: no JSON files in %s — falling back to Piper. "
                "Place supertonic preset JSON files in that folder.",
                self._st_dir,
            )
            return

        logger.info(
            "Supertonic: found %d voice style(s): %s",
            len(jsons), [f.name for f in jsons],
        )

        # Load model in thread (may auto-download on first run)
        try:
            await asyncio.to_thread(
                _init_tts_sync, self._model, self._model_dir
            )
        except Exception as exc:
            logger.error("Supertonic: model load failed: %s", exc)
            return

        # Pre-load the default (first) voice style
        self._default_json = str(jsons[0])
        try:
            await asyncio.to_thread(_load_style_sync, self._default_json)
            logger.info("Supertonic: ✓ default voice — %s", jsons[0].name)
        except Exception as exc:
            logger.warning("Supertonic: could not pre-load voice style: %s", exc)

    async def teardown(self) -> None:
        global _tts_instance, _style_cache
        _tts_instance = None
        _style_cache.clear()

    # ── Synthesis ─────────────────────────────────────────────────────────────

    async def synthesize(
            self,
            text: str,
            lang: str,
            speed: float = 1.0,
            voice_override: Optional[str] = None,
    ) -> bytes:
        text = text.strip()
        if not text:
            return b""

        # Fall back to Piper for unsupported languages or if model not loaded
        if lang not in _SUPERTONIC_LANGS or _tts_instance is None:
            logger.debug(
                "Supertonic: using Piper fallback for [%s] (model_ready=%s)",
                lang, _tts_instance is not None,
                      )
            return await self._fallback.synthesize(text, lang, speed, voice_override)

        json_path = self._resolve_json(voice_override)
        if not json_path:
            logger.warning("Supertonic: no voice style — using Piper for [%s]", lang)
            return await self._fallback.synthesize(text, lang, speed, voice_override)

        logger.debug(
            "Supertonic [%s] speed=%.2f voice=%s %r",
            lang, speed, Path(json_path).name, text[:60],
        )
        try:
            result = await asyncio.to_thread(
                _synthesize_sync,
                json_path, text, lang, speed,
                self._total_steps, self._silence_duration,
            )
            if result:
                return result
            logger.warning("Supertonic: empty result for [%s] — using Piper", lang)
            return await self._fallback.synthesize(text, lang, speed)
        except Exception as exc:
            logger.error("Supertonic error [%s]: %s — using Piper", lang, exc)
            return await self._fallback.synthesize(text, lang, speed)