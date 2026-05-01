"""
TTS Engine — Piper 1.4.x  (piper1-gpl / piper-tts)
=====================================================
pip install piper-tts   (or the OHF-Voice fork)

Confirmed API for piper 1.4.2:
  voice.synthesize(text, syn_config=SynthesisConfig(...)) -> Iterable[AudioChunk]

AudioChunk has:
  .audio        numpy int16 array   (raw PCM samples)
  .sample_rate  int                 (Hz, e.g. 22050)

SynthesisConfig controls speed via length_scale (>1 = slower, <1 = faster).
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from .base import BaseTTS
from .utils import pcm_to_wav
from ..config import LANGUAGES, VOICES_DIR

logger = logging.getLogger(__name__)

# Module-level cache: onnx_path -> PiperVoice
_voice_cache: dict[str, object] = {}


# ── Piper imports ─────────────────────────────────────────────────────────────

def _import_piper():
    """Return (PiperVoice, SynthesisConfig) classes."""
    try:
        from piper.voice import PiperVoice
        from piper.config import SynthesisConfig
        return PiperVoice, SynthesisConfig
    except ImportError:
        pass
    try:
        from piper import PiperVoice
        from piper import SynthesisConfig
        return PiperVoice, SynthesisConfig
    except ImportError:
        pass
    raise ImportError(
        "piper-tts 1.4.x is not installed.\n"
        "Run:  pip install piper-tts\n"
        "Or:   pip install git+https://github.com/OHF-Voice/piper1-gpl.git"
    )


def _load_voice(onnx_path: str):
    if onnx_path in _voice_cache:
        return _voice_cache[onnx_path]
    PiperVoice, _ = _import_piper()
    logger.info("Piper: loading %s …", Path(onnx_path).name)
    voice = PiperVoice.load(onnx_path)
    _voice_cache[onnx_path] = voice
    logger.info("Piper: cached — %s", Path(onnx_path).name)
    return voice


# ── AudioChunk → PCM bytes ────────────────────────────────────────────────────

def _chunk_to_pcm(chunk) -> tuple[bytes, int]:
    """
    Extract (raw_pcm_bytes, sample_rate) from an AudioChunk.

    AudioChunk attributes (piper 1.4.x):
      .audio        numpy int16 ndarray
      .sample_rate  int

    Falls back to attribute scanning if the structure ever changes.
    """
    # Primary path — piper 1.4.x
    audio = getattr(chunk, "audio", None)
    sr    = getattr(chunk, "sample_rate", None)

    if audio is None:
        # Scan all attributes for a numpy array
        import numpy as np
        for attr in vars(chunk) if hasattr(chunk, "__dict__") else []:
            val = getattr(chunk, attr)
            if isinstance(val, np.ndarray):
                audio = val
                break

    if audio is None:
        logger.warning("Piper: could not extract audio from AudioChunk %s", type(chunk))
        return b"", sr or 22050

    # Ensure int16
    try:
        import numpy as np
        if audio.dtype != np.int16:
            audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        return audio.tobytes(), int(sr) if sr else 22050
    except Exception as exc:
        logger.error("Piper: audio conversion failed: %s", exc)
        return b"", 22050


# ── Blocking synthesis (runs in thread pool) ──────────────────────────────────

def _synthesize_sync(onnx_path: str, text: str, speed: float) -> bytes:
    """
    Synthesise text using piper 1.4.x AudioChunk generator API.
    Returns complete WAV bytes ready to send to the browser.
    """
    PiperVoice, SynthesisConfig = _import_piper()
    voice = _load_voice(onnx_path)

    # length_scale: 1.0 = normal, >1 = slower, <1 = faster
    length_scale = round(1.0 / max(speed, 0.1), 3)

    try:
        syn_config = SynthesisConfig(length_scale=length_scale)
    except TypeError:
        # Older SynthesisConfig may not accept length_scale
        try:
            syn_config = SynthesisConfig()
        except Exception:
            syn_config = None

    pcm_chunks:  list[bytes] = []
    sample_rate: int         = 22050

    try:
        gen = voice.synthesize(text, syn_config=syn_config) if syn_config is not None \
            else voice.synthesize(text)

        for chunk in gen:
            pcm, sr = _chunk_to_pcm(chunk)
            if pcm:
                pcm_chunks.append(pcm)
                sample_rate = sr   # use rate from last chunk (all chunks share same rate)

    except Exception as exc:
        logger.error("Piper: synthesis failed for %r: %s", text[:50], exc)
        return b""

    if not pcm_chunks:
        logger.warning("Piper: no audio chunks produced for %r", text[:50])
        return b""

    raw_pcm = b"".join(pcm_chunks)
    wav     = pcm_to_wav(raw_pcm, sample_rate=sample_rate, channels=1, sample_width=2)

    logger.debug(
        "Piper: %d PCM bytes → %d WAV bytes @ %d Hz for %r",
        len(raw_pcm), len(wav), sample_rate, text[:40],
    )
    return wav


# ── PiperTTS class ────────────────────────────────────────────────────────────

class PiperTTS(BaseTTS):
    name = "piper"

    def __init__(self, voices_dir: str = VOICES_DIR):
        self._voices_dir = Path(voices_dir)

    # ── Voice discovery ───────────────────────────────────────────────────────

    def _find_onnx_files(self, lang: str) -> list[Path]:
        lang_dir = self._voices_dir / lang
        if not lang_dir.exists():
            return []
        return sorted(lang_dir.rglob("*.onnx"))

    def _resolve_voice(self, lang: str, voice_override: Optional[str] = None) -> Optional[str]:
        if voice_override:
            p = Path(voice_override)
            if p.is_absolute() and p.exists():
                return str(p)
            lang_dir = self._voices_dir / lang
            if lang_dir.exists():
                hits = sorted(lang_dir.rglob(voice_override))
                if hits:
                    return str(hits[0])

        files = self._find_onnx_files(lang)
        if files:
            return str(files[0])

        default = LANGUAGES.get(lang, {}).get("piper_voice", "")
        if default:
            flat = self._voices_dir / f"{default}.onnx"
            if flat.exists():
                return str(flat)

        logger.warning(
            "Piper: no .onnx for lang=%r — place a model in voices/%s/",
            lang, lang,
        )
        return None

    def available_voices(self, lang: str) -> list:
        lang_dir = self._voices_dir / lang
        if not lang_dir.exists():
            return []
        return [str(f.relative_to(lang_dir)) for f in self._find_onnx_files(lang)]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        try:
            _import_piper()
        except ImportError as exc:
            logger.error("piper-tts NOT installed — audio will be silent.\n%s", exc)
            return

        logger.info("Piper: pre-loading voice models …")
        for lang in LANGUAGES:
            path = self._resolve_voice(lang)
            if path:
                try:
                    await asyncio.to_thread(_load_voice, path)
                    logger.info("Piper: ✓ [%s] — %s", lang, Path(path).name)
                except Exception as exc:
                    logger.warning("Piper: could not load [%s]: %s", lang, exc)
            else:
                logger.warning("Piper: no model for [%s] — will be silent", lang)

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

        path = self._resolve_voice(lang, voice_override)
        if not path:
            logger.error("Piper: cannot synthesise [%s] — no model found", lang)
            return b""

        logger.debug("Piper [%s] speed=%.2f %r", lang, speed, text[:60])
        try:
            return await asyncio.to_thread(_synthesize_sync, path, text, speed)
        except Exception as exc:
            logger.error("Piper synthesis error [%s]: %s", lang, exc)
            return b""