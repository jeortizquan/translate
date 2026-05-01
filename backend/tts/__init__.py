"""TTS sub-package."""
from .base import BaseTTS
from .piper_engine import PiperTTS
from .parkiet_engine import ParkietTTS
from .supertonic_engine import SupertonicTTS


def make_tts(engine: str, voices_dir: str) -> BaseTTS:
    """
    Factory — returns the right TTS engine for the given key.

    Accepted values for *engine*:
      "piper"      → PiperTTS      (all 8 languages, Python library, no binary)
      "parkiet"    → ParkietTTS    (all 8 languages, requires parkiet binary)
      "supertonic" → SupertonicTTS (es/pt/fr native; Piper fallback for others)
    """
    if engine == "piper":
        return PiperTTS(voices_dir=voices_dir)
    if engine == "parkiet":
        return ParkietTTS(voices_dir=voices_dir)
    if engine == "supertonic":
        return SupertonicTTS(voices_dir=voices_dir)
    raise ValueError(
        f"Unknown TTS engine: {engine!r}. "
        "Valid options: 'piper', 'parkiet', 'supertonic'"
    )


__all__ = ["BaseTTS", "PiperTTS", "ParkietTTS", "SupertonicTTS", "make_tts"]
