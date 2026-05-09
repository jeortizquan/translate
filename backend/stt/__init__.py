"""STT sub-package — whisper.cpp and faster-whisper engines."""
from .whisper_manager import WhisperStreamManager
from .faster_whisper_manager import FasterWhisperManager, list_microphones

__all__ = ["WhisperStreamManager", "FasterWhisperManager", "list_microphones"]
