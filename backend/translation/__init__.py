"""Translation sub-package."""
from .base import BaseTranslator
from .argos_engine import ArgosTranslator
from .gemma_engine import GemmaTranslator


def make_translator(engine: str, ollama_host: str = "http://localhost:11434") -> BaseTranslator:
    """
    Factory — returns the right translator for the given engine key.

    Accepted values for *engine*:
      "argos"    → ArgosTranslator  (fully offline, no GPU required)
      "gemma4b"  → GemmaTranslator  (Gemma 3 4B via Ollama)
      "gemma12b" → GemmaTranslator  (Gemma 3 12B via Ollama)
    """
    if engine == "argos":
        return ArgosTranslator()
    if engine in ("gemma4b", "gemma12b"):
        return GemmaTranslator(model_key=engine, ollama_host=ollama_host)
    raise ValueError(
        f"Unknown translation engine: {engine!r}. "
        "Valid options: 'argos', 'gemma4b', 'gemma12b'"
    )


__all__ = [
    "BaseTranslator",
    "ArgosTranslator",
    "GemmaTranslator",
    "make_translator",
]
