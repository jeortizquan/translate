"""Abstract base class for all TTS engines."""
from abc import ABC, abstractmethod
from typing import Optional


class BaseTTS(ABC):
    name: str = "base"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        lang: str,
        speed: float = 1.0,
        voice_override: Optional[str] = None,
    ) -> bytes:
        """Return raw WAV bytes for the given text."""
        ...

    async def setup(self) -> None:
        """Optional async initialisation."""

    async def teardown(self) -> None:
        """Optional cleanup."""

    def available_voices(self, lang: str) -> list:
        """Return a list of available voice filenames for *lang*."""
        return []
