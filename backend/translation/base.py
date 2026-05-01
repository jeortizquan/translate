"""Abstract base class for all translation engines."""
from abc import ABC, abstractmethod


class BaseTranslator(ABC):
    name: str = "base"

    @abstractmethod
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate *text* from *source_lang* to *target_lang*. Return translated string."""
        ...

    async def setup(self) -> None:
        """Optional async initialisation (download packages, warm-up, etc.)."""

    async def teardown(self) -> None:
        """Optional cleanup."""
