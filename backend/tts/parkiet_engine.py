"""
TTS Engine — Parkiet
https://github.com/pevers/parkiet
Offline TTS — all required languages.

Binary:  place on PATH or set PARKIET_BIN env var.
Models:  put files in  voices/parkiet_<lang>/
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from .base import BaseTTS
from .utils import pcm_to_wav
from ..config import PARKIET_BIN, VOICES_DIR

logger = logging.getLogger(__name__)


class ParkietTTS(BaseTTS):
    name = "parkiet"

    def __init__(self, voices_dir: str = VOICES_DIR, binary: str = PARKIET_BIN):
        self._voices_dir = Path(voices_dir)
        self._binary     = binary

    def _resolve_model(self, lang: str, voice_override: Optional[str]) -> Optional[str]:
        lang_dir = self._voices_dir / f"parkiet_{lang}"
        if not lang_dir.exists():
            lang_dir = self._voices_dir / lang

        if voice_override:
            p = Path(voice_override)
            if p.exists():
                return str(p)
            c = lang_dir / voice_override
            if c.exists():
                return str(c)

        if lang_dir.exists():
            for pattern in ("*.bin", "*.onnx", "*.pt"):
                files = sorted(lang_dir.rglob(pattern))
                if files:
                    return str(files[0])

        logger.warning("Parkiet: no model found for lang=%r", lang)
        return None

    def available_voices(self, lang: str) -> list:
        lang_dir = self._voices_dir / f"parkiet_{lang}"
        if lang_dir.exists():
            return [f.name for f in sorted(lang_dir.iterdir()) if f.is_file()]
        return []

    async def synthesize(
        self,
        text: str,
        lang: str,
        speed: float = 1.0,
        voice_override: Optional[str] = None,
    ) -> bytes:
        model_path = self._resolve_model(lang, voice_override)
        if not model_path:
            logger.error("Parkiet: no model for [%s]", lang)
            return b""

        cmd = [
            self._binary,
            "--model",         model_path,
            "--lang",          lang,
            "--speed",         str(round(speed, 2)),
            "--output-format", "raw",
            "--text",          text,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                logger.error("Parkiet error [%s]: %s", lang, stderr.decode()[:300])
                return b""
            return pcm_to_wav(stdout)
        except asyncio.TimeoutError:
            logger.error("Parkiet timeout [%s]", lang)
            return b""
        except FileNotFoundError:
            logger.error("Parkiet binary not found: %r — set PARKIET_BIN env var", self._binary)
            return b""
        except Exception as exc:
            logger.error("Parkiet exception [%s]: %s", lang, exc)
            return b""
