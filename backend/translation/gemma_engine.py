"""
Translation Engine — Gemma 3 4B / 12B via Ollama
Runs locally via Ollama (https://ollama.com).
  ollama pull gemma3:4b    (~2.3 GB)
  ollama pull gemma3:12b   (~7 GB)
"""
import asyncio
import logging
import re
from typing import Optional

import httpx

from .base import BaseTranslator

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a professional real-time interpreter. "
    "Translate the user message from {src} to {tgt}. "
    "Output ONLY the translated text — no explanation, no quotes, no preamble."
)

_MODEL_MAP = {
    "gemma4b":  "gemma4:e4b",
    "gemma12b": "gemma3:12b",
}

_LANG_NAMES = {
    "en": "English", "nl": "Dutch",     "es": "Spanish",    "pt": "Portuguese",
    "fr": "French",  "de": "German",    "it": "Italian",    "uk": "Ukrainian",
}


class GemmaTranslator(BaseTranslator):
    def __init__(self, model_key: str = "gemma4b", ollama_host: str = "http://localhost:11434"):
        self.name = model_key
        self._model = _MODEL_MAP.get(model_key, "gemma4:e4b")
        self._host  = ollama_host.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def setup(self) -> None:
        self._client = httpx.AsyncClient(base_url=self._host, timeout=60)
        try:
            resp = await self._client.get("/api/tags")
            names = [m["name"] for m in resp.json().get("models", [])]
            if any(self._model in n for n in names):
                logger.info("GemmaTranslator ready — model: %s", self._model)
            else:
                logger.warning(
                    "Model '%s' not found in Ollama. Run: ollama pull %s",
                    self._model, self._model,
                )
        except Exception as exc:
            logger.warning("Could not reach Ollama at %s: %s", self._host, exc)

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text
        if not self._client:
            await self.setup()

        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": _SYSTEM.format(
                        src=_LANG_NAMES.get(source_lang, source_lang),
                        tgt=_LANG_NAMES.get(target_lang, target_lang),
                    ),
                },
                {"role": "user", "content": text},
            ],
        }
        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            translated = resp.json()["message"]["content"].strip()
            # Strip accidental leading/trailing quotes
            translated = re.sub(r'^["\']|["\']$', "", translated).strip()
            return translated or text
        except Exception as exc:
            logger.error("Gemma translate error: %s", exc)
            return text
