"""
Broadcast Manager
Manages WebSocket connections per language channel.
Distributes binary audio and JSON text to all subscribed clients.
Supports 500-1000 simultaneous users via asyncio.gather fan-out.
"""
import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        # lang_code -> set of active WebSocket connections
        self._channels: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, lang: str) -> None:
        await ws.accept()
        async with self._lock:
            self._channels[lang].add(ws)
        logger.info(f"Client joined [{lang}] — total in channel: {self.count(lang)}")

    async def disconnect(self, ws: WebSocket, lang: str) -> None:
        async with self._lock:
            self._channels[lang].discard(ws)
        logger.info(f"Client left [{lang}] — remaining: {self.count(lang)}")

    def count(self, lang: str) -> int:
        return len(self._channels.get(lang, set()))

    def counts(self) -> Dict[str, int]:
        return {lang: len(sockets) for lang, sockets in self._channels.items()}

    def total(self) -> int:
        return sum(len(s) for s in self._channels.values())

    async def broadcast_audio(self, lang: str, audio_bytes: bytes) -> None:
        """Send binary WAV audio chunk to all subscribers of a language."""
        sockets = list(self._channels.get(lang, set()))
        if not sockets:
            return
        results = await asyncio.gather(
            *[ws.send_bytes(audio_bytes) for ws in sockets],
            return_exceptions=True,
        )
        dead = [ws for ws, r in zip(sockets, results) if isinstance(r, Exception)]
        if dead:
            async with self._lock:
                for ws in dead:
                    self._channels[lang].discard(ws)

    async def broadcast_text(self, lang: str, payload: dict) -> None:
        """Send a JSON event to all subscribers of a language."""
        sockets = list(self._channels.get(lang, set()))
        if not sockets:
            return
        message = json.dumps(payload)
        results = await asyncio.gather(
            *[ws.send_text(message) for ws in sockets],
            return_exceptions=True,
        )
        dead = [ws for ws, r in zip(sockets, results) if isinstance(r, Exception)]
        if dead:
            async with self._lock:
                for ws in dead:
                    self._channels[lang].discard(ws)

    async def broadcast_all_text(self, payload: dict) -> None:
        """Broadcast a JSON message to every connected client (e.g. original transcript)."""
        all_sockets = [ws for sockets in self._channels.values() for ws in sockets]
        if not all_sockets:
            return
        message = json.dumps(payload)
        await asyncio.gather(
            *[ws.send_text(message) for ws in all_sockets],
            return_exceptions=True,
        )


# Singleton used by the whole app
manager = ConnectionManager()
