"""
Translation Pipeline
====================
Orchestrates the full chain:
  STT segment → translate to all active languages → TTS → broadcast audio + subtitles

Flow:
  whisper-stream stdout
      │
      └─► on_segment()  ──►  _segment_queue
                                   │
                             [_translation_worker]
                                   │ translates to each active language in parallel
                                   ├─► broadcast subtitle (JSON) to lang channel
                                   └─► _audio_queues[lang]
                                              │
                                       [_tts_worker per lang]
                                              │ synthesises WAV once
                                              └─► broadcast_audio(lang, wav_bytes)
                                                       │
                                                  all N listeners for that lang
"""
import asyncio
import logging
import time
from typing import Optional

from .broadcast import manager
from .config import LANGUAGES, VOICES_DIR, settings
from .translation import make_translator, BaseTranslator
from .tts import make_tts, BaseTTS

logger = logging.getLogger(__name__)

_SEGMENT_QUEUE_MAXSIZE = 50   # drop oldest if STT is faster than translation
_AUDIO_QUEUE_MAXSIZE   = 20   # per language


class Pipeline:
    """Singleton orchestrator — one instance for the lifetime of the server."""

    def __init__(self):
        self._segment_queue: asyncio.Queue = asyncio.Queue(maxsize=_SEGMENT_QUEUE_MAXSIZE)
        self._audio_queues:  dict[str, asyncio.Queue] = {
            lang: asyncio.Queue(maxsize=_AUDIO_QUEUE_MAXSIZE)
            for lang in LANGUAGES
        }
        self._translator: Optional[BaseTranslator] = None
        self._tts:        Optional[BaseTTS]        = None
        self._tasks:      list[asyncio.Task]       = []
        self._running:    bool                     = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            await self.stop()
        await self._init_engines()
        self._running = True
        self._tasks.append(asyncio.create_task(self._translation_worker()))
        for lang in LANGUAGES:
            self._tasks.append(asyncio.create_task(self._tts_worker(lang)))
        logger.info("Pipeline started")

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._translator:
            await self._translator.teardown()
        if self._tts:
            await self._tts.teardown()
        logger.info("Pipeline stopped")

    async def restart_engines(self) -> None:
        """Call after operator changes TTS or translation engine at runtime."""
        if self._translator:
            await self._translator.teardown()
        if self._tts:
            await self._tts.teardown()
        await self._init_engines()

    async def _init_engines(self) -> None:
        self._translator = make_translator(settings.translation_engine, settings.gemma_host)
        await self._translator.setup()

        self._tts = make_tts(settings.tts_engine, VOICES_DIR)
        await self._tts.setup()

        logger.info(
            "Engines ready — translation=%s  tts=%s",
            settings.translation_engine, settings.tts_engine,
        )

    # ── Public entry point (called by WhisperStreamManager) ───────────────────

    async def on_segment(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        # Broadcast original transcript to every connected client
        await manager.broadcast_all_text({
            "type": "transcript",
            "lang": settings.source_language,
            "text": text,
            "ts":   time.time(),
        })

        # Enqueue for translation (drop oldest if full)
        if self._segment_queue.full():
            try:
                self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._segment_queue.put_nowait((text, settings.source_language))

    # ── Workers ───────────────────────────────────────────────────────────────

    async def _translation_worker(self) -> None:
        while self._running:
            try:
                text, source_lang = await asyncio.wait_for(
                    self._segment_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Only translate to languages that have at least one active listener
            active = [lang for lang in LANGUAGES if manager.count(lang) > 0]
            if not active:
                self._segment_queue.task_done()
                continue

            async def _translate_one(lang: str) -> None:
                translated = (
                    text
                    if lang == source_lang
                    else await self._translator.translate(text, source_lang, lang)
                )
                # Send subtitle text to all listeners of this language
                await manager.broadcast_text(lang, {
                    "type":     "subtitle",
                    "lang":     lang,
                    "text":     translated,
                    "original": text,
                    "ts":       time.time(),
                })
                # Enqueue translated text for TTS synthesis
                try:
                    self._audio_queues[lang].put_nowait(translated)
                except asyncio.QueueFull:
                    logger.debug("Audio queue [%s] full — skipping", lang)

            await asyncio.gather(
                *[_translate_one(lang) for lang in active],
                return_exceptions=True,
            )
            self._segment_queue.task_done()

    async def _tts_worker(self, lang: str) -> None:
        queue = self._audio_queues[lang]
        while self._running:
            try:
                text = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if manager.count(lang) == 0:
                queue.task_done()
                continue

            voice_override = settings.voice_overrides.get(lang)
            audio_bytes = await self._tts.synthesize(
                text, lang,
                speed=settings.tts_speed,
                voice_override=voice_override,
            )
            if audio_bytes:
                await manager.broadcast_audio(lang, audio_bytes)
            queue.task_done()


# Singleton
pipeline = Pipeline()
