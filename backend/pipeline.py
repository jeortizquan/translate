"""
Translation Pipeline
====================
STT segment → [profanity filter] → translate to all active languages
           → [profanity filter on translation] → TTS → broadcast audio + subtitles
"""
import asyncio
import logging
import time
from typing import Optional

from .broadcast import manager
from .config import LANGUAGES, VOICES_DIR, settings
from .filter import profanity_filter
from .translation import make_translator, BaseTranslator
from .tts import make_tts, BaseTTS

logger = logging.getLogger(__name__)

_SEGMENT_QUEUE_MAXSIZE = 50
_AUDIO_QUEUE_MAXSIZE   = 20


class Pipeline:
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

    # ── Entry point ───────────────────────────────────────────────────────────

    async def on_segment(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        # Apply profanity filter to the original STT text
        clean_text = profanity_filter.clean(text)

        # Broadcast original (filtered) transcript to every connected client
        await manager.broadcast_all_text({
            "type": "transcript",
            "lang": settings.source_language,
            "text": clean_text,
            "ts":   time.time(),
        })

        # Drop into translation queue
        if self._segment_queue.full():
            try:
                self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._segment_queue.put_nowait((clean_text, settings.source_language))

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

            active = [lang for lang in LANGUAGES if manager.count(lang) > 0]
            if not active:
                self._segment_queue.task_done()
                continue

            async def _translate_one(lang: str) -> None:
                # Translate
                translated = (
                    text
                    if lang == source_lang
                    else await self._translator.translate(text, source_lang, lang)
                )
                # Apply profanity filter to the translated text too
                translated = profanity_filter.clean(translated)

                await manager.broadcast_text(lang, {
                    "type":     "subtitle",
                    "lang":     lang,
                    "text":     translated,
                    "original": text,
                    "ts":       time.time(),
                })
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


pipeline = Pipeline()
