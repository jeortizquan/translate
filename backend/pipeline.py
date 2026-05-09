"""
Translation Pipeline
====================
Orchestrates:
  whisper-stream → SegmentCoalescer → translate all active languages
                 → profanity filter → TTS + silence trim → broadcast

Natural-sounding improvements:
  1. SegmentCoalescer accumulates partial STT fragments into complete sentences
     before they reach translation/TTS — eliminates per-fragment gaps.
  2. WAV silence trimming removes Piper/Supertonic's leading/trailing dead air.
  3. Per-language TTS workers run in parallel so all languages synthesise
     concurrently, not sequentially.
"""
import asyncio
import logging
import time
from typing import Optional

from .broadcast import manager
from .coalescer import SegmentCoalescer
from .config import LANGUAGES, VOICES_DIR, settings
from .filter import profanity_filter
from .translation import make_translator, BaseTranslator
from .tts import make_tts, BaseTTS
from .tts.utils import trim_wav_silence

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

        # SegmentCoalescer buffers partial whisper fragments into full sentences
        self._coalescer = SegmentCoalescer(
            on_sentence      = self._on_sentence,
            silence_timeout  = 2.0,   # flush after 2s of silence
            max_words        = 60,    # never hold more than 60 words
            min_words        = 3,     # don't synthesise single words
        )

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
        # Flush any buffered text before stopping
        await self._coalescer.flush_now()
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

    # ── STT entry point (called by WhisperStreamManager) ─────────────────────

    async def on_segment(self, text: str) -> None:
        """
        Receives raw STT fragments from whisper-stream.
        Forwards to the SegmentCoalescer which buffers until a complete
        sentence is ready, then calls _on_sentence().
        """
        text = text.strip()
        if not text:
            return
        await self._coalescer.push(text)

    # ── Coalescer callback (complete sentence ready) ──────────────────────────

    async def _on_sentence(self, text: str) -> None:
        """Called by the coalescer with a complete, natural sentence."""
        # Apply profanity filter to original text
        clean_text = profanity_filter.clean(text)

        # Broadcast original (filtered) transcript to all clients
        await manager.broadcast_all_text({
            "type": "transcript",
            "lang": settings.source_language,
            "text": clean_text,
            "ts":   time.time(),
        })

        # Enqueue for translation (drop oldest if queue is full)
        if self._segment_queue.full():
            try:
                self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._segment_queue.put_nowait((clean_text, settings.source_language))

    # ── Translation worker ────────────────────────────────────────────────────

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
                # Apply profanity filter to translated output too
                translated = profanity_filter.clean(translated)

                # Send subtitle to clients
                await manager.broadcast_text(lang, {
                    "type":     "subtitle",
                    "lang":     lang,
                    "text":     translated,
                    "original": text,
                    "ts":       time.time(),
                })

                # Enqueue for TTS synthesis
                try:
                    self._audio_queues[lang].put_nowait(translated)
                except asyncio.QueueFull:
                    logger.debug("Audio queue [%s] full — skipping segment", lang)

            # All languages translated in parallel
            await asyncio.gather(
                *[_translate_one(lang) for lang in active],
                return_exceptions=True,
            )
            self._segment_queue.task_done()

    # ── TTS workers (one per language, run in parallel) ───────────────────────

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
                # Trim leading/trailing silence for continuous playback
                audio_bytes = trim_wav_silence(audio_bytes)
                await manager.broadcast_audio(lang, audio_bytes)

            queue.task_done()


pipeline = Pipeline()
