"""
Segment Coalescer
==================
Buffers incoming STT fragments and flushes complete, natural-sounding
units to the translation pipeline.

Problem without this:
  whisper emits:  "The" → "The Bible" → "The Bible says" → "The Bible says the heart"
  TTS receives 4 fragments → 4 audio chunks with gaps between them → choppy output

With coalescer:
  whisper emits same fragments, coalescer accumulates them
  Flushes "The Bible says the heart of man is deceitful." as ONE chunk
  TTS receives 1 complete sentence → 1 smooth audio chunk → natural output

Flush triggers (whichever fires first):
  A) Sentence boundary detected   . ! ? ; — followed by whitespace or end
  B) Hard word-count limit         (prevent very long TTS inputs)
  C) Silence timeout               (BLANK_AUDIO or no new text for N seconds)
  D) Explicit flush request        (operator stops broadcast)

Deduplication:
  Whisper re-emits context from the previous window. The coalescer strips
  leading text that duplicates the end of the last flushed sentence.
"""
import asyncio
import logging
import re
import time
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

# Sentence-ending punctuation followed by whitespace or end-of-string
_SENTENCE_END_RE = re.compile(r'[.!?;]\s*$|[.!?;]\s+')

# Filler / repeated noise tokens whisper sometimes emits
_FILLER_RE = re.compile(
    r'\b(um+|uh+|er+|ah+|hmm+|huh|like|you know|i mean)\b',
    re.IGNORECASE,
)


class SegmentCoalescer:
    """
    Accumulates STT text fragments and emits complete sentences.

    Usage:
        coalescer = SegmentCoalescer(on_sentence=pipeline.on_sentence)
        # wire whisper output to:
        await coalescer.push("The")
        await coalescer.push("The Bible says")
        # → on_sentence fires with "The Bible says the heart of man…" when complete
    """

    def __init__(
        self,
        on_sentence: Callable[[str], Awaitable[None]],
        silence_timeout:  float = 2.0,   # flush after N seconds of no new text
        max_words:        int   = 60,    # flush when buffer exceeds N words
        min_words:        int   = 3,     # don't flush very short fragments
    ):
        self._on_sentence      = on_sentence
        self._silence_timeout  = silence_timeout
        self._max_words        = max_words
        self._min_words        = min_words

        self._buffer:     str   = ""
        self._last_push:  float = 0.0
        self._last_flushed: str = ""   # last sentence we sent (for dedup)
        self._timer_task: asyncio.Task | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def push(self, text: str) -> None:
        """
        Receive a new STT segment. Deduplicates context, accumulates,
        and flushes when a sentence boundary or limit is reached.
        """
        text = text.strip()
        if not text:
            return

        # Deduplicate: whisper re-sends context from the previous window.
        # If the new text STARTS with what we already flushed, strip that prefix.
        text = self._strip_duplicate_prefix(text)
        if not text:
            return

        # Update buffer with the freshest recognised text.
        # Whisper continuously refines the same window, so replace rather than append.
        self._buffer     = text
        self._last_push  = time.monotonic()

        # Reset silence timer
        self._cancel_timer()

        # Check flush conditions
        if self._has_sentence_boundary(self._buffer):
            # Extract everything up to and including the sentence end
            sentences, remainder = self._split_at_boundary(self._buffer)
            for s in sentences:
                await self._flush(s)
            self._buffer = remainder
            if remainder:
                self._start_timer()
        elif self._word_count(self._buffer) >= self._max_words:
            # Hard limit — flush the whole buffer
            await self._flush(self._buffer)
            self._buffer = ""
        else:
            # No sentence boundary yet — start silence timer
            self._start_timer()

    async def flush_now(self) -> None:
        """Force-flush whatever is in the buffer (e.g. when broadcast stops)."""
        self._cancel_timer()
        if self._buffer.strip():
            await self._flush(self._buffer)
            self._buffer = ""

    # ── Internal ──────────────────────────────────────────────────────────────

    def _strip_duplicate_prefix(self, new_text: str) -> str:
        """
        Whisper re-emits context from the previous window.
        E.g. last flushed = "The Bible says the heart of man is deceitful."
             new_text     = "The Bible says the heart of man is deceitful. Who can know it?"
        We strip the shared prefix and return "Who can know it?"
        """
        if not self._last_flushed:
            return new_text

        last = self._last_flushed.lower().strip()
        new  = new_text.lower().strip()

        # Check if new_text starts with (or contains) last flushed content
        if new.startswith(last):
            remainder = new_text[len(self._last_flushed):].strip()
            return remainder if len(remainder) > 2 else ""

        # Partial overlap: last 5 words of last_flushed appear at start of new_text
        last_words = last.split()[-5:]
        if len(last_words) >= 3:
            tail = " ".join(last_words)
            idx = new.find(tail)
            if idx == 0:
                cut = len(tail)
                return new_text[cut:].strip()

        return new_text

    def _has_sentence_boundary(self, text: str) -> bool:
        return bool(_SENTENCE_END_RE.search(text))

    def _split_at_boundary(self, text: str) -> tuple[list[str], str]:
        """
        Split text into complete sentences + a trailing remainder.
        Returns ([sentence1, sentence2, ...], remainder)
        """
        sentences = []
        # Split on sentence-ending punctuation while keeping the delimiter
        parts = re.split(r'(?<=[.!?;])\s+', text)
        remainder = ""

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # Last part may be incomplete (no sentence-end punctuation)
            if i == len(parts) - 1 and not _SENTENCE_END_RE.search(part):
                remainder = part
            else:
                if self._word_count(part) >= self._min_words:
                    sentences.append(part)
                elif sentences:
                    # Append short fragment to previous sentence
                    sentences[-1] = sentences[-1].rstrip() + " " + part
                else:
                    remainder = part  # too short, hold back

        return sentences, remainder

    async def _flush(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self._word_count(text) < self._min_words:
            logger.debug("Coalescer: too short to flush (%r) — holding", text)
            return

        logger.info("Coalescer → pipeline: %r", text)
        self._last_flushed = text
        await self._on_sentence(text)

    # ── Silence timer ─────────────────────────────────────────────────────────

    def _start_timer(self) -> None:
        self._cancel_timer()
        self._timer_task = asyncio.create_task(self._silence_watchdog())

    def _cancel_timer(self) -> None:
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self._timer_task = None

    async def _silence_watchdog(self) -> None:
        """Flush if no new text arrives within silence_timeout seconds."""
        await asyncio.sleep(self._silence_timeout)
        if self._buffer.strip():
            logger.debug("Coalescer: silence timeout — flushing %r", self._buffer)
            await self._flush(self._buffer)
            self._buffer = ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())
