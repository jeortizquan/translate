"""
STT — whisper.cpp whisper-stream wrapper

Fixes vs previous version:
  - '[Start speaking]' and all whisper bracket-tags now filtered after extraction
  - General rule: any clean segment that is entirely wrapped in [...] is a tag, not speech
  - ANSI escape codes stripped before text reaches the pipeline
  - Only the LAST (most complete) phrase from each whisper output burst is forwarded
"""
import asyncio
import logging
import re
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

# ── Terminal / ANSI noise ──────────────────────────────────────────────────────
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

# Whisper bracket tags — checked as lowercase against segment.lower()
_NOISE_TAGS = frozenset({
    "[blank_audio]",
    "[start]",
    "[end]",
    "[start speaking]",   # ← whisper-stream startup prompt
    "[music]",
    "[noise]",
    "[laughter]",
    "[applause]",
    "[silence]",
})

# Meta lines from the binary itself — skip the whole raw line
_SKIP_PREFIXES = (
    "init:", "main:", "whisper_", "ggml_", "error:",
    "system_info:", "processing", "loading", "encode",
    "operator()",
)


def _is_bracket_tag(text: str) -> bool:
    """Return True if the entire cleaned text is a whisper bracket tag like [BLANK_AUDIO]."""
    s = text.strip()
    return s.startswith("[") and s.endswith("]")


def _extract_final_segment(raw: str) -> str:
    """
    whisper-stream rewrites the terminal line using ANSI erase sequences:
        \x1b[2K\r [BLANK_AUDIO] \x1b[2K\r The Bible says. \x1b[2K\r The Bible says the heart...

    We split on \r, strip ANSI codes, skip empty/noise chunks, and return
    the LAST (most complete) meaningful phrase only.
    """
    chunks = raw.split('\r')
    meaningful = []
    for chunk in chunks:
        clean = _ANSI_RE.sub('', chunk).strip()
        if not clean:
            continue
        # Skip pure bracket tags (BLANK_AUDIO etc.)
        if _is_bracket_tag(clean):
            continue
        if len(clean) < 3:
            continue
        meaningful.append(clean)

    return meaningful[-1] if meaningful else ""


class WhisperStreamManager:
    """Wraps a whisper-stream subprocess and feeds clean text segments to the pipeline."""

    def __init__(self):
        self._process: Optional[asyncio.subprocess.Process] = None
        self._tasks:   list[asyncio.Task] = []
        self._running: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(
        self,
        binary: str,
        model_path: str,
        language: str,
        mic_index: int,
        on_segment: Callable[[str], Awaitable[None]],
    ) -> None:
        if self._running:
            await self.stop()

        cmd = self._build_cmd(binary, model_path, language, mic_index)
        logger.info("Launching whisper-stream:\n  %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._running = True

        self._tasks = [
            asyncio.create_task(
                self._reader(self._process.stdout, "stdout", on_segment),
                name="ws-stdout",
            ),
            asyncio.create_task(
                self._reader(self._process.stderr, "stderr", on_segment),
                name="ws-stderr",
            ),
            asyncio.create_task(self._watchdog(), name="ws-watchdog"),
        ]
        logger.info("whisper-stream PID %s started", self._process.pid)

    async def stop(self) -> None:
        self._running = False
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("whisper-stream stopped")

    @property
    def is_running(self) -> bool:
        return (
            self._running
            and self._process is not None
            and self._process.returncode is None
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_cmd(self, binary: str, model_path: str, language: str, mic_index: int) -> list:
        return [
            binary,
            "-m",          model_path,
            "-l",          language,
            "--step",      "520",
            "--length",    "5200",
            "--keep",      "220",
            "--threads",   "8",
            "--vad-thold", "0.6",
            "--freq-thold","100.0",
            "--capture",   str(mic_index),
        ]

    async def _reader(
        self,
        stream: asyncio.StreamReader,
        stream_name: str,
        on_segment: Callable[[str], Awaitable[None]],
    ) -> None:
        while self._running:
            try:
                raw_bytes = await asyncio.wait_for(stream.readline(), timeout=60)
            except asyncio.TimeoutError:
                logger.debug("[whisper/%s] no output for 60 s", stream_name)
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("[whisper/%s] read error: %s", stream_name, exc)
                break

            if not raw_bytes:
                logger.debug("[whisper/%s] EOF", stream_name)
                break

            raw = raw_bytes.decode("utf-8", errors="replace")
            logger.debug("[whisper/%s] raw: %r", stream_name, raw.strip())

            # Skip meta/binary lines on the raw text
            lower_raw = raw.lower().strip()
            if any(lower_raw.startswith(p) for p in _SKIP_PREFIXES):
                continue

            # Extract the final clean segment from the terminal-overwrite burst
            segment = _extract_final_segment(raw)
            if not segment:
                continue

            # Skip whisper bracket tags like [Start speaking], [BLANK_AUDIO] etc.
            if segment.lower() in _NOISE_TAGS:
                continue

            # General bracket-tag guard: skip anything that is entirely [...]
            if _is_bracket_tag(segment):
                logger.debug("[whisper/%s] skipping bracket tag: %r", stream_name, segment)
                continue

            logger.info("STT ✓ [%s]: %r", stream_name, segment)
            await on_segment(segment)

    async def _watchdog(self) -> None:
        if self._process is None:
            return
        code = await self._process.wait()
        if self._running:
            logger.error(
                "whisper-stream exited unexpectedly (code %s). "
                "Check: binary path correct, SDL2 installed, mic index valid. "
                "Run with --debug to see raw output.",
                code,
            )
            self._running = False


def list_microphones() -> list:
    """Return [{index, name}] for every audio input device via PyAudio."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({"index": i, "name": info["name"]})
        pa.terminate()
        return devices
    except Exception as exc:
        logger.warning("PyAudio not available (%s) — returning default mic", exc)
        return [{"index": 0, "name": "Default Microphone"}]
