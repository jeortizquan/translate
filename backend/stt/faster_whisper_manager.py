"""
STT Engine — faster-whisper 1.2.1
===================================
pip install faster-whisper sounddevice numpy

Captures microphone audio in real-time using sounddevice,
applies energy-based VAD to detect speech boundaries,
transcribes complete speech segments with faster-whisper.

macOS GPU (Apple Silicon):
  CTranslate2 (faster-whisper backend) uses CPU INT8 on macOS ARM,
  which is extremely fast on M-series chips (often faster than CUDA
  on small/medium models). Metal GPU is used automatically when
  ctranslate2 is built with Metal support.

  If mlx-whisper is installed (pip install mlx-whisper), it is used
  instead for true Apple Silicon GPU acceleration via the MLX framework.
"""
import asyncio
import logging
import platform
import queue
import threading
import time
from typing import Callable, Awaitable, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE   = 16000   # Whisper requires 16 kHz
CHANNELS      = 1
CHUNK_MS      = 50      # audio callback chunk size in ms
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


def _detect_device() -> tuple[str, str]:
    """
    Return (device, compute_type) optimal for this machine.

    macOS Apple Silicon → cpu / int8   (fast NEON SIMD, or Metal if CT2 supports it)
    CUDA available      → cuda / float16
    Fallback            → cpu / int8
    """
    system  = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine in ("arm64", "aarch64"):
        logger.info("faster-whisper: Apple Silicon detected — using cpu/int8 (optimised NEON)")
        return "cpu", "int8"

    try:
        import torch
        if torch.cuda.is_available():
            logger.info("faster-whisper: CUDA GPU detected")
            return "cuda", "float16"
    except ImportError:
        pass

    return "cpu", "int8"


class FasterWhisperManager:
    """
    Real-time STT using faster-whisper 1.2.1.

    Architecture:
      sounddevice callback → raw audio queue (thread-safe)
      → VAD loop (background thread) → detect speech/silence boundaries
      → transcribe complete speech segment → call on_segment (async)
    """

    def __init__(self):
        self._model          = None
        self._model_name     = "base"
        self._language       = "en"
        self._device         = "cpu"
        self._compute_type   = "int8"
        self._mic_index: Optional[int] = None
        self._on_segment: Optional[Callable] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._audio_queue: queue.Queue = queue.Queue()
        self._sd_stream   = None
        self._vad_thread: Optional[threading.Thread] = None
        self._running      = False

        # VAD parameters
        self.vad_energy_threshold = 0.01    # RMS energy above this = speech
        self.vad_speech_pad_ms    = 300     # ms of silence to keep after speech ends
        self.vad_min_speech_ms    = 200     # minimum speech duration to transcribe
        self.vad_max_speech_ms    = 8000    # maximum speech buffer before forced flush
        self.vad_silence_ms       = 700     # ms of silence to trigger end-of-speech

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(
        self,
        model_name: str,
        language: str,
        mic_index: int,
        on_segment: Callable[[str], Awaitable[None]],
    ) -> None:
        if self._running:
            await self.stop()

        self._model_name = model_name
        self._language   = language
        self._mic_index  = mic_index
        self._on_segment = on_segment
        self._loop       = asyncio.get_event_loop()
        self._device, self._compute_type = _detect_device()

        # Load model in thread (can be slow first time)
        logger.info(
            "faster-whisper: loading model=%r device=%s compute_type=%s …",
            model_name, self._device, self._compute_type,
        )
        await asyncio.to_thread(self._load_model)
        logger.info("faster-whisper: model ready")

        self._running = True
        self._start_microphone()
        self._vad_thread = threading.Thread(
            target=self._vad_loop, daemon=True, name="fw-vad"
        )
        self._vad_thread.start()
        logger.info("faster-whisper: listening on mic index=%s", mic_index)

    async def stop(self) -> None:
        self._running = False
        self._stop_microphone()
        if self._vad_thread:
            self._vad_thread.join(timeout=3)
        logger.info("faster-whisper: stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Try mlx-whisper first (Apple GPU), fall back to faster-whisper."""
        # Attempt mlx-whisper on Apple Silicon
        if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
            try:
                import mlx_whisper  # type: ignore
                self._model = ("mlx", mlx_whisper, self._model_name)
                logger.info("faster-whisper: using mlx-whisper (Apple Silicon GPU)")
                return
            except ImportError:
                logger.info("mlx-whisper not installed — using faster-whisper CPU")

        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
            download_root=None,   # uses ~/.cache/huggingface by default
        )

    # ── Microphone capture ────────────────────────────────────────────────────

    def _start_microphone(self) -> None:
        try:
            import sounddevice as sd

            def _callback(indata, frames, time_info, status):
                if status:
                    logger.debug("sounddevice status: %s", status)
                # indata is float32 (−1…1), shape (frames, channels)
                self._audio_queue.put(indata[:, 0].copy())

            self._sd_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                device=self._mic_index,
                callback=_callback,
            )
            self._sd_stream.start()
            logger.info("sounddevice stream started")
        except Exception as exc:
            logger.error("Could not open microphone: %s", exc)
            self._running = False

    def _stop_microphone(self) -> None:
        if self._sd_stream:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

    # ── VAD + transcription loop (runs in background thread) ─────────────────

    def _vad_loop(self) -> None:
        """
        Reads audio chunks from the queue, detects speech/silence,
        and fires transcription when a complete speech segment ends.
        """
        speech_buffer: list[np.ndarray] = []
        in_speech      = False
        silence_frames = 0
        speech_frames  = 0

        silence_chunks_threshold = int(self.vad_silence_ms  / CHUNK_MS)
        pad_chunks               = int(self.vad_speech_pad_ms / CHUNK_MS)
        min_speech_chunks        = int(self.vad_min_speech_ms / CHUNK_MS)
        max_speech_chunks        = int(self.vad_max_speech_ms / CHUNK_MS)

        pad_buffer: list[np.ndarray] = []   # recent audio for pre-speech padding

        while self._running:
            try:
                chunk: np.ndarray = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                # Flush if we've been collecting speech and hit silence
                if in_speech and speech_frames >= min_speech_chunks:
                    self._transcribe(np.concatenate(speech_buffer))
                    speech_buffer.clear()
                    in_speech     = False
                    silence_frames = 0
                    speech_frames  = 0
                continue

            # RMS energy as speech indicator
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_speech = rms > self.vad_energy_threshold

            if is_speech:
                if not in_speech:
                    # Speech onset — include recent pre-speech pad
                    in_speech = True
                    silence_frames = 0
                    speech_buffer = list(pad_buffer)  # prepend pad
                silence_frames = 0
                speech_buffer.append(chunk)
                speech_frames += 1

                # Hard limit — flush immediately
                if speech_frames >= max_speech_chunks:
                    self._transcribe(np.concatenate(speech_buffer))
                    speech_buffer.clear()
                    in_speech     = False
                    silence_frames = 0
                    speech_frames  = 0

            else:
                if in_speech:
                    speech_buffer.append(chunk)
                    silence_frames += 1
                    if silence_frames >= silence_chunks_threshold:
                        # End of speech detected
                        if speech_frames >= min_speech_chunks:
                            self._transcribe(np.concatenate(speech_buffer))
                        speech_buffer.clear()
                        in_speech     = False
                        silence_frames = 0
                        speech_frames  = 0

                # Maintain a rolling pre-speech pad buffer
                pad_buffer.append(chunk)
                if len(pad_buffer) > pad_chunks:
                    pad_buffer.pop(0)

    def _transcribe(self, audio: np.ndarray) -> None:
        """Run faster-whisper (or mlx-whisper) on a complete speech segment."""
        if not self._running or self._model is None:
            return
        try:
            text = self._run_model(audio)
            if text and self._on_segment and self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._on_segment(text), self._loop
                )
        except Exception as exc:
            logger.error("faster-whisper transcription error: %s", exc)

    def _run_model(self, audio: np.ndarray) -> str:
        """Dispatch to mlx-whisper or faster-whisper and return clean text."""
        # mlx-whisper path
        if isinstance(self._model, tuple) and self._model[0] == "mlx":
            _, mlx_whisper, model_name = self._model
            # mlx_whisper expects path or audio array
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=f"mlx-community/whisper-{model_name}-mlx",
                language=self._language,
                fp16=True,
            )
            return result.get("text", "").strip()

        # faster-whisper path
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,          # built-in silero VAD as second pass
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.debug("faster-whisper raw: %r (lang=%s)", text, info.language)
        return text


def list_microphones() -> list:
    """Return [{index, name}] for every audio input device via sounddevice."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception:
        pass
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        result = [
            {"index": i, "name": pa.get_device_info_by_index(i)["name"]}
            for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i)["maxInputChannels"] > 0
        ]
        pa.terminate()
        return result
    except Exception as exc:
        logger.warning("Could not list microphones: %s", exc)
        return [{"index": 0, "name": "Default Microphone"}]
