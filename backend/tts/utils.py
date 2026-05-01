"""Shared TTS utilities."""
import struct


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = 22050,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw s16-le PCM bytes in a minimal WAV container."""
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16,
        1,                                      # PCM
        channels,
        sample_rate,
        sample_rate * channels * sample_width,  # byte-rate
        channels * sample_width,                # block-align
        sample_width * 8,                       # bits-per-sample
        b"data", data_size,
    )
    return header + pcm
