"""
TTS shared utilities
====================
  pcm_to_wav()      — wrap raw s16le PCM in a WAV container
  trim_wav_silence() — strip leading/trailing silence from a WAV
                       (Piper/Supertonic often add ~200-300 ms of dead air)
"""
import struct
import wave
import io


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = 22050,
    channels:    int = 1,
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


def trim_wav_silence(
    wav_bytes: bytes,
    threshold:        int   = 200,    # s16 amplitude below this = silence (~0.006 full scale)
    min_silence_ms:   float = 50,     # ignore silence shorter than this (ms)
    keep_padding_ms:  float = 80,     # keep N ms of silence at each end for natural onset/offset
) -> bytes:
    """
    Strip leading and trailing silence from WAV bytes.

    Piper and Supertonic often pad output with 200-400 ms of silence which
    causes audible gaps between consecutive TTS chunks during live playback.
    Trimming this makes the audio stream feel continuous.

    Parameters
    ----------
    threshold       : s16 amplitude considered silence (0-32767)
    min_silence_ms  : don't trim silence shorter than this (keeps natural micro-pauses)
    keep_padding_ms : re-add this many ms of silence at start and end (natural feel)

    Returns WAV bytes with silence trimmed. Falls back to original on any error.
    """
    if not wav_bytes:
        return wav_bytes

    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels   = wf.getnchannels()
            sampwidth    = wf.getsampwidth()
            framerate    = wf.getframerate()
            n_frames     = wf.getnframes()
            raw_frames   = wf.readframes(n_frames)

        if sampwidth != 2:
            # Only handle 16-bit PCM
            return wav_bytes

        import struct as _struct
        samples_per_frame = n_channels
        total_samples     = len(raw_frames) // 2
        samples           = _struct.unpack(f"<{total_samples}h", raw_frames)

        min_silence_samples  = int(framerate * min_silence_ms  / 1000) * n_channels
        keep_padding_samples = int(framerate * keep_padding_ms / 1000) * n_channels

        # Find first non-silent sample
        start = 0
        for i in range(0, len(samples), n_channels):
            if any(abs(samples[i + c]) > threshold for c in range(n_channels) if i + c < len(samples)):
                start = max(0, i - keep_padding_samples)
                break

        # Find last non-silent sample
        end = len(samples)
        for i in range(len(samples) - n_channels, -1, -n_channels):
            if any(abs(samples[i + c]) > threshold for c in range(n_channels) if i + c < len(samples)):
                end = min(len(samples), i + n_channels + keep_padding_samples)
                break

        if end <= start:
            return wav_bytes

        # Only trim if we actually removed more than min_silence_ms worth
        removed_start = start
        removed_end   = len(samples) - end
        if removed_start < min_silence_samples and removed_end < min_silence_samples:
            return wav_bytes

        trimmed_samples = samples[start:end]
        trimmed_pcm = _struct.pack(f"<{len(trimmed_samples)}h", *trimmed_samples)
        return pcm_to_wav(trimmed_pcm, sample_rate=framerate,
                          channels=n_channels, sample_width=sampwidth)

    except Exception:
        # Never break audio pipeline — return original on any error
        return wav_bytes
