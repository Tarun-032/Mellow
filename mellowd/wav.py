"""WAV in and out, on the stdlib. Cloud speech needs to send a recording and receive a voice."""

import io
import wave

import numpy as np


def encode(audio: np.ndarray, rate: int) -> bytes:
    """Mono float32 -> 16-bit PCM WAV bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    # 32767, not 32768: scaling by 32768 makes -1.0 valid but +1.0 overflow.
    pcm = (clipped * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes -> (mono float32, rate). Mixes down anything multi-channel."""
    with wave.open(io.BytesIO(data), "rb") as src:
        width = src.getsampwidth()
        if width != 2:
            raise ValueError(f"expected 16-bit WAV audio, got {width * 8}-bit")
        channels = src.getnchannels()
        rate = src.getframerate()
        frames = src.readframes(src.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(pcm), rate
