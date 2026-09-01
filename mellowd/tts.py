"""Text to speech with Kokoro-82M. kokoro-onnx runs on ONNX Runtime"""

import asyncio
import logging
import re
from contextlib import suppress
from urllib.parse import quote

import httpx
import numpy as np
import sounddevice as sd

from mellowd import config, errors, models, wav

log = logging.getLogger("mellowd.tts")

# What Kokoro emits. A hosted voice may use any rate
SAMPLE_RATE = 24_000
MAX_CHARS = 180  # force a sentence break past this so speech can start

# A single sentence. Longer than this and the pause is worse than the silence.
CLOUD_TIMEOUT = 30.0

# What to ask each provider for. wav everywhere it is offered
CLOUD_FORMATS = {"openrouter": "pcm"}

# ponytail: assumed only when raw PCM arrives with no rate in the content type.
CLOUD_PCM_RATE = 24_000

# ElevenLabs clamps voice_settings.speed to this range and 422s outside it, but Mellow's slider goes
ELEVEN_SPEED = (0.7, 1.2)
# wav, not the default mp3, for the same reason as _cloud_synth
ELEVEN_FORMAT = "wav_24000"

_kokoro = None


def local_available() -> bool:
    """True only when both files needed for an offline voice are complete."""
    return models.available("kokoro-v1.0.onnx") and models.available("voices-v1.0.bin")


def load(progress=None, cfg: dict | None = None):
    """Load once and keep warm."""
    global _kokoro
    if _kokoro is not None:
        return _kokoro

    from kokoro_onnx import Kokoro

    onnx = models.ensure("kokoro-v1.0.onnx", progress)
    voices = models.ensure("voices-v1.0.bin", progress)
    k = Kokoro(str(onnx), str(voices))

    # Constructing the model proves nothing — the step 3 CUDA bug hid behind exactly this.
    cfg = cfg or config.load()
    samples, rate = k.create(
        "ready", voice=cfg["tts"]["local_voice"], speed=1.0, lang="en-us"
    )
    if samples is None or len(samples) == 0:
        raise RuntimeError("kokoro loaded but produced no audio")

    _kokoro = k
    log.info("kokoro ready, voice=%s, %dHz", cfg["tts"]["local_voice"], rate)
    return _kokoro


_MARKDOWN = re.compile(r"[*_`#>|]+")
_EMOJI = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]+"
)


# Typographic punctuation, mapped to the ASCII a phoneme frontend expects.
_PUNCTUATION = str.maketrans(
    {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "‑": "-", "‒": "-", "–": "-", "—": ",",
        "…": "...", " ": " ", "⁄": "/", "·": " ",
    }
)


# The screen marker (llm.LOOK), if one ever survives main._pass's scan window.
_MARKER = re.compile(
    r"\[look\](?![0-9A-Za-z])|\[POINT:[^\]]*\]",
    re.IGNORECASE,
)


def clean_for_speech(text: str) -> str:
    """Small models leak asterisks and emoji despite the prompt banning markdown, and Kokoro will happily"""
    text = text.translate(_PUNCTUATION)
    text = _MARKDOWN.sub("", text)
    text = _MARKER.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# Requires whitespace or end-of-string after the punctuation
_SENTENCE = re.compile(r"(.*?[.!?…]+[\"')\]]*)(?=\s|$)", re.S)


class SentenceBuffer:
    """Accumulates streamed LLM text and yields whole sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out = []
        while True:
            m = _SENTENCE.match(self._buf)
            if m:
                out.append(m.group(1).strip())
                self._buf = self._buf[m.end() :].lstrip()
                continue
            # No punctuation in sight
            if len(self._buf) > MAX_CHARS:
                cut = self._buf.rfind(" ", 0, MAX_CHARS) or MAX_CHARS
                out.append(self._buf[:cut].strip())
                self._buf = self._buf[cut:].lstrip()
                continue
            break
        return [s for s in out if s]

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


def _pcm_rate(content_type: str) -> int:
    """`audio/pcm; rate=44100` if the provider bothered to say, else the guess."""
    for part in content_type.split(";")[1:]:
        name, _, value = part.partition("=")
        if name.strip() in ("rate", "sample_rate") and value.strip().isdigit():
            return int(value.strip())
    return CLOUD_PCM_RATE


def _decode_audio(response: httpx.Response) -> tuple[np.ndarray, int]:
    """Whatever the provider sent, as (float32, rate)."""
    data = response.content
    if data[:4] == b"RIFF":
        return wav.decode(data)
    rate = _pcm_rate(response.headers.get("content-type", ""))
    # 16-bit mono little-endian, which is what /audio/speech pcm is everywhere.
    pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
    log.info("raw pcm: %d samples at %dHz", len(pcm), rate)
    return np.ascontiguousarray(pcm.astype(np.float32) / 32767.0), rate


def _ok(response: httpx.Response, section: dict) -> httpx.Response:
    """One error shape for every hosted call, so a wrong key reads the same whichever provider rejected it."""
    if not response.is_success:
        raise errors.provider_error(
            response.status_code, response.text, section["provider"], section["model"]
        )
    return response


def _cloud_synth(text: str, section: dict) -> tuple[np.ndarray, int]:
    """One sentence from an OpenAI-compatible /audio/speech endpoint."""
    url = f"{section['base_url'].rstrip('/')}/audio/speech"
    headers = {"Authorization": f"Bearer {section['api_key']}"} if section["api_key"] else {}
    payload = {
        "model": section["model"],
        "input": text,
        "speed": section["speech_speed"],
        "response_format": CLOUD_FORMATS.get(section["provider"], "wav"),
    }
    # Only when the user set one. Several hosted models have a single voice and either ignore the field
    if voice := section.get("voice"):
        payload["voice"] = voice
    response = _ok(
        httpx.post(url, headers=headers, json=payload, timeout=CLOUD_TIMEOUT), section
    )
    return _decode_audio(response)


def _elevenlabs_synth(text: str, section: dict) -> tuple[np.ndarray, int]:
    """One sentence from ElevenLabs, which is not OpenAI-compatible."""
    low, high = ELEVEN_SPEED
    response = _ok(
        httpx.post(
            f"{section['base_url'].rstrip('/')}/text-to-speech/{quote(section['voice'])}",
            params={"output_format": ELEVEN_FORMAT},
            headers={"xi-api-key": section["api_key"]},
            json={
                "text": text,
                "model_id": section["model"],
                "voice_settings": {"speed": min(high, max(low, section["speech_speed"]))},
            },
            timeout=CLOUD_TIMEOUT,
        ),
        section,
    )
    return _decode_audio(response)


def voices(cfg: dict | None = None) -> list[dict]:
    """The account's ElevenLabs voices, for the settings dropdown."""
    section = (cfg or config.load())["tts"]
    if section["provider"] != "elevenlabs":
        raise RuntimeError("only ElevenLabs can list its voices")
    base = section["base_url"].rstrip("/")
    # The voice list is a v2 endpoint while everything else here is v1
    url = f"{base[:-3]}/v2/voices" if base.endswith("/v1") else f"{base}/voices"
    response = _ok(
        httpx.get(
            url,
            params={"page_size": 100},
            headers={"xi-api-key": section["api_key"]},
            timeout=CLOUD_TIMEOUT,
        ),
        section,
    )
    found = response.json().get("voices") or []
    return [
        {"id": v["voice_id"], "name": v.get("name") or v["voice_id"]}
        for v in found
        if v.get("voice_id")
    ]


def _kokoro_synth(text: str, section: dict) -> tuple[np.ndarray, int]:
    return load().create(
        text,
        voice=section["local_voice"],
        speed=section["speech_speed"],
        lang="en-us",
    )


def synth(text: str, cfg: dict | None = None) -> tuple[np.ndarray, int]:
    """Blocking."""
    section = (cfg or config.load())["tts"]
    if section["mode"] != "cloud":
        return _kokoro_synth(text, section)
    if section["provider"] == "elevenlabs":
        return _elevenlabs_synth(text, section)
    return _cloud_synth(text, section)


def backend(cfg: dict | None = None) -> str:
    """Takes a cfg so an unsaved settings form can be reported accurately."""
    section = (cfg or config.load())["tts"]
    if section["mode"] == "cloud":
        return f"{section['provider']}/{section['model']}"
    return f"kokoro/{section['local_voice']}"


# How many synthesised sentences may sit ahead of the one being played.
LOOKAHEAD = 2


class Speaker:
    """Speech for one WebSocket connection."""

    def __init__(self, ws, send) -> None:
        self._ws = ws
        self._send = send
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._talking = False

    def begin(self) -> None:
        self._queue = asyncio.Queue()
        self._talking = False
        self._task = asyncio.create_task(self._run())

    async def speak(self, text: str) -> None:
        if text := clean_for_speech(text):
            await self._queue.put(text)

    async def finish(self) -> None:
        """Wait until every queued sentence has actually finished playing."""
        if self._task is None:
            return
        await self._queue.put(None)
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def stop(self) -> None:
        """Barge-in. Cuts off playback immediately."""
        if self._task is None or self._task.done():
            self._task = None
            return
        # sd.stop() FIRST
        sd.stop()
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        log.info("speech interrupted")

    async def _run(self) -> None:
        """Runs the two stages and owns the cleanup of both."""
        clips: asyncio.Queue = asyncio.Queue(maxsize=LOOKAHEAD)
        fetch = asyncio.create_task(self._fetch(clips))
        try:
            await self._play(clips)
        finally:
            fetch.cancel()
            with suppress(asyncio.CancelledError):
                await fetch
            sd.stop()

    async def _fetch(self, clips: asyncio.Queue) -> None:
        """Text in, audio out. Blocks on `clips` once the lookahead is full."""
        while True:
            text = await self._queue.get()
            if text is None:
                await clips.put(None)
                return
            # Before the synth, not after. Locally that was a 30ms difference nobody could see
            if not self._talking:
                self._talking = True
                await self._send(self._ws, type="state", state="talking")
            await clips.put(await asyncio.to_thread(synth, text))

    async def _play(self, clips: asyncio.Queue) -> None:
        while True:
            clip = await clips.get()
            if clip is None:
                return
            samples, rate = clip
            await asyncio.to_thread(sd.play, samples, rate, blocking=True)
