"""Speech to text. Records from the mic, transcribes with Parakeet or whisper."""

import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from mellowd import config, errors, wav

log = logging.getLogger("mellowd.stt")

# Generous: a long hold on a slow connection is still worth waiting for, and the user is already
CLOUD_TIMEOUT = 60.0

SAMPLE_RATE = 16_000  # what both engines want
MIN_SECONDS = 0.3  # shorter than this is a stray keypress, not speech
MIN_PEAK = 0.01  # measured just above this machine's empty-room peak (~0.007)
TARGET_PEAK = 0.25
# Applied only *after* MIN_PEAK has judged the take to be speech
MAX_GAIN = 20.0

# Windows' modern audio API
WASAPI = "Windows WASAPI"

# How the take is split when ranking channels, and how much of it counts.
RANK_FRAMES = 100
RANK_LOUDEST = 0.2

# Starting a WASAPI stream on this laptop transiently fails with "Unanticipated host error
OPEN_RETRIES = 3
OPEN_RETRY_DELAY = 0.5

# Audio kept from *before* the hotkey press.
PREROLL_SECONDS = 0.5

# Parakeet's usable attention window.
MAX_SECONDS = 25.0

PARAKEET = "parakeet-tdt-0.6b-v2"
PARAKEET_REPO = "nemo-parakeet-tdt-0.6b-v2"


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """FFT resample."""
    if src == dst:
        return x.astype(np.float32)
    n_in = len(x)
    n_out = int(round(n_in * dst / src))
    if n_in == 0 or n_out == 0:
        return np.zeros(0, dtype=np.float32)
    spec = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=complex)
    keep = min(len(spec), len(out))
    out[:keep] = spec[:keep]
    return (np.fft.irfft(out, n=n_out) * (n_out / n_in)).astype(np.float32)

_model = None
_model_key: str | None = None
_backend = "not loaded"

# One second of silence.
_PROBE = np.zeros(SAMPLE_RATE, dtype=np.float32)


def _load_whisper(model_name: str):
    # CUDA needs cuDNN 9 / cuBLAS DLLs we deliberately don't bundle
    for device, compute in (("cuda", "int8_float16"), ("cpu", "int8")):
        try:
            m = WhisperModel(model_name, device=device, compute_type=compute)
            list(m.transcribe(_PROBE, language="en", vad_filter=False)[0])
        except Exception as e:
            log.warning("whisper unusable on %s: %s", device, e)
            continue
        return m, f"{device}/{compute}"
    raise RuntimeError("could not load whisper on cuda or cpu")


# The onnx-asr name resolves to this HuggingFace repo (see onnx_asr.resolver).
PARAKEET_HF_REPO = "istupakov/parakeet-tdt-0.6b-v2-onnx"
# The int8 weights, vocabulary and config that onnx-asr currently resolves.
PARAKEET_PATTERNS = (
    "config.json",
    "vocab.txt",
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
)
# The denominator when the repo metadata can't be reached.
PARAKEET_TOTAL = 640_000_000


def _ensure_parakeet(progress=None) -> None:
    """Pre-download the parakeet weights, reporting bytes if anyone watches."""
    if progress is None:
        return
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return

    total = 0
    try:
        info = HfApi().model_info(PARAKEET_HF_REPO, files_metadata=True)
        total = sum(
            s.size or 0
            for s in info.siblings
            if s.rfilename in PARAKEET_PATTERNS
        )
        if total <= 0:
            total = PARAKEET_TOTAL
    except Exception as e:
        log.info("could not measure %s (%s); using the known total", PARAKEET_HF_REPO, e)
        total = PARAKEET_TOTAL

    cache = Path(HF_HUB_CACHE) / f"models--{PARAKEET_HF_REPO.replace('/', '--')}"

    def cached() -> int:
        return sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())

    # Report once before the watcher starts.
    progress(PARAKEET_HF_REPO, min(cached(), total), total)

    stop = threading.Event()

    def watch() -> None:
        # The cache directory is the only byte count huggingface_hub exposes without a tqdm hook
        while True:
            try:
                progress(PARAKEET_HF_REPO, min(cached(), total), total)
            except Exception:
                return
            if stop.wait(0.5):
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        snapshot_download(PARAKEET_HF_REPO, allow_patterns=list(PARAKEET_PATTERNS))
    finally:
        stop.set()
    # The cache can already have been complete
    progress(PARAKEET_HF_REPO, total, total)


def _load_parakeet(progress=None):
    """Parakeet via onnx-asr. Prefers the int8 weights; falls back to full."""
    # Imported here, not at module scope
    import onnx_asr

    _ensure_parakeet(progress)

    for quantization in ("int8", None):
        try:
            m = onnx_asr.load_model(PARAKEET_REPO, quantization=quantization)
            m.recognize(_PROBE, sample_rate=SAMPLE_RATE)
        except Exception as e:
            log.warning("parakeet unusable at quantization=%s: %s", quantization, e)
            continue
        return m, f"onnx/{quantization or 'float32'}"
    raise RuntimeError("could not load parakeet")


_inference_lock = threading.RLock()


def load(cfg: dict | None = None, progress=None):
    with _inference_lock:
        return _load(cfg, progress)


def _load(cfg: dict | None = None, progress=None):
    """Load the on-device model once and keep warm."""
    global _model, _model_key, _backend
    cfg = cfg or config.load()
    model_name = cfg["stt"]["local_model"]
    if _model is not None and _model_key == model_name:
        return _model

    if progress and model_name != PARAKEET:
        # faster-whisper downloads through its own hub layer; no hook exists.
        progress(model_name, 0, 0)
    _model, _backend = (
        _load_parakeet(progress)
        if model_name == PARAKEET
        else _load_whisper(model_name)
    )
    _model_key = model_name
    log.info("stt %r on %s", model_name, _backend)
    return _model


def backend() -> str:
    return _backend


def _cloud_transcribe(audio: np.ndarray, section: dict) -> str:
    """Upload the take to an OpenAI-compatible /audio/transcriptions endpoint."""
    global _backend
    import httpx

    url = f"{section['base_url'].rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {section['api_key']}"} if section["api_key"] else {}
    log.info("transcribe via %s model=%s", section["provider"], section["model"])
    response = httpx.post(
        url,
        headers=headers,
        files={"file": ("speech.wav", wav.encode(audio, SAMPLE_RATE), "audio/wav")},
        data={"model": section["model"], "response_format": "json"},
        timeout=CLOUD_TIMEOUT,
    )
    if not response.is_success:
        raise errors.provider_error(
            response.status_code, response.text, section["provider"], section["model"]
        )
    _backend = f"{section['provider']}/{section['model']}"
    return str(response.json().get("text", "")).strip()


def _wasapi() -> dict | None:
    for api in sd.query_hostapis():
        if api["name"] == WASAPI:
            return api
    return None


def refresh_devices() -> bool:
    """Re-enumerate PortAudio's cached device list."""
    try:
        if sd.get_stream().active:
            return False
    except RuntimeError:
        pass  # nothing has played yet
    # ponytail: doesn't see streams built outside sd.play (e.g.
    sd._terminate()
    sd._initialize()
    return True


def _inputs(indices) -> list[int]:
    return [i for i in indices if int(sd.query_devices(i)["max_input_channels"]) > 0]


def _candidates() -> tuple[list[int], int | None]:
    """(input device indices, the default one) - WASAPI's view where there is one."""
    api = _wasapi()
    if api is None:
        return _inputs(range(len(sd.query_devices()))), sd.default.device[0]
    indices = _inputs(api["devices"])
    default = api["default_input_device"]
    # A default that is not in its own host API's input list is stale.
    if default not in indices:
        if default is not None:
            log.warning("host api default %s is stale, ignoring it", default)
        default = indices[0] if indices else None
    return indices, default


def input_devices() -> list[dict]:
    """One entry per real microphone, for the settings window."""
    indices, default = _candidates()
    out, seen = [], set()
    for index in indices:
        device = sd.query_devices(index)
        name = str(device["name"])
        if int(device["max_input_channels"]) <= 0 or name in seen:
            continue
        seen.add(name)
        out.append(
            {
                "name": name,
                "channels": int(device["max_input_channels"]),
                "default_samplerate": int(device["default_samplerate"]),
                "default": index == default,
            }
        )
    return out


def _resolve_all(name: str | None) -> list[int]:
    """Every microphone worth trying, best first."""
    indices, default = _candidates()
    order = []
    if name:
        order = [i for i in indices if str(sd.query_devices(i)["name"]) == name]
        if not order:
            log.warning("microphone %r is not connected, using the default", name)
    if default is not None and default not in order:
        order.append(default)
    return order + [i for i in indices if i not in order]


def _resolve(name: str | None) -> int | None:
    """The single best microphone for `name`. See _resolve_all."""
    order = _resolve_all(name)
    return order[0] if order else None


def choose_channel(audio: np.ndarray, requested: int | None = None) -> tuple[np.ndarray, int]:
    """Pick an explicit channel, or the strongest channel from one recording."""
    if audio.ndim != 2 or audio.shape[1] == 0 or audio.shape[0] == 0:
        raise ValueError("microphone recording has no channels")
    if requested is not None:
        if requested >= audio.shape[1]:
            raise ValueError(
                f"channel {requested + 1} is unavailable; device has {audio.shape[1]}"
            )
        return audio[:, requested], requested
    channel = int(np.argmax(_speech_energy(audio)))
    return audio[:, channel], channel


def _speech_energy(audio: np.ndarray) -> np.ndarray:
    """Per-channel level over the loudest fifth of the take."""
    frames = max(1, min(RANK_FRAMES, len(audio)))
    usable = len(audio) - len(audio) % frames
    blocks = audio[:usable].reshape(frames, -1, audio.shape[1])
    power = np.mean(np.square(blocks, dtype=np.float64), axis=1)
    keep = max(1, int(frames * RANK_LOUDEST))
    return np.sqrt(np.mean(np.sort(power, axis=0)[-keep:], axis=0))


def levels(audio: np.ndarray) -> tuple[float, float]:
    if not audio.size:
        return 0.0, 0.0
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    return peak, rms


def apply_quiet_gain(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Lift quiet accepted speech without ever clipping or amplifying silence."""
    peak, _ = levels(audio)
    if peak < MIN_PEAK or peak >= TARGET_PEAK:
        return audio, 1.0
    gain = min(MAX_GAIN, TARGET_PEAK / peak)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32), gain


class Recorder:
    """Mic capture between push-to-talk press and release."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg
        self._frames: list[np.ndarray] = []
        self._ring: deque[np.ndarray] = deque()
        self._ring_samples = 0
        self._preroll = 0
        self._armed = False
        # The callback runs on PortAudio's thread.
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._device: str | None = None
        # open()/close() run on to_thread pool threads
        self._io = threading.RLock()
        self._rate = SAMPLE_RATE
        # For the log line.
        self._opened_at = time.monotonic()
        self._takes = 0
        # Callback status flags seen during the current take, e.g.
        self._status = ""
        self.last_stats = {
            "seconds": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "channel": 0,
            "device": "",
            "levels": [],
        }

    @property
    def active(self) -> bool:
        return self._armed

    def _capture(self, indata, _frames, _time, status) -> None:
        if status:
            # Now that the stream stays open
            (log.warning if self._armed else log.debug)(
                "microphone callback: %s", status
            )
            if self._armed:
                self._status = str(status)
        block = indata.copy()
        with self._lock:
            if self._armed:
                self._frames.append(block)
            self._ring.append(block)
            self._ring_samples += len(block)
            # Trim only while the *oldest* block is entirely surplus
            while self._ring_samples - len(self._ring[0]) >= self._preroll:
                self._ring_samples -= len(self._ring.popleft())

    def open(self, quiet: bool = False) -> None:
        """Begin filling the pre-roll ring."""
        cfg = self._cfg or config.load()
        name = cfg["stt"].get("input_device")
        with self._io:
            # Compared by name, before anything touches PortAudio
            if self._stream is not None and name == self._device:
                return
            self.close()

            failure = None
            for attempt in range(OPEN_RETRIES):
                if attempt:
                    time.sleep(OPEN_RETRY_DELAY)
                # Re-resolved each round: cheap, and the device list can change.
                for device in _resolve_all(name):
                    try:
                        self._start(device)
                    except sd.PortAudioError as e:
                        # Enumeration cannot see everything: an index can be stale
                        (log.debug if quiet else log.warning)(
                            "microphone %s refused to start: %s", device, e
                        )
                        failure = e
                        continue
                    self._device = name
                    return
        # The raw PortAudio text is already in the log
        raise RuntimeError(
            "the microphone refused to start. try again in a moment"
        ) from failure

    def _start(self, device: int | None) -> None:
        dev = sd.query_devices(device, kind="input")
        self._rate = int(dev["default_samplerate"])
        self._preroll = int(self._rate * PREROLL_SECONDS)

        # Deliberately the default ('high') buffer
        stream = sd.InputStream(
            device=device,
            samplerate=self._rate,
            channels=int(dev["max_input_channels"]),
            dtype="float32",
            callback=self._capture,
        )
        # Assigned only once it is genuinely running.
        try:
            stream.start()
        except BaseException:
            stream.close()
            raise
        self._stream = stream
        self._opened_at = time.monotonic()
        self._takes = 0

        self.last_stats["device"] = str(dev["name"])
        log.info(
            "microphone open: %s at %dHz x%d, %.0fms pre-roll",
            dev["name"],
            self._rate,
            int(dev["max_input_channels"]),
            PREROLL_SECONDS * 1000,
        )

    def reopen(self) -> None:
        """Throw the stream away and get a fresh one."""
        if self._stream is None:
            return
        log.info("reopening the microphone after %d take(s)", self._takes)
        self.close()
        try:
            self.open()
        except Exception as e:
            # Best-effort: this runs mid-handler, and the next keypress opens again anyway.
            log.warning("reopen failed, staying closed: %s", e)

    def close(self) -> None:
        """Release the microphone. Windows stops showing the in-use indicator."""
        with self._io:
            stream, self._stream = self._stream, None
            self._device = None
            with self._lock:
                self._armed = False
                self._frames = []
                self._ring.clear()
                self._ring_samples = 0
            if stream is not None:
                stream.stop()
                stream.close()
                log.info("microphone closed")

    def start(self) -> None:
        """Arm. The take begins PREROLL_SECONDS *before* this call."""
        self.open()
        with self._lock:
            self._frames = list(self._ring)
            self._armed = True

    def stop(self) -> np.ndarray:
        with self._lock:
            if not self._armed:
                return np.zeros(0, dtype=np.float32)
            self._armed = False
            frames, self._frames = self._frames, []
        if not frames:
            return np.zeros(0, dtype=np.float32)
        raw = np.concatenate(frames)
        # Always automatic
        mono, channel = choose_channel(raw)
        audio = resample(mono, self._rate, SAMPLE_RATE)
        peak, rms = levels(audio)
        # Every channel, not just the winner.
        per_channel = [round(float(v), 4) for v in _speech_energy(raw)]
        self._takes += 1
        status, self._status = self._status, ""
        self.last_stats.update(
            {
                "seconds": audio.size / SAMPLE_RATE,
                "peak": peak,
                "rms": rms,
                "channel": channel,
                "levels": per_channel,
            }
        )
        log.info(
            "recorded %.1fs ch=%d/%d peak=%.4f rms=%.4f levels=%s "
            "take #%d on a %.0fs stream%s",
            audio.size / SAMPLE_RATE,
            channel + 1,
            raw.shape[1],
            peak,
            rms,
            per_channel,
            self._takes,
            time.monotonic() - self._opened_at,
            f" status={status}" if status else "",
        )
        return audio


def transcribe(audio: np.ndarray, cfg: dict | None = None) -> str:
    with _inference_lock:
        return _transcribe(audio, cfg)


def transcribe_meeting(audio: np.ndarray, cfg: dict | None = None) -> str:
    """Gate continuous audio without changing push-to-talk conditioning."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    with _inference_lock:
        speech = get_speech_timestamps(audio, VadOptions(min_speech_duration_ms=120, speech_pad_ms=300))
        if not speech:
            return ""
        # Keep the full window: the gate must not cut off words at its boundaries.
        return _transcribe(audio, cfg, meeting=True)


def transcribe_meeting_segments(audio: np.ndarray, cfg: dict | None = None) -> list[dict]:
    """Time meeting contributions by speech onset, not the capture window."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    with _inference_lock:
        regions = get_speech_timestamps(audio, VadOptions(
            min_speech_duration_ms=120, min_silence_duration_ms=450, speech_pad_ms=0,
        ))
        result = []
        padding = round(SAMPLE_RATE * .3)
        for region in regions:
            # Padding protects quiet first/last syllables; ordering uses speech itself.
            first = max(0, region["start"] - padding)
            last = min(len(audio), region["end"] + padding)
            text = _transcribe(audio[first:last], cfg, meeting=True).strip()
            if text:
                result.append({"start": region["start"] / SAMPLE_RATE,
                               "end": region["end"] / SAMPLE_RATE, "text": text})
        return result


def _transcribe(audio: np.ndarray, cfg: dict | None = None, *, meeting: bool = False) -> str:
    if audio.size < SAMPLE_RATE * MIN_SECONDS:
        return ""
    # Replaces vad_filter, which was clipping the first word off utterances.
    peak, rms = levels(audio)
    log.info("%.1fs peak=%.4f rms=%.4f", audio.size / SAMPLE_RATE, peak, rms)
    min_peak = 0.001 if meeting else MIN_PEAK
    if peak < min_peak:
        log.info("too quiet, skipping (peak %.4f < %.4f)", peak, min_peak)
        return ""
    if meeting:
        # Do not undo echo suppression by amplifying quiet residual playback 20x.
        gain = min(2.0, TARGET_PEAK / max(peak, 0.001)) if peak < TARGET_PEAK else 1.0
        conditioned = audio * gain
    else:
        conditioned, gain = apply_quiet_gain(audio)
    if gain > 1:
        log.info("quiet speech gain %.1fx", gain)
    cfg = cfg or config.load()
    if cfg["stt"]["mode"] == "cloud":
        return _cloud_transcribe(conditioned, cfg["stt"])

    model = load(cfg)
    if cfg["stt"]["local_model"] == PARAKEET:
        limit = int(SAMPLE_RATE * MAX_SECONDS)
        if conditioned.size > limit:
            log.warning(
                "%.0fs hold truncated to %.0fs - parakeet's window",
                conditioned.size / SAMPLE_RATE,
                MAX_SECONDS,
            )
            conditioned = conditioned[:limit]
        # English-only and greedy TDT decoding, so no language or beam settings.
        return str(model.recognize(conditioned, sample_rate=SAMPLE_RATE)).strip()

    segments, _ = model.transcribe(
        conditioned,
        language="en",
        beam_size=5,
        vad_filter=False,
        # Stops one bad guess from steering every later segment
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()
