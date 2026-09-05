"""WASAPI microphone + loopback capture. Audio remains in bounded RAM."""

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from mellowd.meeting_aec import EchoCanceller, RATE as AEC_RATE, SAMPLES as AEC_SAMPLES

RATE = 16000
CHUNK_SECONDS = 8
OVERLAP_SECONDS = 0.4


@dataclass
class Chunk:
    speaker: str
    start: float
    end: float
    audio: np.ndarray
    overlap: bool = False


def _library():
    try:
        import pyaudiowpatch as pa
        return pa
    except ImportError as exc:
        raise RuntimeError("Meeting audio is unavailable. Install the latest Mellow build for Windows.") from exc


def devices():
    pa = _library()
    with pa.PyAudio() as audio:
        host = audio.get_host_api_info_by_type(pa.paWASAPI)
        inputs = [d for d in audio.get_device_info_generator_by_host_api(host_api_index=host["index"])
                  if d["maxInputChannels"] and not d.get("isLoopbackDevice")]
        outputs = list(audio.get_loopback_device_info_generator())
        default_out = audio.get_default_wasapi_loopback()["index"]
        return {
            "inputs": [{"id": d["index"], "name": d["name"], "default": d["index"] == host["defaultInputDevice"]} for d in inputs],
            "outputs": [{"id": d["index"], "name": d["name"], "default": d["index"] == default_out} for d in outputs],
        }


def mono16(raw: bytes, channels: int, rate: int, target: int = RATE) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768.0
    if rate != target and len(data):
        count = round(len(data) * target / rate)
        data = np.interp(np.arange(count) * rate / target, np.arange(len(data)), data).astype(np.float32)
    return data


class Capture:
    def __init__(self, on_chunk, on_error, origin: float, offset: float = 0):
        self.on_chunk = on_chunk
        self.on_error = on_error
        self.origin = origin
        self.offset = offset
        self.audio = None
        self.streams = []
        self.threads = []
        self.stopped = threading.Event()
        self.failed = threading.Event()
        self.levels = {"You": 0.0, "Other participants": 0.0}
        self.aec = None
        self.pending = queue.Queue(maxsize=100)

    def start(self, microphone: int | None, output: int | None, preferred_mic: str | None = None):
        pa = _library()
        self.audio = pa.PyAudio()
        try:
            self.aec = EchoCanceller()
            host = self.audio.get_host_api_info_by_type(pa.paWASAPI)
            if microphone is None and preferred_mic:
                matches = [d for d in self.audio.get_device_info_generator()
                           if d["hostApi"] == host["index"] and d["maxInputChannels"]
                           and not d.get("isLoopbackDevice") and d["name"] == preferred_mic]
                if matches:
                    microphone = matches[0]["index"]
            mic = self.audio.get_device_info_by_index(microphone if microphone is not None else host["defaultInputDevice"])
            out = self.audio.get_device_info_by_index(output) if output is not None else self.audio.get_default_wasapi_loopback()
            if mic["hostApi"] != host["index"] or mic.get("isLoopbackDevice") or not mic["maxInputChannels"]:
                raise RuntimeError("Select a Windows microphone, then try again.")
            if not out.get("isLoopbackDevice"):
                raise RuntimeError("Select a Windows loopback output, then try again.")
            for speaker, device in (("You", mic), ("Other participants", out)):
                rate, channels = int(device["defaultSampleRate"]), int(device["maxInputChannels"])
                frames = max(1, round(rate / 10))
                def callback(data, count, timing, flags, who=speaker, hz=rate, nch=channels):
                    if flags:
                        self.fail(f"Audio from {who} was interrupted. Check the device and resume.")
                    try:
                        now = time.monotonic()
                        age = timing.get("current_time", 0) - timing.get("input_buffer_adc_time", 0)
                        start = now - (age if 0 < age < 1 else count / hz)
                        self.pending.put_nowait((who, data, nch, hz, start))
                    except queue.Full:
                        self.fail("Audio capture fell behind. Recording paused; check your computer's load.")
                    return (None, pa.paComplete if self.stopped.is_set() else pa.paContinue)

                stream = self.audio.open(format=pa.paInt16, channels=channels, rate=rate, input=True,
                                         input_device_index=int(device["index"]), frames_per_buffer=frames,
                                         stream_callback=callback, start=False)
                self.streams.append(stream)
            self.epoch = time.monotonic()
            worker = threading.Thread(target=self.collect, daemon=True)
            self.threads.append(worker)
            worker.start()
            for stream in self.streams:
                stream.start_stream()
        except Exception:
            self.close()
            raise

    def fail(self, message: str):
        if not self.failed.is_set():
            self.failed.set()
            self.stopped.set()
            self.on_error(message)

    def collect(self):
        frames = {"You": {}, "Other participants": {}}
        cursors = {}
        watermarks = {who: 0 for who in frames}
        next_slot = 0
        last_mic = time.monotonic()
        chunks = {who: ChunkBuffer(who, self.on_chunk) for who in frames}
        silence = np.zeros(AEC_SAMPLES, dtype=np.float32)
        try:
            while not self.stopped.is_set() or not self.pending.empty() or any(frames.values()):
                try:
                    who, raw, channels, rate, captured = self.pending.get(timeout=0.02)
                    observed = round((captured - self.epoch) * AEC_RATE)
                    sample = cursors.get(who, observed)
                    # Re-align after genuine device gaps, not callback scheduling jitter.
                    if abs(observed - sample) > 2 * AEC_SAMPLES:
                        sample = observed
                    data = mono16(raw, channels, rate, AEC_RATE)
                    cursors[who] = sample + len(data)
                    if sample < 0:
                        data, sample = data[-sample:], 0
                    if sample < next_slot * AEC_SAMPLES:
                        raise RuntimeError("Meeting audio arrived too late to cancel speaker echo")
                    watermarks[who] = sample + len(data)
                    # Preserve sub-frame timing between independently started devices.
                    # Merely rounding callbacks to 100ms bins misaligns their audio.
                    while len(data):
                        slot, offset = divmod(sample, AEC_SAMPLES)
                        take = min(len(data), AEC_SAMPLES - offset)
                        target = frames[who].setdefault(slot, np.zeros(AEC_SAMPLES, dtype=np.float32))
                        target[offset:offset+take] = data[:take]
                        sample += take
                        data = data[take:]
                    if sum(map(len, frames.values())) > 100:
                        raise RuntimeError("Meeting echo cancellation fell behind. Recording paused; check your computer's load.")
                    if who == "You":
                        last_mic = time.monotonic()
                except queue.Empty:
                    if not self.stopped.is_set() and time.monotonic() - last_mic > 5:
                        self.fail("The microphone stopped delivering audio. Check it and resume.")
                # Drain callbacks before deciding a missing reference is silence.
                if not self.pending.empty():
                    continue
                latest = max((max(source, default=-1) for source in frames.values()), default=-1)
                while next_slot <= latest:
                    paired = all(end >= (next_slot + 1) * AEC_SAMPLES for end in watermarks.values())
                    expired = time.monotonic() >= self.epoch + (next_slot + 1) / 10 + 0.25
                    if not paired and not expired and not self.stopped.is_set():
                        break
                    mic = frames["You"].pop(next_slot, silence)
                    playback = frames["Other participants"].pop(next_slot, silence)
                    cleaned = self.aec.process_pair(mic, playback)
                    begin = max(0, self.epoch - self.origin + self.offset + next_slot / 10)
                    valid = AEC_SAMPLES
                    if self.stopped.is_set():
                        valid = min(valid, max(watermarks.values()) - next_slot * AEC_SAMPLES)
                    for who, audio in (("You", cleaned), ("Other participants", playback)):
                        self.levels[who] = float(np.max(np.abs(audio)))
                        # 24kHz -> 16kHz, preserving the common recording clock.
                        pcm = (np.clip(audio[:valid], -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
                        chunks[who].append(mono16(pcm, 1, AEC_RATE), begin)
                    next_slot += 1
        except Exception as exc:
            self.fail(str(exc) if isinstance(exc, RuntimeError) else "Audio capture failed. Check your devices, then resume.")
        finally:
            for chunk in chunks.values():
                chunk.flush()
            self.levels = {who: 0 for who in frames}

    def healthy(self):
        if self.stopped.is_set():
            return False
        try:
            return all(s.is_active() for s in self.streams)
        except Exception:
            return False

    def close(self):
        self.stopped.set()
        for stream in self.streams:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        for worker in self.threads:
            worker.join(timeout=3)
        if self.aec is not None:
            self.aec.close()
            self.aec = None
        for worker in self.threads:
            worker.join(timeout=2)
        self.streams.clear()
        self.threads.clear()
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None


class ChunkBuffer:
    def __init__(self, speaker, emit):
        self.speaker, self.emit = speaker, emit
        self.parts, self.size, self.begin, self.overlap = [], 0, 0.0, False

    def append(self, data, begin):
        if not self.parts:
            self.begin = begin
        self.parts.append(data)
        self.size += len(data)
        if self.size >= CHUNK_SECONDS * RATE:
            joined = np.concatenate(self.parts)
            self.emit(Chunk(self.speaker, self.begin, self.begin + self.size / RATE, joined, self.overlap))
            tail = joined[-round(OVERLAP_SECONDS * RATE):].copy()
            self.begin += (self.size - len(tail)) / RATE
            self.parts, self.size, self.overlap = [tail], len(tail), True

    def flush(self):
        if self.size > (OVERLAP_SECONDS * RATE if self.overlap else RATE * 0.3):
            self.emit(Chunk(self.speaker, self.begin, self.begin + self.size / RATE,
                            np.concatenate(self.parts), self.overlap))
        self.parts, self.size = [], 0
