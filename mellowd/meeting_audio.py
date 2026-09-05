"""WASAPI microphone + loopback capture. Audio remains in bounded RAM."""

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

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


def mono16(raw: bytes, channels: int, rate: int) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768.0
    if rate != RATE and len(data):
        count = round(len(data) * RATE / rate)
        data = np.interp(np.arange(count) * rate / RATE, np.arange(len(data)), data).astype(np.float32)
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
        self.levels = {"You": 0.0, "Other participants": 0.0}

    def start(self, microphone: int | None, output: int | None, preferred_mic: str | None = None):
        pa = _library()
        self.audio = pa.PyAudio()
        try:
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
                pending = queue.Queue(maxsize=50)

                def callback(data, count, timing, flags, q=pending, who=speaker):
                    if flags:
                        self.fail(f"Audio from {who} was interrupted. Check the device and resume.")
                    try:
                        q.put_nowait((data, time.monotonic()))
                    except queue.Full:
                        self.fail("Audio capture fell behind. Recording paused; check your computer's load.")
                    return (None, pa.paComplete if self.stopped.is_set() else pa.paContinue)

                stream = self.audio.open(format=pa.paInt16, channels=channels, rate=rate, input=True,
                                         input_device_index=int(device["index"]), frames_per_buffer=frames,
                                         stream_callback=callback, start=False)
                self.streams.append(stream)
                worker = threading.Thread(target=self.collect, args=(speaker, pending, channels, rate), daemon=True)
                self.threads.append(worker)
                worker.start()
            for stream in self.streams:
                stream.start_stream()
        except Exception:
            self.close()
            raise

    def fail(self, message: str):
        if not self.stopped.is_set():
            self.stopped.set()
            self.on_error(message)

    def collect(self, speaker, pending, channels, rate):
        parts, size, begin, overlap = [], 0, 0.0, False
        last_data = time.monotonic()
        try:
            while not self.stopped.is_set() or not pending.empty():
                try:
                    raw, captured = pending.get(timeout=0.2)
                except queue.Empty:
                    # Silent loopback may not deliver callbacks; only mic loss is unambiguous here.
                    if speaker == "You" and time.monotonic() - last_data > 5:
                        self.fail("The microphone stopped delivering audio. Check it and resume.")
                    continue
                if captured - last_data > 0.5 and parts:
                    if size > (OVERLAP_SECONDS * RATE if overlap else RATE * 0.3):
                        self.on_chunk(Chunk(speaker, begin, begin + size / RATE, np.concatenate(parts), overlap))
                    parts, size, overlap = [], 0, False
                last_data = captured
                data = mono16(raw, channels, rate)
                self.levels[speaker] = float(np.max(np.abs(data))) if len(data) else 0
                if not parts:
                    begin = max(0, captured - self.origin + self.offset - len(data) / RATE)
                parts.append(data)
                size += len(data)
                if size >= CHUNK_SECONDS * RATE:
                    joined = np.concatenate(parts)
                    self.on_chunk(Chunk(speaker, begin, begin + size / RATE, joined, overlap))
                    tail = joined[-round(OVERLAP_SECONDS * RATE):].copy()
                    begin += (size - len(tail)) / RATE
                    parts, size, overlap = [tail], len(tail), True
            if size > (OVERLAP_SECONDS * RATE if overlap else RATE * 0.3):
                self.on_chunk(Chunk(speaker, begin, begin + size / RATE, np.concatenate(parts), overlap))
        except Exception:
            self.fail("Audio capture failed. Check your devices, then resume.")
        finally:
            self.levels[speaker] = 0

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
        self.streams.clear()
        self.threads.clear()
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None
