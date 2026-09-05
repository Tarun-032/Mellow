"""Bounded, hidden IPC to OpenWhispr's pinned WebRTC AEC3 helper."""

import hashlib
import json
import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

RATE = 24000
SAMPLES = RATE // 10
EXE = "meeting-aec-helper-win32-x64.exe"
SHA256 = "728802049292f3ae1f5cffdabfe42ff4adde299cec619a916fe1ce88c7570b76"


def binary_path():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "meeting-aec" / EXE
    return Path(__file__).resolve().parents[1] / "build" / "meeting-aec" / EXE


class EchoCanceller:
    def __init__(self):
        path = binary_path()
        if not path.is_file():
            raise RuntimeError("Meeting echo cancellation is missing. Reinstall Mellow; developers: run scripts/prepare-meeting-aec.py.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Meeting echo cancellation failed its integrity check. Reinstall Mellow; developers: rerun scripts/prepare-meeting-aec.py.")
        self.responses = queue.Queue(maxsize=4)
        self.ready = queue.Queue(maxsize=1)
        self.requests = queue.Queue(maxsize=2)
        self.closed = threading.Event()
        self.close_lock = threading.Lock()
        self.process = subprocess.Popen(
            [str(path), "--sample-rate", str(RATE)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.readers = [threading.Thread(target=target, daemon=True)
                        for target in (self._read_audio, self._read_status, self._write_audio)]
        for reader in self.readers:
            reader.start()
        try:
            if self.ready.get(timeout=3) != "start":
                raise RuntimeError("Meeting echo cancellation could not start")
        except Exception:
            self.close()
            raise RuntimeError("Meeting echo cancellation could not start. Reinstall Mellow and try again.") from None

    @staticmethod
    def _read_exact(stream, size):
        result = bytearray()
        while len(result) < size:
            part = stream.read(size - len(result))
            if not part:
                raise EOFError("Meeting echo cancellation stopped")
            result.extend(part)
        return bytes(result)

    def _read_audio(self):
        try:
            while True:
                size, = struct.unpack("<I", self._read_exact(self.process.stdout, 4))
                if size != SAMPLES * 2:
                    raise RuntimeError("Invalid meeting echo-cancellation frame")
                self.responses.put_nowait(self._read_exact(self.process.stdout, size))
        except Exception as exc:
            try:
                self.responses.put_nowait(exc)
            except queue.Full:
                pass

    def _read_status(self):
        try:
            while line := self.process.stderr.readline(4096):
                try:
                    message = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if message.get("type") in {"start", "error", "warning"}:
                    try:
                        self.ready.put_nowait(message["type"])
                    except queue.Full:
                        pass
                    if message["type"] != "start":
                        try:
                            self.responses.put_nowait(RuntimeError("Meeting echo cancellation reported an audio error"))
                        except queue.Full:
                            pass
        except (OSError, ValueError):
            pass

    def _write_audio(self):
        try:
            while not self.closed.is_set():
                try:
                    data = self.requests.get(timeout=.2)
                except queue.Empty:
                    continue
                frame = memoryview(data)
                while frame:
                    written = self.process.stdin.write(frame)
                    if not written:
                        raise EOFError("Meeting echo cancellation stopped")
                    frame = frame[written:]
        except Exception as exc:
            try:
                self.responses.put_nowait(exc)
            except queue.Full:
                pass

    def process_pair(self, microphone, playback):
        """One time-aligned 100ms pair; only the microphone is modified."""
        if len(microphone) != SAMPLES or len(playback) != SAMPLES:
            raise ValueError("Meeting AEC requires paired 100ms frames")
        try:
            frames = []
            for source, audio in ((1, playback), (2, microphone)):
                pcm = (np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
                frames.append(struct.pack("<BI", source, len(pcm)) + pcm)
            self.requests.put_nowait(b"".join(frames))
            response = self.responses.get(timeout=2)
            if isinstance(response, Exception):
                raise response
            return np.frombuffer(response, dtype="<i2").astype(np.float32) / 32768
        except Exception as exc:
            self.close()
            raise RuntimeError("Meeting echo cancellation stopped responding. Check your audio devices, then resume.") from exc

    def close(self):
        with self.close_lock:
            if self.closed.is_set():
                return
            self.closed.set()
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            for reader in self.readers:
                if reader is not threading.current_thread():
                    reader.join(timeout=1)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()
