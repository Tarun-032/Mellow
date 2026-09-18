"""Model weights we host ourselves, downloaded on first use."""

import logging
import hashlib
import threading
from pathlib import Path

import httpx

from mellowd.config import CONFIG_DIR

log = logging.getLogger("mellowd.models")

MODELS_DIR = CONFIG_DIR / "models"

_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
URLS = {
    "kokoro-v1.0.onnx": f"{_BASE}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{_BASE}/voices-v1.0.bin",
}
_SEGMENTATION = "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx"
SPEAKER_MODELS = {
    "meeting-segmentation.onnx": (5992913, "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"),
    "wespeaker_en_voxceleb_CAM++_LM.onnx": (29292687, "e197af7e9d473030cf486b3124149a19bf37014d0e4485e4c70c483b0ec10cb2"),
}
URLS.update({"meeting-segmentation.onnx": _SEGMENTATION,
             "wespeaker_en_voxceleb_CAM++_LM.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B_LM.onnx"})
_verified = set()
_locks = {name: threading.Lock() for name in URLS}


def verify(name: str, path: Path | None = None) -> bool:
    path = path or MODELS_DIR / name
    try:
        if not path.is_file():
            return False
        stat = path.stat()
        if name not in SPEAKER_MODELS:
            return stat.st_size > 0
        size, digest = SPEAKER_MODELS[name]
        if stat.st_size != size:
            return False
        signature = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if signature not in _verified:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    return False
            _verified.add(signature)
        return True
    except OSError:
        return False


def available(name: str) -> bool:
    """Whether one complete hosted model file is already on this device."""
    if name not in URLS:
        raise KeyError(f"unknown model file: {name!r}")
    path = MODELS_DIR / name
    try:
        size = path.stat().st_size
        return path.is_file() and (size == SPEAKER_MODELS[name][0] if name in SPEAKER_MODELS else size > 0)
    except OSError:
        # Antivirus, cleanup, or another process may remove the file between the two filesystem checks.
        return False


def ensure(name: str, progress=None, *, cancelled=None) -> Path:
    with _locks[name]:
        return _ensure(name, progress, cancelled)


def _ensure(name: str, progress=None, cancelled=None) -> Path:
    """Return the local path, downloading it if missing."""
    if name not in URLS:
        raise KeyError(f"unknown model file: {name!r}")

    path = MODELS_DIR / name
    if verify(name):
        if progress:
            size = path.stat().st_size
            progress(name, size, size)
        return path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Download to .part and rename only on success
    part = path.with_suffix(path.suffix + ".part")
    log.info("downloading %s ...", name)

    if cancelled and cancelled():
        raise InterruptedError("Model preparation cancelled")
    with httpx.stream("GET", URLS[name], follow_redirects=True, timeout=5.0 if cancelled else 60.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = last_pct = 0
        if progress:
            progress(name, 0, total)
        with part.open("wb") as f:
            for block in r.iter_bytes(1 << 20):
                if cancelled and cancelled():
                    raise InterruptedError("Model preparation cancelled")
                f.write(block)
                done += len(block)
                if progress:
                    progress(name, done, total)
                if total and (pct := done * 100 // total) >= last_pct + 10:
                    last_pct = pct
                    log.info("  %s %d%% (%.0f/%.0f MB)", name, pct, done / 1e6, total / 1e6)

    if cancelled and cancelled():
        raise InterruptedError("Model preparation cancelled")
    if not verify(name, part):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Model integrity check failed: {name}")
    part.replace(path)
    log.info("downloaded %s (%.0f MB)", name, path.stat().st_size / 1e6)
    return path
