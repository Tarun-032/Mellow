"""Model weights we host ourselves, downloaded on first use."""

import logging
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


def available(name: str) -> bool:
    """Whether one complete hosted model file is already on this device."""
    if name not in URLS:
        raise KeyError(f"unknown model file: {name!r}")
    path = MODELS_DIR / name
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        # Antivirus, cleanup, or another process may remove the file between the two filesystem checks.
        return False


def ensure(name: str, progress=None) -> Path:
    """Return the local path, downloading it if missing."""
    if name not in URLS:
        raise KeyError(f"unknown model file: {name!r}")

    path = MODELS_DIR / name
    if available(name):
        if progress:
            size = path.stat().st_size
            progress(name, size, size)
        return path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Download to .part and rename only on success
    part = path.with_suffix(path.suffix + ".part")
    log.info("downloading %s ...", name)

    with httpx.stream("GET", URLS[name], follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = last_pct = 0
        if progress:
            progress(name, 0, total)
        with part.open("wb") as f:
            for block in r.iter_bytes(1 << 20):
                f.write(block)
                done += len(block)
                if progress:
                    progress(name, done, total)
                if total and (pct := done * 100 // total) >= last_pct + 10:
                    last_pct = pct
                    log.info("  %s %d%% (%.0f/%.0f MB)", name, pct, done / 1e6, total / 1e6)

    part.replace(path)
    log.info("downloaded %s (%.0f MB)", name, path.stat().st_size / 1e6)
    return path
