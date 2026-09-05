"""Fetch the pinned Windows meeting AEC helper and its redistribution notices."""

import base64
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "build" / "meeting-aec"
TAG = "meeting-aec-helper-v1.0.0"
REPO = "https://github.com/OpenWhispr/openwhispr"
ARCHIVE_SHA256 = "576b04b2ff0fcb2562cc217b8f7cf71792b483a70bb8404e5523a26fed0ea9a3"
EXE = "meeting-aec-helper-win32-x64.exe"
WEBRTC = "08f235eba0c247f8929045adb090d0b0445cf8ea"
ABSEIL = "9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f"
WEBRTC_NOTICES = "09385480af77e5fc9529349d6d49b56ffd63e5c1"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mellow-build"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "helper.zip"
    data = archive.read_bytes() if archive.exists() else b""
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        data = fetch(f"{REPO}/releases/download/{TAG}/meeting-aec-helper-win32-x64.zip")
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Meeting AEC download checksum mismatch; refusing to package it")
    archive.write_bytes(data)
    with zipfile.ZipFile(io.BytesIO(data)) as package:
        entry = package.getinfo(EXE)
        if entry.file_size > 10_000_000:
            raise RuntimeError("Unexpected meeting AEC executable size")
        executable = package.read(entry)
        target = DEST / EXE
        if not target.is_file() or target.read_bytes() != executable:
            target.write_bytes(executable)
    notices = {
        "OpenWhispr-LICENSE.txt": (f"https://raw.githubusercontent.com/OpenWhispr/openwhispr/{TAG}/LICENSE", False),
        # The ChromeOS source subset omits the root upstream notices.
        "WebRTC-COPYING.txt": (f"https://webrtc.googlesource.com/src/+/{WEBRTC_NOTICES}/LICENSE?format=TEXT", True),
        "WebRTC-PATENTS.txt": (f"https://webrtc.googlesource.com/src/+/{WEBRTC_NOTICES}/PATENTS?format=TEXT", True),
        "PFFFT-LICENSE.txt": (f"https://chromium.googlesource.com/chromiumos/third_party/webrtc-apm/+/{WEBRTC}/third_party/pffft/LICENSE?format=TEXT", True),
        "RNNoise-LICENSE.txt": (f"https://chromium.googlesource.com/chromiumos/third_party/webrtc-apm/+/{WEBRTC}/third_party/rnnoise/COPYING?format=TEXT", True),
        "Abseil-LICENSE.txt": (f"https://raw.githubusercontent.com/abseil/abseil-cpp/{ABSEIL}/LICENSE", False),
        "Silero-LICENSE.txt": ("https://raw.githubusercontent.com/snakers4/silero-vad/v5.1/LICENSE", False),
    }
    for name, (url, encoded) in notices.items():
        path = DEST / name
        if not path.is_file():
            content = fetch(url)
            path.write_bytes(base64.b64decode(content) if encoded else content)
    print(f"Prepared {TAG}: {DEST}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"Meeting AEC preparation failed: {exc}")
