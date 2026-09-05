"""Small, model-free proof that the frozen sidecar contains its native stack."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def run() -> None:
    # Imports here are deliberate.
    import ctranslate2
    import espeakng_loader
    import faster_whisper
    import kokoro_onnx
    import mss
    import onnx_asr
    import onnxruntime
    import sounddevice
    import pyaudiowpatch
    import uiautomation
    from PIL import Image
    from pycaw.pycaw import AudioUtilities
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import BitmapDecoder
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.storage.streams import InMemoryRandomAccessStream

    checks: dict[str, object] = {}
    checks["portaudio"] = sounddevice.get_portaudio_version()[1]
    checks["meeting_portaudio"] = pyaudiowpatch.get_portaudio_version_text()
    from mellowd.meetings import router
    checks["meeting_routes"] = len(router.routes)

    espeak_library = Path(espeakng_loader.get_library_path())
    espeak_data = Path(espeakng_loader.get_data_path())
    if not espeak_library.is_file() or not espeak_data.is_dir():
        raise RuntimeError("the packaged espeak-ng runtime is incomplete")
    if espeakng_loader.load_library() is None:
        raise RuntimeError("the packaged espeak-ng DLL did not load")
    checks["espeak"] = espeak_library.name

    preprocessor = Path(onnx_asr.__file__).parent / "preprocessors" / "data" / "nemo128.onnx"
    if not preprocessor.is_file():
        raise RuntimeError("onnx-asr preprocessing data is missing")
    checks["onnx_asr_data"] = preprocessor.name

    # GetRootControl constructs the real COM client.
    root = uiautomation.GetRootControl()
    if root is None:
        raise RuntimeError("Windows UI Automation did not return a root control")
    checks["uia"] = root.ControlTypeName

    # Prove the generated WinRT extensions were collected.
    checks["winrt"] = all(
        item is not None
        for item in (Language, BitmapDecoder, OcrEngine, InMemoryRandomAccessStream)
    )
    checks["audio_sessions"] = len(AudioUtilities.GetAllSessions())
    checks["pillow"] = Image.__version__
    checks["mss"] = getattr(mss, "__version__", "present")
    checks["onnxruntime"] = onnxruntime.__version__
    checks["ctranslate2"] = ctranslate2.__version__
    checks["faster_whisper"] = faster_whisper.__version__
    checks["kokoro_onnx"] = getattr(kokoro_onnx, "__version__", "present")

    # The release's hard size/security boundary
    if importlib.util.find_spec("torch") is not None:
        raise RuntimeError("torch is present in the sidecar package")
    checks["torch"] = False

    print(json.dumps({"ok": True, "checks": checks}, sort_keys=True))
