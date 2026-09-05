from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


root = Path(SPECPATH)
datas = []
binaries = []
hiddenimports = []
datas += copy_metadata("PyAudioWPatch")

# Collect packages whose native/data dependencies load lazily.
for package in (
    "ctranslate2",
    "espeakng_loader",
    "kokoro_onnx",
    "language_tags",
    "onnx_asr",
    "phonemizer",
    "pyaudiowpatch",
    "uiautomation",
    "winsdk",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    [str(root / "mellowd" / "__main__.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "torchaudio"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="mellowd",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Console mode exposes --package-check; Rust hides it during normal startup.
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="mellowd",
)
