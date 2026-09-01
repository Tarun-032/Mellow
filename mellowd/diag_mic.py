"""Diagnose why transcription is inaccurate."""

import sys
import time

import numpy as np
import sounddevice as sd

from mellowd import config, stt

SECONDS = 6
TARGET = stt.SAMPLE_RATE
SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank"

# The two engines worth comparing head to head: the fast default
ENGINES = (stt.PARAKEET, "distil-large-v3")


resample = stt.resample  # test the shipped resampler, not a stand-in


def stats(name: str, x: np.ndarray) -> None:
    if x.size == 0:
        print(f"  {name:<28} empty")
        return
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    dbfs = 20 * np.log10(rms) if rms > 0 else -999
    print(f"  {name:<28} rms={rms:.4f} ({dbfs:6.1f} dBFS)  peak={peak:.3f}")


def say(label: str, audio: np.ndarray, cfg: dict) -> tuple[str, float]:
    """Transcribe with whichever engine `cfg` selects, and time it."""
    model = stt.load(cfg)
    t0 = time.perf_counter()
    if cfg["stt"]["local_model"] == stt.PARAKEET:
        text = str(model.recognize(audio, sample_rate=TARGET)).strip()
    else:
        segs, _ = model.transcribe(
            audio, language="en", vad_filter=False, beam_size=5
        )
        text = " ".join(s.text.strip() for s in segs).strip()
    dt = time.perf_counter() - t0
    print(f"  {label:<28} {dt:5.2f}s  {text!r}")
    return text, dt


def main() -> None:
    cfg = config.load()
    device = cfg["stt"].get("input_device")
    dev = sd.query_devices(device, kind="input")
    native = int(dev["default_samplerate"])
    chans = int(dev["max_input_channels"])
    print(f"device : {dev['name']}")
    print(f"native : {native} Hz, {chans} channels")
    print(f"model  : {cfg['stt']['local_model']}\n")

    print(f'Say this clearly, {SECONDS}s:\n\n    "{SENTENCE}"\n')
    input("press enter to start recording... ")

    # One take for every comparison.
    print(f"recording {SECONDS}s at {native}Hz x{chans} (native)...")
    nat = sd.rec(
        SECONDS * native,
        samplerate=native,
        channels=chans,
        dtype="float32",
        device=device,
    )
    sd.wait()

    print("\n--- levels ---")
    converted = []
    for c in range(chans):
        stats(f"native ch{c}", nat[:, c])
        converted.append(resample(nat[:, c], native, TARGET))
    selected, channel = stt.choose_channel(nat, cfg["stt"].get("input_channel"))
    selected = resample(selected, native, TARGET)
    gained, gain = stt.apply_quiet_gain(selected)
    stats(f"selected ch{channel}", selected)

    print(f"\n--- transcripts, {cfg['stt']['local_model']} (target: {SENTENCE!r}) ---")
    for channel, audio in enumerate(converted):
        say(f"channel {channel + 1}", audio, cfg)
    say("automatic selection", selected, cfg)
    say(f"bounded gain ({gain:.1f}x)", gained, cfg)

    # The engine comparison, on the one take that already survived selection.
    print("\n--- engine comparison (same audio, best channel) ---")
    for name in ENGINES:
        try:
            say(name, gained, {**cfg, "stt": {**cfg["stt"], "local_model": name}})
        except Exception as e:  # a missing model must not kill the diagnostic
            print(f"  {name:<28} unavailable: {e}")

    print(
        "\nread it like this:\n"
        "  one channel clearly wins             -> select it in Mellow settings\n"
        "  bounded gain clearly wins            -> input level is too low\n"
        "  one engine clearly wins on YOUR voice-> select it in Mellow settings\n"
        "  all channels are quiet or inaccurate -> choose another device or a larger model\n"
    )


if __name__ == "__main__":
    sys.exit(main())
