"""Bounded, read-only observation while a visible guide target is pending."""
from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
from dataclasses import dataclass, field
import time
import uuid
from typing import Callable

import numpy as np
from PIL import Image

POLL = .05
TIMEOUT = 120.0
MAX_SEGMENTS = 8

# Current native target flights take at most 1.1s. This is a failure deadline,
# not a pause added to every sentence. No model retry is made on expiry.
ARRIVAL_TIMEOUT = 2.5


@dataclass
class Presentation:
    """One dispatched pointer, owned by one connection and playback sequence."""

    mark: Callable[[str], None] = field(default=lambda _: None, repr=False)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    deadline: float = field(default_factory=lambda: time.monotonic() + ARRIVAL_TIMEOUT)
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    outcome: str | None = None
    retired: bool = False

    def acknowledge(self, outcome: str) -> None:
        if self.retired or self.outcome is not None or outcome not in ("arrived", "failed"):
            return
        self.outcome = outcome if time.monotonic() <= self.deadline else "timeout"
        self.mark("pointer_arrival_received" if self.outcome == "arrived" else f"pointer_{self.outcome}")
        self.settled.set()

    def retire(self) -> None:
        # Even an already-arrived ticket cannot release old queued speech.
        self.retired = True
        self.settled.set()

    async def wait(self) -> bool:
        if not self.settled.is_set():
            try:
                await asyncio.wait_for(self.settled.wait(), max(0, self.deadline - time.monotonic()))
            except TimeoutError:
                self.outcome = "timeout"
                self.mark("pointer_timeout")
                self.settled.set()
        if self.retired:
            raise asyncio.CancelledError
        if self.outcome == "arrived":
            self.mark("pointer_playback_ready")
            return True
        return False


@dataclass(frozen=True)
class Continuation:
    after_bone: int
    remaining: str
    expected_change: str
    trigger: str
    handover: str
    instruction: str = ""


SCHEMA = {
    "type": "object",
    "properties": {
        "after_bone": {"type": "integer", "minimum": 1, "maximum": 3},
        "remaining": {"type": "string"},
        "expected_change": {"type": "string"},
        "trigger": {"type": "string", "enum": ["click", "hover"]},
        "handover": {"type": "string"},
        "instruction": {"type": "string"},
    },
    "required": ["after_bone", "remaining", "expected_change", "trigger", "handover", "instruction"],
    "additionalProperties": False,
}


def parse(value, *, after_bone: int | None = None) -> Continuation | None:
    if not isinstance(value, dict):
        return None
    # A caller with an actual pointer sequence owns its ordinal. An E-row
    # number from the model is not a step count and must not invalidate it.
    count = after_bone if after_bone is not None else value.get("after_bone")
    trigger = value.get("trigger")
    trigger = trigger.strip().lower() if isinstance(trigger, str) else trigger
    if type(count) is not int or not 1 <= count <= 3 or trigger not in ("click", "hover"):
        return None
    fields = [value.get(key) for key in ("remaining", "expected_change", "handover")]
    if any(not isinstance(text, str) or not text.strip() or len(text) > 400 for text in fields):
        return None
    instruction = value.get("instruction", "")
    if not isinstance(instruction, str) or len(instruction) > 400:
        return None
    return Continuation(count, fields[0].strip(), fields[1].strip(), trigger, fields[2].strip(), instruction.strip())


def invalid_fields(value) -> dict:
    """Diagnostic shapes, never screen content or generated narration."""
    if not isinstance(value, dict):
        return {"continue_after": type(value).__name__}
    problems = {}
    trigger = value.get("trigger")
    if not isinstance(trigger, str) or trigger.strip().lower() not in ("click", "hover"):
        problems["trigger"] = {"type": type(trigger).__name__, "allowed_value": False}
    for key in ("remaining", "expected_change", "handover", "instruction"):
        text = value.get(key)
        if not isinstance(text, str) or not text.strip() or len(text) > 400:
            problems[key] = {"type": type(text).__name__,
                             "length": len(text) if isinstance(text, str) else None}
    return problems


@dataclass
class Pending:
    goal: str
    spec: Continuation
    target: object
    shot: object
    partial: dict
    segments: int = 1
    explained: list[str] = field(default_factory=list)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    observer: asyncio.Task | None = None
    deadline: float = field(default_factory=lambda: time.monotonic() + TIMEOUT)


def _sample():
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.wintypes.HWND
    cursor = ctypes.wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(cursor))
    return (bool(user32.GetAsyncKeyState(1) & 0x8000),
            bool(user32.GetAsyncKeyState(13) & 0x8000 or user32.GetAsyncKeyState(32) & 0x8000),
            cursor.x, cursor.y, user32.GetForegroundWindow())


def _focused_point():
    try:
        import uiautomation as auto
        with auto.UIAutomationInitializerInThread(debug=False):
            box = auto.GetFocusedControl().BoundingRectangle
            return ((box.left + box.right) / 2, (box.top + box.bottom) / 2)
    except Exception:
        return None


def inside(bounds, x, y):
    left, top, width, height = bounds
    return left <= x <= left + width and top <= y <= top + height


def same_app(first, second):
    if first == second:
        return True
    def pid(hwnd):
        value = ctypes.wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(ctypes.wintypes.HWND(hwnd), ctypes.byref(value))
        return value.value
    owner = pid(first)
    return bool(owner and owner == pid(second))


async def interaction(target, hwnd, trigger, timeout=TIMEOUT) -> bool:
    """New click / focused-key activation, or a declared submenu hover dwell.

    This does not assert completion. A fresh screen must subsequently establish
    the expected result. No global hooks, clicks, typing, or background polling.
    """
    if not target.bounds:
        return False
    end = time.monotonic() + timeout
    was_mouse, was_key, *_ = await asyncio.to_thread(_sample)
    entered = None
    while time.monotonic() < end:
        await asyncio.sleep(POLL)
        mouse, key, x, y, foreground = await asyncio.to_thread(_sample)
        clicked, keyed = mouse and not was_mouse, key and not was_key
        was_mouse, was_key = mouse, key
        if not same_app(hwnd, foreground):
            entered = None
            continue
        if keyed:
            focused = await asyncio.to_thread(_focused_point)
            if focused is not None:
                x, y = focused
                clicked = True
        hit = inside(target.bounds, x, y)
        if hit and clicked:
            return True
        if hit and trigger == "hover":
            entered = entered or time.monotonic()
            if time.monotonic() - entered >= .45:
                return True
        else:
            entered = None
    return False


def changed(before, after) -> bool:
    """Meaningful change anywhere in the app, including a newly opened menu.

    A menu can appear far from its trigger, whose own pixels stay identical.
    This is only a cheap gate; the model verifies the declared expected state.
    """
    if before.monitor != after.monitor or not same_app(before.hwnd, after.hwnd):
        return False
    def small(shot):
        return np.asarray(Image.fromarray(shot.pixels).convert('RGB').resize((160, 90)), dtype=np.int16)
    delta = np.max(np.abs(small(before) - small(after)), axis=2)
    return bool(np.count_nonzero(delta > 25) >= 45)
