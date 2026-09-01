"""Screen capture for step 13: one JPEG of the monitor being worked on."""

import ctypes
import ctypes.wintypes
import logging
import re

from mellowd import config

log = logging.getLogger(__name__)

# Practical provider limit for the screenshot's long edge.
MAX_EDGE = 2048
JPEG_QUALITY = 85

# What a *pointing* turn gets instead, and it is smaller on purpose.
POINT_EDGE = 1024

# Step 14 watches for the screen changing under a bone
THUMB_W = 64
CHANGE_FRAC = 0.02  # share of the thumbnail that has to move
CHANGE_LEVEL = 16  # by at least this much, 0-255


# Does this question need eyes? Measured
SURFACE = r"""
    \b(?:screen|display|monitor|desktop|screenshot)\b
  | \b(?:this|that|the)\s+(?:tab|window|page|dialog|popup|menu|panel|toolbar|
       email|message|error|warning|code|file|folder|form|button|chart|graph|
       table|image|photo|picture|video|document|paragraph|line|cell|box)\b
"""

# Watching, rather than knowing: asked about something present.
PERCEIVE = r"""
    \b(?:see|seeing|look|looking|read|reading|show|showing|describe|
       watch|watching|view|viewing|glance|spot)\b
"""

# Something present to point at. "it" is deliberately absent
DEICTIC = r"\b(?:this|that|these|those|here|mine)\b"

# A deictic with nothing after it — "can you explain this", "what's this". There is no noun
BARE_DEICTIC = r"""
    \b(?:explain|describe|read|check|translate|summari[sz]e
      | what(?:'?s|\s+is|\s+are)
      | how\s+about)
    \s+(?:this|that|these|those)
    (?:\s+(?:to|for)\s+\w+)?      # "...to me", "...for us"
    \s*[?.!]*\s*$
"""

# Asked about the viewer's own vantage point. Needs no deictic
FIRST_PERSON = r"""
    \b(?:i\s*(?:'m|\sam)?\s*(?:looking\s+at|seeing|reading|viewing)
      | what\s+(?:i|im|i'm)\s*(?:am\s+)?(?:see|seeing|looking|read|reading|viewing)
      | in\s+front\s+of\s+me
      | on\s+(?:my|the)\s+end)\b
"""

# Things that read as perception but are about knowledge
NOT_SCREEN = r"""
    \b(?:book|article|novel|paper|study|recipe|lyrics?|poem)\b
  | \blook(?:s|ed|ing)?\s+(?:into|up|forward|after|like)\b
  | \bsee\s+(?:you|if|whether|what\s+happens)\b
  | \blet\s*'?s\s+see\b
"""

# Does this question want a finger pointed at something?

# A thing you operate with the mouse.
CONTROL = r"""
    \b(?:click|clicking|press|pressing|tap|button|buttons|menu|menus|toolbar|
       icon|icons|tab|tabs|checkbox|dropdown|option|options|setting|settings|
       preferences?|panel|slider|toggle|field)\b
"""

# The question frame that turns a noun into "where is it".
ASKING = r"\b(?:where|which|what|how|show|point|find)\b"

# Asking to be walked through doing something, rather than told about it.
GUIDE = r"""
    \bhow\s+(?:do|can|could|would|should)\s+i\b
  | \bshow\s+me\s+how\b
  | \bwalk\s+me\s+through\b
  | \bwhat\s+do\s+i\s+(?:do|press|hit|use)\b
"""

# Asking for the finger in so many words.
ERRAND = r"""
    \b(?:point|show|take|guide|lead|direct)\s+me\b
  | \bwhere(?:'?s\b|\s+(?:is|are|was|were|do|does|did|can|could|should|would|
      will|to|abouts)\b)
  | \bhelp\s+me\s+(?:find|get|open|reach|see)\b
  | \bhow\s+(?:do|can|could|would|should)\s+i\s+(?:find|open|access|reach|
      see|view|get\s+to|switch\s+to|turn\s+(?:on|off))\b
  | \bwhich\s+(?:one|button|tab|menu|icon|option|link|setting)\b
  | \b(?:locate|navigate\s+to)\b
  | \b(?:find|open|get\s+to)\s+(?:the|my|a|an)\b
"""

_F = re.VERBOSE | re.IGNORECASE
SURFACE_RE, PERCEIVE_RE = re.compile(SURFACE, _F), re.compile(PERCEIVE, _F)
DEICTIC_RE, FIRST_RE = re.compile(DEICTIC, _F), re.compile(FIRST_PERSON, _F)
NOT_RE = re.compile(NOT_SCREEN, _F)
BARE_RE = re.compile(BARE_DEICTIC, _F)
CONTROL_RE, ASK_RE, GUIDE_RE = (
    re.compile(CONTROL, _F),
    re.compile(ASKING, _F),
    re.compile(GUIDE, _F),
)
ERRAND_RE = re.compile(ERRAND, _F)


def wants_screen(text: str) -> bool:
    if NOT_RE.search(text):
        return False
    if SURFACE_RE.search(text) or FIRST_RE.search(text) or BARE_RE.search(text.strip()):
        return True
    return bool(PERCEIVE_RE.search(text) and DEICTIC_RE.search(text))


def wants_pointing(text: str) -> bool:
    """Would a bone help here, or is this a question that only wants words?"""
    if NOT_RE.search(text):
        return False
    if ERRAND_RE.search(text):
        return True
    if CONTROL_RE.search(text) and ASK_RE.search(text):
        return True
    # "how do i export this" — being walked through something
    return bool(GUIDE_RE.search(text) and (DEICTIC_RE.search(text) or SURFACE_RE.search(text)))


# Does this question want something *done*?
ERRAND_DO = r"""
    \b(?:open|launch|start|run|fire\s+up|bring\s+up)\b
  | \b(?:play|put\s+on|listen\s+to)\b
  | \bgo\s+to\b | \btake\s+me\s+to\b
  | \bshow\s+me\s+(?:my|the)\b
  | \bvolume\b | \b(?:louder|quieter|mute|unmute)\b
  | \bturn\s+(?:\w+\s+){0,3}?(?:up|down)\b
  | \bsearch\s+(?:for|up)\b | \blook\s+up\b | \bgoogle\b
  | \bremind\b | \breminder\b | \bwake\s+me\b | \bnudge\s+me\b
  | \bpomodoro\b | \bfocus\s+(?:timer|session|round)\b | \bset\s+a\s+timer\b
"""
DO_RE = re.compile(ERRAND_DO, _F)


# A question about *where* is never a command to *do*
ASKS_WHERE = r"""
    \bwhere\b | \bwhich\b
  | \b(?:point|show|guide|lead|direct)\s+me\s+(?:to|towards|at|where)\b
  | \bwhereabouts\b
"""
WHERE_RE = re.compile(ASKS_WHERE, _F)


def wants_action(text: str) -> bool:
    """Might they be asking for something to happen, rather than be told?"""
    if WHERE_RE.search(text):
        return False
    return bool(DO_RE.search(text))


def _intersection(a: dict, b: tuple[int, int, int, int]) -> int:
    """Area shared by an mss monitor and a physical desktop rectangle."""
    left = max(a["left"], b[0])
    top = max(a["top"], b[1])
    right = min(a["left"] + a["width"], b[2])
    bottom = min(a["top"] + a["height"], b[3])
    return max(0, right - left) * max(0, bottom - top)


def monitors() -> list[dict]:
    """Every physical monitor in mss/Win32 coordinates."""
    try:
        import mss

        with mss.mss() as sct:
            return [dict(m) for m in sct.monitors[1:]]
    except Exception:
        log.exception("could not enumerate monitors")
        return []


def known_monitor(value) -> dict | None:
    """Accept monitor geometry only when it names a monitor Windows reports."""
    if not isinstance(value, dict):
        return None
    try:
        wanted = {
            "left": int(value["left"]),
            "top": int(value["top"]),
            "width": int(value["width"]),
            "height": int(value["height"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    return next((m for m in monitors() if m == wanted), None)


def active_monitor() -> dict | None:
    """The monitor containing most of the foreground window."""
    try:
        available = monitors()
        if not available:
            return None

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        rect = ctypes.wintypes.RECT()
        if hwnd and user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            box = (rect.left, rect.top, rect.right, rect.bottom)
            best = max(available, key=lambda m: _intersection(m, box))
            if _intersection(best, box):
                return best

        cursor = ctypes.wintypes.POINT()
        if user32.GetCursorPos(ctypes.byref(cursor)):
            for mon in available:
                if (
                    mon["left"] <= cursor.x < mon["left"] + mon["width"]
                    and mon["top"] <= cursor.y < mon["top"] + mon["height"]
                ):
                    return mon
        return available[0]
    except Exception:
        log.exception("could not resolve the active monitor")
        return None


def window_on_monitor(monitor: dict) -> tuple[int, str, str]:
    """Topmost real application window on ``monitor``, excluding Mellow."""
    try:
        user32 = ctypes.windll.user32
        found: list[tuple[int, str, str]] = []
        ignored_classes = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}

        hwnd_t = ctypes.wintypes.HWND
        user32.IsWindowVisible.argtypes = [hwnd_t]
        user32.IsWindowVisible.restype = ctypes.wintypes.BOOL
        user32.IsIconic.argtypes = [hwnd_t]
        user32.IsIconic.restype = ctypes.wintypes.BOOL
        user32.GetWindowTextLengthW.argtypes = [hwnd_t]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [hwnd_t, ctypes.c_wchar_p, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetClassNameW.argtypes = [hwnd_t, ctypes.c_wchar_p, ctypes.c_int]
        user32.GetClassNameW.restype = ctypes.c_int
        user32.GetWindowRect.argtypes = [hwnd_t, ctypes.POINTER(ctypes.wintypes.RECT)]
        user32.GetWindowRect.restype = ctypes.wintypes.BOOL
        user32.GetWindowThreadProcessId.argtypes = [
            hwnd_t,
            ctypes.POINTER(ctypes.wintypes.DWORD),
        ]
        user32.GetWindowThreadProcessId.restype = ctypes.wintypes.DWORD

        callback_t = ctypes.WINFUNCTYPE(
            ctypes.wintypes.BOOL, hwnd_t, ctypes.wintypes.LPARAM
        )

        @callback_t
        def visit(hwnd, _):
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            title_buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title_buf, length + 1)
            title = title_buf.value.strip()
            if not title:
                return True
            class_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buf, len(class_buf))
            if class_buf.value in ignored_classes:
                return True
            rect = ctypes.wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            box = (rect.left, rect.top, rect.right, rect.bottom)
            if _intersection(monitor, box) < max(1, monitor["width"] * monitor["height"] // 200):
                return True
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            app = _process_name(pid.value or 0)
            if app.casefold() == "mellow.exe":
                return True
            found.append((int(hwnd or 0), app, title))
            return False

        user32.EnumWindows.argtypes = [callback_t, ctypes.wintypes.LPARAM]
        user32.EnumWindows.restype = ctypes.wintypes.BOOL
        user32.EnumWindows(visit, 0)
        return found[0] if found else (0, "", "")
    except Exception:
        log.exception("could not resolve the application beneath Mellow")
        return 0, "", ""


def grab(
    max_edge: int = MAX_EDGE, monitor: dict | None = None
) -> tuple[bytes, int, int, "object"] | None:
    """One screenshot: (JPEG bytes, width, height, full-resolution pixels)."""
    try:
        import io

        import mss
        from PIL import Image

        with mss.mss() as sct:
            chosen = monitor or active_monitor() or dict(sct.monitors[1])
            shot = sct.grab(chosen)
            image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        import numpy as np

        pixels = np.asarray(image)
        long_edge = max(image.size)
        if long_edge > max_edge:
            scale = max_edge / long_edge
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.LANCZOS,
            )
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=JPEG_QUALITY)
        data = out.getvalue()
        log.info(
            "captured %dx%d at %d,%d -> %d bytes",
            image.width,
            image.height,
            chosen["left"],
            chosen["top"],
            len(data),
        )
        return data, image.width, image.height, pixels
    except Exception:
        # Deliberately broad: a headless session
        log.exception("screen capture failed")
        return None


def thumbnail():
    """A tiny grayscale frame of the active monitor, for change detection only."""
    try:
        import mss
        import numpy as np
        from PIL import Image

        with mss.mss() as sct:
            shot = sct.grab(active_monitor() or dict(sct.monitors[1]))
            image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        height = max(1, round(image.height * THUMB_W / image.width))
        image = image.resize((THUMB_W, height), Image.BILINEAR).convert("L")
        return np.asarray(image, dtype=np.int16)
    except Exception:
        log.exception("thumbnail failed")
        return None


def changed(before, after) -> bool:
    """Did enough of the screen move to mean the user did something?"""
    if before is None or after is None or before.shape != after.shape:
        return False
    import numpy as np

    moved = np.abs(after - before) > CHANGE_LEVEL
    return bool(moved.mean() > CHANGE_FRAC)


def foreground() -> tuple[str, str]:
    """(app name, window title) of whatever has focus, for the audit event."""
    try:
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "", ""
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value or ""

        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        app = _process_name(pid.value or 0)
        return app, title
    except Exception:
        return "", ""


def _process_name(pid: int) -> str:
    try:
        import ctypes.wintypes
        import os

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        kernel32.QueryFullProcessImageNameW.restype = ctypes.c_bool
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.wintypes.DWORD(len(buf))
            if not kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)
            ):
                return ""
            return os.path.basename(buf.value)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return ""


def media_bytes(image: bytes) -> str | None:
    """Where the shot would be persisted, written only when logging is on."""
    if not config.load()["remember_conversations"]:
        return None  # in-memory only: the turn still works, nothing is kept
    path = sessions_media_path()
    try:
        path.write_bytes(image)
        return str(path)
    except OSError:
        log.exception("could not save the screenshot")
        return None


def sessions_media_path():
    from mellowd import sessions

    return sessions.media_path(".jpg")
