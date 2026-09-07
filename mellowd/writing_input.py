"""Focused Windows text insertion. No targeting clicks, Enter, or elevation."""

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import io
import threading
import time

from mellowd import capture

LIMIT = 20000
_INPUT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Target:
    hwnd: int = 0
    pid: int = 0
    runtime: tuple = ()
    app: str = ""
    title: str = ""
    text: str = ""
    caret: int | None = None
    terminal: bool = False
    native: int = 0
    error: str = "Click an editable field before speaking."


@dataclass(frozen=True)
class Insertion:
    status: str
    message: str
    target: Target | None = None
    start: int | None = None
    text: str = ""


def normalized(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def native_offset(raw: str, position: int) -> int:
    index = count = units = 0
    while count < position and index < len(raw):
        width = 2 if raw[index:index + 2] == "\r\n" else 1
        units += len(raw[index:index + width].encode("utf-16-le")) // 2
        index += width
        count += 1
    if count != position:
        raise ValueError("Text position is out of range")
    return units


def clean_text(text: str, terminal: bool = False) -> str:
    text = normalized(text)
    text = "".join(c for c in text if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) <= 159))
    return " ".join(text.split()) if terminal else text


def same_field(a: Target, b: Target) -> bool:
    return bool(a.hwnd and a.runtime and (a.hwnd, a.pid, a.runtime) == (b.hwnd, b.pid, b.runtime))


def unchanged(a: Target, b: Target) -> bool:
    return same_field(a, b) and not b.error and a.text == b.text and a.caret == b.caret


def _win():
    user = C.WinDLL("user32", use_last_error=True)
    user.GetForegroundWindow.restype = W.HWND
    user.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
    user.GetAsyncKeyState.argtypes = [C.c_int]
    user.GetAsyncKeyState.restype = C.c_short
    return user


def _message(hwnd, message, wparam=0, lparam=0):
    user = _win()
    user.SendMessageTimeoutW.argtypes = [W.HWND, W.UINT, C.c_size_t, C.c_ssize_t,
        W.UINT, W.UINT, C.POINTER(C.c_size_t)]
    user.SendMessageTimeoutW.restype = C.c_ssize_t
    result = C.c_size_t()
    if not user.SendMessageTimeoutW(hwnd, message, wparam, lparam, 2, 300, C.byref(result)):
        raise RuntimeError("The field did not respond.")
    return result.value


def _native_edit(control, base):
    hwnd = control.NativeWindowHandle
    user = _win()
    user.GetClassNameW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
    user.GetWindowLongW.argtypes = [W.HWND, C.c_int]
    name = C.create_unicode_buffer(128)
    user.GetClassNameW(hwnd, name, len(name))
    classname = name.value.lower()
    supported = {"edit", "richedit20w", "richedit50w", "richeditd2dpt"}
    if classname not in supported and not any(classname.startswith(f"windowsforms10.{kind}.app.") for kind in supported):
        return Target(**base, error="This field does not expose a safe text position. Copy the draft instead."), None
    if user.GetWindowLongW(hwnd, -16) & 0x800:
        return Target(**base, error="This field is read-only."), None
    size = _message(hwnd, 0x000E)
    if size > LIMIT:
        return Target(**base, error="This field is too large to verify safely."), None
    buffer = C.create_unicode_buffer(size + 1)
    _message(hwnd, 0x000D, size + 1, C.addressof(buffer))
    start, end = W.DWORD(), W.DWORD()
    _message(hwnd, 0x00B0, C.addressof(start), C.addressof(end))
    if start.value != end.value:
        return Target(**base, error="Clear the text selection and place the caret before speaking."), None
    caret = len(normalized(buffer.value.encode("utf-16-le")[:start.value * 2].decode("utf-16-le")))
    return Target(**base, native=int(hwnd), text=normalized(buffer.value), caret=caret, error=""), control


def _read(auto) -> tuple[Target, object]:
    user = _win()
    hwnd = user.GetForegroundWindow()
    pid = W.DWORD()
    user.GetWindowThreadProcessId(hwnd, C.byref(pid))
    app, title = capture.foreground()
    base = dict(hwnd=int(hwnd or 0), pid=pid.value, app=app, title=title)
    control = auto.GetFocusedControl()
    if not control:
        return Target(**base), None
    ancestor = control
    for _ in range(24):
        if ancestor.NativeWindowHandle == hwnd:
            break
        ancestor = ancestor.GetParentControl()
        if ancestor is None:
            return Target(**base), None
    else:
        return Target(**base), None
    base["runtime"] = tuple(control.GetRuntimeId())
    if control.IsPassword or not control.IsEnabled:
        return Target(**base, error="Mellow cannot write in protected or disabled fields."), None
    if app.lower() in {"mellow.exe", "mellowd.exe"}:
        return Target(**base), None
    kind = control.ControlTypeName
    terminal = (app.lower() in {"windowsterminal.exe", "openconsole.exe", "conhost.exe"}
                or "terminal" in str(control.Name).lower())
    if terminal and "claude" not in title.lower():
        return Target(**base, error="This terminal is not an identified Claude Code prompt. Copy the draft instead."), None
    if kind not in {"EditControl", "DocumentControl"} and not terminal:
        return Target(**base), None
    pattern = control.GetPattern(auto.PatternId.TextPattern)
    if not pattern:
        return _native_edit(control, base)
    doc = pattern.DocumentRange
    text = normalized(doc.GetText(LIMIT + 1))
    if len(text) > LIMIT:
        return Target(**base, error="This field is too large to verify safely. Copy the draft instead."), None
    ranges = pattern.GetSelection()
    caret = None
    if len(ranges) == 1:
        selection = ranges[0]
        # uiautomation 2.0.29's CompareEndpoints wrapper misses the COM unwrap.
        if selection.textRange.CompareEndpoints(0, selection.textRange, 1) != 0:
            return Target(**base, error="Clear the text selection and place the caret before speaking."), None
        prefix = doc.Clone()
        prefix.MoveEndpointByRange(1, selection, 0, waitTime=0)
        caret = len(normalized(prefix.GetText(LIMIT + 1)))
    if caret is None and not terminal:
        return Target(**base, error="The caret is unavailable. Copy the draft instead."), None
    # Read-only documents can expose TextPattern too.
    readonly = doc.GetAttributeValue(auto.TextAttributeId.IsReadOnlyAttribute)
    if readonly is not False and not terminal:
        return Target(**base, error="This field is read-only."), None
    return Target(**base, text=text, caret=caret, terminal=terminal, error=""), control


def snapshot() -> Target:
    try:
        import uiautomation as auto
        with auto.UIAutomationInitializerInThread(debug=False):
            return _read(auto)[0]
    except Exception:
        return Target(error="Mellow could not verify this field. Click an editable field or copy the draft.")


def context(target: Target) -> tuple[str, bytes | None]:
    """Visible, foreground-window-only context, not an offscreen document dump."""
    if not target.hwnd or _win().GetForegroundWindow() != target.hwnd:
        return "", None
    import uiautomation as auto
    from PIL import ImageGrab
    with auto.UIAutomationInitializerInThread(debug=False):
        root = auto.ControlFromHandle(target.hwnd)
        rect = root.BoundingRectangle
        lines, queue = [], [root]
        deadline = time.monotonic() + 0.5
        while queue and len(lines) < 160 and time.monotonic() < deadline:
            node = queue.pop(0)
            try:
                box = node.BoundingRectangle
                if node.IsOffscreen or node.IsPassword or box.width() <= 0:
                    continue
                if box.left >= rect.right or box.right <= rect.left or box.top >= rect.bottom or box.bottom <= rect.top:
                    continue
                if node.Name:
                    lines.append(str(node.Name)[:500])
                queue.extend(node.GetChildren())
            except Exception:
                continue
        image = None
        if _win().GetForegroundWindow() == target.hwnd and rect.width() > 0 and rect.height() > 0:
            shot = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True)
            shot.thumbnail((1600, 1600))
            output = io.BytesIO()
            shot.convert("RGB").save(output, format="JPEG", quality=85)
            image = output.getvalue()
        return "\n".join(lines)[:12000], image


class _Mouse(C.Structure):
    _fields_ = [("dx", W.LONG), ("dy", W.LONG), ("data", W.DWORD), ("flags", W.DWORD), ("time", W.DWORD), ("extra", C.c_size_t)]


class _Keyboard(C.Structure):
    _fields_ = [("vk", W.WORD), ("scan", W.WORD), ("flags", W.DWORD), ("time", W.DWORD), ("extra", C.c_size_t)]


class _Union(C.Union):
    _fields_ = [("ki", _Keyboard), ("mi", _Mouse)]


class _Input(C.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", W.DWORD), ("u", _Union)]


def _paste_keys(user) -> bool:
    keys = (_Input * 4)()
    for entry, (vk, flags) in zip(keys, [(0x11, 0), (0x56, 0), (0x56, 2), (0x11, 2)]):
        entry.type, entry.ki.vk, entry.ki.flags = 1, vk, flags
    user.SendInput.argtypes = [W.UINT, C.POINTER(_Input), C.c_int]
    count = user.SendInput(4, keys, C.sizeof(_Input))
    if count != 4:
        # Release only the keys this operation may have pressed.
        user.SendInput(2, C.cast(C.byref(keys, 2 * C.sizeof(_Input)), C.POINTER(_Input)), C.sizeof(_Input))
    return count == 4


class _Clipboard:
    """Keep the original OLE data object, including non-text formats."""

    def __enter__(self):
        self.user = _win()
        self.user.CreateWindowExW.argtypes = [W.DWORD, W.LPCWSTR, W.LPCWSTR, W.DWORD,
            C.c_int, C.c_int, C.c_int, C.c_int, W.HWND, W.HMENU, W.HINSTANCE, C.c_void_p]
        self.user.CreateWindowExW.restype = W.HWND
        self.user.DestroyWindow.argtypes = [W.HWND]
        self.user.OpenClipboard.argtypes = [W.HWND]
        self.window = self.user.CreateWindowExW(0, "STATIC", "Mellow clipboard", 0, 0, 0, 0, 0, None, None, None, None)
        if not self.window:
            raise RuntimeError("Could not prepare a clipboard owner.")
        self.ole = C.OleDLL("ole32")
        self.ole.OleInitialize(None)
        self.original = C.c_void_p()
        self.sequence = None
        self.ole.OleGetClipboard.argtypes = [C.POINTER(C.c_void_p)]
        self.ole.OleSetClipboard.argtypes = [C.c_void_p]
        try:
            self.ole.OleGetClipboard(C.byref(self.original))
        except Exception:
            self.ole.OleUninitialize()
            self.user.DestroyWindow(self.window)
            raise RuntimeError("Could not preserve your clipboard. Copy the draft instead.")
        self.before = self.user.GetClipboardSequenceNumber()
        return self

    def set(self, text):
        user = self.user
        kernel = C.WinDLL("kernel32", use_last_error=True)
        kernel.GlobalAlloc.argtypes = [W.UINT, C.c_size_t]
        kernel.GlobalAlloc.restype = W.HGLOBAL
        kernel.GlobalLock.argtypes = [W.HGLOBAL]
        kernel.GlobalLock.restype = C.c_void_p
        kernel.GlobalUnlock.argtypes = [W.HGLOBAL]
        kernel.GlobalFree.argtypes = [W.HGLOBAL]
        user.SetClipboardData.argtypes = [W.UINT, W.HANDLE]
        user.SetClipboardData.restype = W.HANDLE
        raw = (text + "\0").encode("utf-16-le")
        handle = kernel.GlobalAlloc(0x42, len(raw))
        if not handle:
            raise RuntimeError("Could not prepare the clipboard.")
        owned = True
        try:
            ptr = kernel.GlobalLock(handle)
            if not ptr:
                raise RuntimeError("Could not prepare the clipboard.")
            C.memmove(ptr, raw, len(raw))
            kernel.GlobalUnlock(handle)
            if not user.OpenClipboard(self.window):
                raise RuntimeError("Your clipboard is busy. Retry when it is free.")
            try:
                if user.GetClipboardSequenceNumber() != self.before:
                    raise RuntimeError("Your clipboard changed. Retry insertion.")
                if not user.EmptyClipboard():
                    raise RuntimeError("Could not prepare the clipboard.")
                self.sequence = user.GetClipboardSequenceNumber()
                if not user.SetClipboardData(13, handle):
                    raise RuntimeError("Could not prepare the clipboard.")
                owned = False
                self.sequence = user.GetClipboardSequenceNumber()
            finally:
                user.CloseClipboard()
        finally:
            if owned:
                kernel.GlobalFree(handle)

    def __exit__(self, *_):
        try:
            if self.sequence is not None and self.user.GetClipboardSequenceNumber() == self.sequence:
                self.ole.OleSetClipboard(self.original)
                self.ole.OleFlushClipboard()
        finally:
            if self.original:
                vtable = C.cast(self.original, C.POINTER(C.POINTER(C.c_void_p))).contents
                C.WINFUNCTYPE(W.ULONG, C.c_void_p)(vtable[2])(self.original)
            self.ole.OleUninitialize()
            self.user.DestroyWindow(self.window)


def insert(target: Target, text: str, cancelled: threading.Event, previous: Insertion | None = None) -> Insertion:
    attempted = False
    try:
        import uiautomation as auto
        with _INPUT_LOCK, auto.UIAutomationInitializerInThread(debug=False):
            user = _win()
            deadline = time.monotonic() + 2
            while any(user.GetAsyncKeyState(k) & 0x8000 for k in (0x10, 0x11, 0x12, 0x5B, 0x5C)):
                if cancelled.is_set() or time.monotonic() > deadline:
                    return Insertion("blocked", "Release the shortcut keys, then retry.")
                time.sleep(0.02)
            current, control = _read(auto)
            if cancelled.is_set() or not unchanged(target, current):
                return Insertion("blocked", "The destination changed. Refocus it and retry insertion.")
            text = clean_text(text, current.terminal)
            if not text or len(text) > LIMIT:
                return Insertion("blocked", "This draft cannot be inserted safely. Copy it instead.")
            start = current.caret
            expected = None
            replacement = None
            if previous:
                if current.terminal or not previous.target or not unchanged(previous.target, current) or previous.start is None:
                    return Insertion("blocked", "The last draft is no longer unchanged. Copy the revision instead.")
                start = previous.start
                if current.text[start:start + len(previous.text)] != previous.text:
                    return Insertion("blocked", "The last draft changed. Copy the revision instead.")
                if current.native:
                    # Native Edit offsets count UTF-16 code units and CRLF pairs.
                    size = _message(current.native, 0x000E)
                    buffer = C.create_unicode_buffer(size + 1)
                    _message(current.native, 0x000D, size + 1, C.addressof(buffer))
                    raw = buffer.value
                    replacement = (native_offset(raw, start), native_offset(raw, start + len(previous.text)))
                else:
                    pattern = control.GetPattern(auto.PatternId.TextPattern)
                    replacement = pattern.DocumentRange.FindText(previous.text, False, False)
                    if not replacement:
                        return Insertion("blocked", "Could not locate the last draft safely.")
                    prefix = pattern.DocumentRange.Clone()
                    prefix.MoveEndpointByRange(1, replacement, 0, waitTime=0)
                    if len(normalized(prefix.GetText(LIMIT + 1))) != start:
                        return Insertion("blocked", "The draft's text occurs more than once. Copy the revision instead.")
                expected = current.text[:start] + text + current.text[start + len(previous.text):]
            elif start is not None:
                expected = current.text[:start] + text + current.text[start:]
            with _Clipboard() as clipboard:
                clipboard.set(text)
                if cancelled.is_set() or not unchanged(current, _read(auto)[0]):
                    return Insertion("blocked", "The destination changed. Refocus it and retry insertion.")
                if replacement:
                    if current.native:
                        _message(current.native, 0x00B1, *replacement)
                    else:
                        replacement.Select(waitTime=0)
                if (cancelled.is_set() or not same_field(current, _read(auto)[0])
                        or any(user.GetAsyncKeyState(k) & 0x8000 for k in (0x10, 0x11, 0x12, 0x5B, 0x5C))):
                    return Insertion("blocked", "Insertion cancelled.")
                attempted = True
                sent = _paste_keys(user)
                # Keep the clipboard until the receiving app has consumed the paste.
                deadline = time.monotonic() + 1.5
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                    after, _ = _read(auto)
                    if sent and same_field(current, after) and not after.error:
                        if expected is not None and after.text == expected:
                            return Insertion("inserted", "Draft inserted.", after, start, text)
                    if cancelled.is_set():
                        break
                return Insertion("uncertain", "Paste attempted, but I could not verify it. Check the field before copying anything.")
    except Exception:
        if attempted:
            return Insertion("uncertain", "Paste attempted, but verification failed. Check the field; do not paste twice.")
        return Insertion("blocked", "This app or your clipboard could not be accessed safely. Copy the draft instead.")
