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

GA_ROOT = 2
# Deep enough for a web page: Chrome prunes generic containers, so a Gmail-shaped
# compose box measures ten levels, not the forty-five its DOM has.
WALK_LIMIT = 60

NO_FIELD = "Click an editable field before speaking."
UNKNOWN_FIELD = "Mellow could not identify that field. Copy the draft instead."
NOT_TEXT = "Mellow can only write into a text field"


def settling(error: str) -> bool:
    """Could a moment's wait clear this on its own?

    Chromium publishes a page's accessibility tree lazily, so a cold read lands
    on the render-widget pane or on nothing at all. A read-only field, a
    password box or an unidentified terminal will still be exactly that in two
    seconds, and retrying those only delays the answer.
    """
    return error in (NO_FIELD, UNKNOWN_FIELD) or error.startswith(NOT_TEXT)


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
    opaque: bool = False
    # The field's accessibility label — "Message Body", "Subject", "Search".
    # Gmail puts a subject line in the body without it.
    label: str = ""
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


def squashed(text: str) -> str:
    """Whitespace-insensitive form, for comparing text an app may have re-wrapped."""
    return " ".join(text.split())


def bare(text: str) -> str:
    """Letters and digits only — what survives being drawn on a screen.

    A TUI wraps a long prompt across lines and rules a border down both edges,
    so the draft comes back with box characters and padding threaded through it
    and even a whitespace-insensitive comparison misses. Nothing but the
    characters the user actually dictated is left here, which no amount of
    decoration disturbs; a chance match across seventy of them is not a thing
    that happens.
    """
    return "".join(c for c in text.lower() if c.isalnum())


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


# Apps that publish an advisory in place of their contents until screen-reader
# mode is switched on. VS Code is the common one: its terminal reports "Run the
# command: Toggle Screen Reader Accessibility Mode…" and its editor "The editor
# is not accessible at this time…". The text is real, it just is not the field,
# so nothing read back there can ever confirm a paste.
_NOT_CONTENT = (
    "toggle screen reader accessibility mode",
    "to enable screen reader optimized mode",
    "not accessible at this time",
)


def opaque_text(text: str, caret: int | None) -> bool:
    """Is this the field's contents, or the app explaining that it won't say?"""
    # A caret past the end of the text is proof the two describe different
    # things — VS Code's editor reports a 99-character notice and caret 1593.
    if caret is not None and caret > len(text):
        return True
    lowered = text.lower()
    return any(mark in lowered for mark in _NOT_CONTENT)


def landed(before: str, after: str, text: str) -> bool:
    """Did one more copy of the draft appear than was there before?

    What "the paste worked" actually means wherever the app rewrites the field:
    a TUI repaints its status lines between reads and a web editor re-wraps, so
    reconstructing the whole document exactly is impossible. Whitespace is
    collapsed on both sides so a wrapped line still matches.
    """
    needle = bare(text)
    return bool(needle) and bare(after).count(needle) > bare(before).count(needle)


def stable(before: Target, after: Target) -> bool:
    """Is this still the destination we measured, closely enough to paste into?

    A TUI repaints its own status lines between two reads, so demanding
    identical text there would abort a perfectly good paste. Terminals never use
    the caret offset, so the identity of the field is the whole requirement.
    """
    if before.terminal or before.opaque:
        return same_field(before, after) and not after.error
    return unchanged(before, after)


def _win():
    user = C.WinDLL("user32", use_last_error=True)
    user.GetForegroundWindow.restype = W.HWND
    user.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
    user.GetAsyncKeyState.argtypes = [C.c_int]
    user.GetAsyncKeyState.restype = C.c_short
    user.GetAncestor.argtypes = [W.HWND, W.UINT]
    user.GetAncestor.restype = W.HWND
    return user


def _owned_by(user, control, hwnd):
    """Does the focused control belong to the foreground window?

    True/False when provable, None when the tree ran deeper than we walk. Climb
    to the first ancestor owning *any* window handle and ask Windows for its
    root, rather than hunting the top-level handle level by level: Chrome hosts
    a page inside a Chrome_RenderWidgetHostHWND child window, so the handle the
    climb meets first is never the one being looked for.
    """
    node = control
    for _ in range(WALK_LIMIT):
        handle = node.NativeWindowHandle
        if handle:
            return bool(user.GetAncestor(handle, GA_ROOT) == hwnd)
        node = node.GetParentControl()
        if node is None:
            return False
    return None


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
    return Target(**base, native=int(hwnd), text=normalized(buffer.value), caret=caret,
                  label=str(control.Name or "")[:120], error=""), control


def _read(auto) -> tuple[Target, object]:
    user = _win()
    hwnd = user.GetForegroundWindow()
    pid = W.DWORD()
    user.GetWindowThreadProcessId(hwnd, C.byref(pid))
    app, title = capture.foreground()
    base = dict(hwnd=int(hwnd or 0), pid=pid.value, app=app, title=title)
    control = auto.GetFocusedControl()
    # Every rejection below names itself. Five of them used to share one message,
    # which made a refusal impossible to diagnose from what the user was shown.
    if not control:
        return Target(**base, error=NO_FIELD), None
    owned = _owned_by(user, control, hwnd)
    if owned is None:
        return Target(**base, error=UNKNOWN_FIELD), None
    if not owned:
        return Target(**base, error="That field belongs to another window. Click where you want the text."), None
    base["runtime"] = tuple(control.GetRuntimeId())
    if control.IsPassword or not control.IsEnabled:
        return Target(**base, error="Mellow cannot write in protected or disabled fields."), None
    if app.lower() in {"mellow.exe", "mellowd.exe"}:
        return Target(**base, error="Click the field in the other app, not Mellow's own window."), None
    kind = control.ControlTypeName
    # Not an allowlist any more: a terminal is treated like any other field. What
    # keeps that safe is that a terminal draft is collapsed to one line by
    # clean_text(terminal=True), so it cannot carry a newline, and _paste_keys
    # sends Ctrl+V and nothing else — never Enter. The text arrives at the prompt
    # and waits for the user. The flag still selects the caret-optional read and
    # the repaint-tolerant stable()/landed() checks a TUI needs.
    terminal = (app.lower() in {"windowsterminal.exe", "openconsole.exe", "conhost.exe"}
                or "terminal" in str(control.Name).lower())
    if kind not in {"EditControl", "DocumentControl"} and not terminal:
        return Target(**base, error=f"{NOT_TEXT}, and this is a {kind}."), None
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
    # Read-only documents can expose TextPattern too. Ask at the caret rather
    # than across the whole document: a range spanning mixed content answers
    # with a COM sentinel, not a bool, and "not False" reads that as read-only.
    # Only a literal True is a refusal — verifying the paste afterwards is the
    # real guard, so a field that lied simply fails to verify and says so.
    probe = ranges[0] if len(ranges) == 1 else doc
    if probe.GetAttributeValue(auto.TextAttributeId.IsReadOnlyAttribute) is True and not terminal:
        return Target(**base, error="This field is read-only."), None
    return Target(**base, text=text, caret=caret, terminal=terminal,
                  opaque=opaque_text(text, caret), label=str(control.Name or "")[:120],
                  error=""), control


def snapshot() -> Target:
    try:
        import uiautomation as auto
        with auto.UIAutomationInitializerInThread(debug=False):
            return _read(auto)[0]
    except Exception:
        return Target(error="Mellow could not verify this field. Click an editable field or copy the draft.")


def resolve(seconds: float) -> Target:
    """Keep reading until an editable field appears, or the budget runs out.

    Chrome builds a page's accessibility tree lazily, so the first read after a
    hotkey press can land on the render-widget pane while the real text box is
    still being published — a single instant reading refuses a field that is
    about to be perfectly writable. The user is still speaking, so waiting is
    free; the caller runs this off the message loop.
    """
    target = snapshot()
    deadline = time.monotonic() + seconds
    while settling(target.error) and time.monotonic() < deadline:
        time.sleep(0.1)
        target = snapshot()
    return target


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
            if cancelled.is_set() or not stable(target, current):
                return Insertion("blocked", "The destination changed. Refocus it and retry insertion.")
            text = clean_text(text, current.terminal)
            if not text or len(text) > LIMIT:
                return Insertion("blocked", "This draft cannot be inserted safely. Copy it instead.")
            start = current.caret
            expected = None
            replacement = None
            if current.opaque:
                # The readable text is the app's "I won't show you this" notice,
                # not the field, so there is no offset to paste at and nothing to
                # compare against afterwards. The keystrokes still reach the real
                # focus; only our ability to read it back is gone.
                start = None
            if previous:
                if current.terminal or current.opaque or not previous.target or not unchanged(previous.target, current) or previous.start is None:
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
                if cancelled.is_set() or not stable(current, _read(auto)[0]):
                    return Insertion("blocked", "The destination changed. Refocus it and retry insertion.")
                if replacement:
                    if current.native:
                        _message(current.native, 0x00B1, *replacement)
                    else:
                        replacement.Select(waitTime=0)
                        # Chrome can accept Select() on a text range and leave
                        # the DOM selection where it was, in which case the
                        # paste would append the revision instead of replacing
                        # the draft. It also applies the move asynchronously, so
                        # wait for proof rather than sampling once — and send no
                        # keys at all until the selection really covers the old
                        # draft.
                        selected = False
                        until = time.monotonic() + 0.5
                        while not selected and time.monotonic() < until:
                            picked = control.GetPattern(auto.PatternId.TextPattern).GetSelection()
                            selected = (len(picked) == 1
                                        and normalized(picked[0].GetText(LIMIT + 1)) == previous.text)
                            if not selected:
                                time.sleep(0.05)
                        if not selected:
                            return Insertion("blocked", "This app will not let Mellow replace the last draft. Copy the revision instead.")
                if (cancelled.is_set() or not same_field(current, _read(auto)[0])
                        or any(user.GetAsyncKeyState(k) & 0x8000 for k in (0x10, 0x11, 0x12, 0x5B, 0x5C))):
                    return Insertion("blocked", "Insertion cancelled.")
                attempted = True
                sent = _paste_keys(user)
                # Keep the clipboard until the receiving app has consumed the
                # paste. Each poll re-runs the whole gate chain, and Chrome is
                # slow, so this is a handful of samples rather than fifty.
                # Nothing will ever appear in an opaque field's text, so waiting
                # the full window there only delays the spoken answer.
                deadline = time.monotonic() + (0.4 if current.opaque else 2.5)
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                    after, _ = _read(auto)
                    if sent and same_field(current, after) and not after.error:
                        if expected is not None and after.text == expected:
                            return Insertion("inserted", "Draft inserted.", after, start, text)
                        # For a revision the old draft must also be gone: an app
                        # that appended rather than replaced satisfies "the new
                        # text is here" while leaving both on screen.
                        gone = previous is None or landed(after.text, current.text, previous.text)
                        if landed(current.text, after.text, text) and gone:
                            at = after.text.find(text)
                            # An exact-range revision needs an unambiguous span.
                            return Insertion("inserted", "Draft inserted.", after,
                                             at if at >= 0 and after.text.count(text) == 1 else None, text)
                    if cancelled.is_set():
                        break
                if (current.opaque or current.terminal) and sent and not cancelled.is_set():
                    # Not "verification failed" — there was nothing dependable to
                    # verify against. An opaque field publishes a notice instead
                    # of its contents, and a terminal exposes a painted screen
                    # that scrolls, wraps and redraws under us. Reading nothing
                    # back there is the normal case, not a warning sign, and no
                    # Enter is ever sent so an unconfirmed paste sits harmlessly
                    # at the prompt. The log keeps this apart from "inserted".
                    return Insertion("sent", "Draft sent.", None, None, text)
                return Insertion("uncertain", "Paste attempted, but I could not verify it. Check the field before copying anything.")
    except Exception:
        if attempted:
            return Insertion("uncertain", "Paste attempted, but verification failed. Check the field; do not paste twice.")
        return Insertion("blocked", "This app or your clipboard could not be accessed safely. Copy the draft instead.")
