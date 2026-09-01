"""What is on the screen right now, so the model can point at one of it."""

import logging
import re
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Bounds on the UIA walk.
MAX_NODES = 3000
MAX_DEPTH = 30
UIA_BUDGET = 2.5

# Windows' OCR reads a 1080p screen in about 0.2s
OCR_BUDGET = 8.0

# How much bigger the frame is made before OCR reads it. Measured, see _read.
SCALE = 2

# How many rows the model is shown. Two real screens came to 55 and 89 after deduplication
MAX_ITEMS = 90

# Longest a row may be and still be something you click.
MAX_WORDS = 5
MAX_CHARS = 80

# How much of the captured screen a document has to cover before it is believed to be the page.
MIN_PAGE = 0.2

# How many of the rows may be the browser's own furniture.
MAX_CHROME = 12

# Control types worth pointing at, and what to call them in front of the model.
INTERACTIVE = {
    "ButtonControl": "button",
    "SplitButtonControl": "button",
    "HyperlinkControl": "link",
    "MenuItemControl": "menu item",
    "TabItemControl": "tab",
    "ListItemControl": "list item",
    "CheckBoxControl": "checkbox",
    "RadioButtonControl": "radio button",
    "ComboBoxControl": "dropdown",
    "EditControl": "text box",
    "TreeItemControl": "tree item",
}

# A match has to clear this to become a bone.
THRESHOLD = 0.65

# A control whose box is most of the screen is the window
MAX_SPAN = (0.6, 0.4)

# Query words that do not identify the target.
STOP = frozenset("""
    a an the this that these those my your it its is are was were be am
    i me we you they he she him her us them
    do does did done doing can could would should shall will
    what which where when who how why whose
    to at in on of for from with by into onto towards toward near beside
    and or but if then than so as
    please show tell find locate point take get go open click clicking press
    pressing tap hit push select choose use using need want let s
    button buttons icon icons option options thing something anything
    screen page window here there now again
    hey hi hello hiya yo mellow please kindly pls thanks thank
    menu menus toolbar sidebar dropdown panel dialog box field link links
    item items section area list bar
""".split())


@dataclass
class Target:
    """Where the bone goes, already in the transport's own units."""

    nx: float
    ny: float
    label: str
    source: str
    score: float
    # What kind of control it is, in plain words
    kind: str = ""
    # True when this is the browser's own furniture rather than the page
    chrome: bool = False
    # Physical desktop rectangle.
    bounds: tuple[float, float, float, float] | None = None
    # Physical monitor containing this target.
    monitor: dict | None = None


# OCR whitespace is unreliable, so matching uses alphanumerics only.

_ALNUM = re.compile(r"[^a-z0-9]+")


def squash(text: str) -> str:
    """Lowercase alphanumerics only. "API Keys" and "APIKeys" both become one."""
    return _ALNUM.sub("", text.lower())


def terms(query: str) -> list[str]:
    """The words in a question that name the thing, in the order they appear."""
    words = _ALNUM.sub(" ", query.lower()).split()
    kept = [w for w in words if w not in STOP and len(w) > 1]
    # Keep a fallback for stopword-only questions.
    return kept or [w for w in words if len(w) > 2]


def phrases(words: list[str]) -> list[tuple[int, str]]:
    """(how many words, the run squashed together), longest run first."""
    out = []
    for size in range(len(words), 0, -1):
        for start in range(len(words) - size + 1):
            out.append((size, "".join(words[start : start + size])))
    return out


def score(name: str, wanted: list[str]) -> float:
    """How much this row looks like an answer to that question, 0 to 1."""
    cand = squash(name)
    if not cand or not wanted:
        return 0.0

    for size, phrase in phrases(wanted):
        at = cand.find(phrase)
        if at < 0:
            continue
        # Coverage says how much of the label the match accounts for, and how much of the question it used.
        covers = len(phrase) / len(cand)
        uses = len(phrase) / max(1, len("".join(wanted)))
        # Straight penalty for only using some of what they asked for, and it is load-bearing twice
        whole = 0.35 + 0.65 * size / len(wanted)
        return (0.55 + 0.45 * max(covers, uses)) * whole

    # No run of words survives as a phrase
    parts = [p for p in re.split(r"(?<=[a-z])(?=[A-Z])|[^A-Za-z0-9]+", name) if p]
    if not parts:
        return 0.0
    hit = sum(
        1
        for p in parts
        if any(
            squash(p).startswith(w) or w.startswith(squash(p))
            for w in wanted
            if len(w) >= 3 and len(squash(p)) >= 3
        )
    )
    return hit / len(parts) * 0.8


# --- tier 1: the accessibility tree -----------------------------------------


def uia_candidates(hwnd: int = 0) -> tuple[list[tuple], tuple | None]:
    """(every named visible control, where the page inside it starts)."""
    try:
        import uiautomation as auto
    except Exception:
        log.warning("uiautomation is not installed; the exact tier is off")
        return [], None
    out = []
    page = None
    try:
        # uiautomation owns CoInitialize
        with auto.UIAutomationInitializerInThread(debug=False):
            root = auto.ControlFromHandle(hwnd) if hwnd else auto.GetForegroundControl()
            if root is None:
                return [], None
            deadline = time.monotonic() + UIA_BUDGET
            queue, seen = [(root, 0)], 0
            while queue and seen < MAX_NODES and time.monotonic() < deadline:
                node, depth = queue.pop(0)
                seen += 1
                try:
                    name, kind, box = node.Name, node.ControlTypeName, node.BoundingRectangle
                except Exception:
                    continue
                role = INTERACTIVE.get(kind, "")
                # Icon-only controls are exactly the things OCR cannot save.
                if box.width() > 0 and (name or role):
                    automation = ""
                    help_text = ""
                    if not name:
                        try:
                            automation = str(node.AutomationId or "")
                        except Exception:
                            pass
                        try:
                            help_text = str(node.HelpText or "")
                        except Exception:
                            pass
                    label = str(name or help_text or automation or role)
                    out.append(
                        (label, box.left, box.top, box.width(), box.height(), role, "uia")
                    )
                # The biggest document in the tree is the page being read.
                if kind == "DocumentControl" and box.width() > 0:
                    size = box.width() * box.height()
                    if page is None or size > page[4]:
                        page = (box.left, box.top, box.width(), box.height(), size)
                if depth < MAX_DEPTH:
                    try:
                        queue.extend((child, depth + 1) for child in node.GetChildren())
                    except Exception:
                        pass
            log.info("uia walked %d nodes, %d named", seen, len(out))
    except Exception:
        log.exception("uia walk failed")
    return out, page[:4] if page else None


# --- tier 2: what the screen actually says ----------------------------------

async def _read(pixels) -> list[tuple]:
    """Windows' own OCR engine, over one frame."""
    import io

    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import BitmapDecoder
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    from PIL import Image

    buffer = io.BytesIO()
    frame = Image.fromarray(pixels)
    # Twice the size, and this is not a nicety.
    frame = frame.resize((frame.width * SCALE, frame.height * SCALE), Image.LANCZOS)
    # BMP rather than PNG: no compression to do on the way in or out
    frame.save(buffer, format="BMP")
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream.get_output_stream_at(0))
    writer.write_bytes(buffer.getvalue())
    await writer.store_async()
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()

    engine = OcrEngine.try_create_from_language(
        Language("en-US")
    ) or OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        log.warning("no OCR language pack installed; the text tier is off")
        return []
    read = await engine.recognize_async(bitmap)

    out = []
    for line in read.lines:
        # Materialised, because these are WinRT vectors
        words = [w for w in line.words if w.text.strip()]
        if not words:
            continue
        boxes = [w.bounding_rect for w in words]
        tall = max(b.height for b in boxes)
        gaps = [b.x - (a.x + a.width) for a, b in zip(boxes, boxes[1:])]
        # The line or its words
        phrase = len(words) > 1 and all(g <= tall * 0.6 for g in gaps)
        if phrase:
            first, last = boxes[0], boxes[-1]
            out.append(
                (
                    line.text,
                    first.x / SCALE,
                    first.y / SCALE,
                    (last.x + last.width - first.x) / SCALE,
                    tall / SCALE,
                    "",
                    "ocr",
                )
            )
            continue
        for word, box in zip(words, boxes):
            out.append(
                (word.text, box.x / SCALE, box.y / SCALE,
                 box.width / SCALE, box.height / SCALE, "", "ocr")
            )
    return out


def ocr_candidates(pixels) -> list[tuple]:
    """Every word on screen, with its box."""
    if pixels is None:
        return []
    import asyncio
    import threading

    # winsdk is async all the way down
    done: list = []

    def work():
        try:
            done.append(asyncio.run(_read(pixels)))
        except Exception:
            log.exception("ocr failed")

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(OCR_BUDGET)
    if not done:
        log.warning("ocr gave nothing back within %.1fs", OCR_BUDGET)
        return []
    out = done[0]
    log.info("ocr read %d words and lines", len(out))
    return out


# --- the ladder -------------------------------------------------------------


def monitor() -> dict | None:
    """The primary monitor in physical pixels, and the reason we are DPI-aware."""
    try:
        import mss

        with mss.mss() as sct:
            return dict(sct.monitors[1])
    except Exception:
        log.exception("could not read the monitor")
        return None


# what is on screen


# Words that are never a thing you click.
NOISE = frozenset("""
    a an the this that and or of to in on at is are was were be am it its as by
    for from with if so no up we my me you i do can will not but out off
""".split())


def _trim(name: str) -> str:
    """Drop an orphan glyph OCR glued onto the front of a real label."""
    parts = name.split()
    while len(parts) > 1 and len(parts[0]) == 1 and not parts[0].isdigit():
        parts = parts[1:]
    while len(parts) > 1 and len(parts[-1]) == 1 and not parts[-1].isalnum():
        parts = parts[:-1]
    return " ".join(parts) if parts else name


_HOTKEY = re.compile(
    r"\s*\((?=[^)]*(?:ctrl|shift|alt|win|cmd|command|\+|f\d+))[^)]*\)",
    re.IGNORECASE,
)
_BADGE = re.compile(
    r"\s*[-\u2013\u2014,]\s*\d+\s+(?:pending\s+)?(?:changes?|problems?|"
    r"notifications?|items?|results?).*$",
    re.IGNORECASE,
)


def _display(name: str, source: str) -> str:
    """A compact label for the model without throwing the control away."""
    name = " ".join(name.replace("\n", " ").split())
    if source == "uia":
        name = _HOTKEY.sub("", name)
        name = _BADGE.sub("", name)
    return _trim(name).strip()[:MAX_CHARS]


def _fit(name: str, kind: str = "", source: str = "ocr") -> bool:
    """Could this be a thing you click, or is it prose, or is it noise?"""
    words = squash(name)
    return (
        len(words) >= 2
        and (bool(kind) or len(name.split()) <= MAX_WORDS)
        and words not in NOISE
        and not name.strip().isdigit()
    )


def _shared(rect: tuple, mon: dict) -> float:
    """How much of `rect` is actually on the monitor being captured, in pixels."""
    left = max(rect[0], mon["left"])
    top = max(rect[1], mon["top"])
    right = min(rect[0] + rect[2], mon["left"] + mon["width"])
    bottom = min(rect[1] + rect[3], mon["top"] + mon["height"])
    return max(0, right - left) * max(0, bottom - top)


def _covers(outer: tuple, inner: tuple, slack: float = 2.0) -> bool:
    """Is `inner`'s box inside `outer`'s, give or take a pixel or two?"""
    return (
        inner[1] >= outer[1] - slack
        and inner[2] >= outer[2] - slack
        and inner[1] + inner[3] <= outer[1] + outer[3] + slack
        and inner[2] + inner[4] <= outer[2] + outer[4] + slack
    )


def candidates(
    query: str,
    pixels=None,
    mon: dict | None = None,
    limit: int | None = MAX_ITEMS,
    hwnd: int = 0,
) -> list[Target]:
    """Everything on screen worth offering, best guesses first."""
    mon = mon or monitor()
    if mon is None:
        return []
    tree, page = uia_candidates(hwnd)
    # Ignore a document that belongs to another monitor.
    if page is not None and _shared(page, mon) < MIN_PAGE * mon["width"] * mon["height"]:
        log.info("point: the document is not on this screen; no split")
        page = None
    raw = []
    for name, left, top, width, height, kind, source in tree + ocr_candidates(pixels):
        if not name or not name.strip():
            continue
        # OCR boxes are local to the captured bitmap
        if source == "ocr":
            left += mon["left"]
            top += mon["top"]
        raw.append((_display(name, source), left, top, width, height, kind, source))

    def furniture(row) -> bool:
        """Is this the browser around the page rather than the page itself?"""
        if page is None:
            return False
        left, top, width, height = row[1], row[2], row[3], row[4]
        cx, cy = left + width / 2, top + height / 2
        return not (
            page[0] <= cx <= page[0] + page[2]
            and page[1] <= cy <= page[1] + page[3]
        )

    keep = []
    for row in raw:
        name, left, top, width, height, _, _ = row
        if width <= 0 or height <= 0 or not _fit(name, row[5], row[6]):
            continue
        if width > mon["width"] * MAX_SPAN[0] and height > mon["height"] * MAX_SPAN[1]:
            # The window, or a pane repeating a child's name. Never the thing to click
            continue
        cx, cy = left + width / 2, top + height / 2
        if not (
            mon["left"] <= cx < mon["left"] + mon["width"]
            and mon["top"] <= cy < mon["top"] + mon["height"]
        ):
            # A second monitor
            continue
        keep.append(row)

    # Same words in the same place from both tiers
    merged: list[tuple] = []
    for row in sorted(keep, key=lambda r: (r[6] != "uia", not r[5], r[3] * r[4])):
        text = squash(row[0])
        if any(
            squash(other[0]) == text and (_covers(other, row) or _covers(row, other))
            for other in merged
        ):
            continue
        merged.append(row)

    wanted = terms(query)
    # Relevance first, and being a real control only breaks ties.
    def rank(rows):
        return sorted(rows, key=lambda r: (-score(r[0], wanted), not r[5], r[3] * r[4]))

    inside = rank([r for r in merged if not furniture(r)])
    around = rank([r for r in merged if furniture(r)])
    if not inside:
        # Belt and braces for the same failure: whatever the rectangles said
        inside, around = around, []
    around = around[:MAX_CHROME]
    if limit is None:
        ranked = inside + around
    else:
        ranked = inside[: max(0, limit - len(around))] + around[:limit]
    log.info(
        "point: %d on screen, offering %d in the page and %d around it",
        len(merged), len(ranked) - len(around), len(around),
    )
    return [
        Target(
            nx=(left + width / 2 - mon["left"]) / mon["width"],
            ny=(top + height / 2 - mon["top"]) / mon["height"],
            label=name[:MAX_CHARS],
            source=source,
            score=score(name, wanted),
            kind=kind,
            chrome=furniture(row),
            bounds=(left, top, width, height),
            monitor=dict(mon),
        )
        for row in ranked
        for name, left, top, width, height, kind, source in [row]
    ]


def describe(cands: list[Target]) -> str:
    """The numbered list, as the model sees it."""
    def line(i, c):
        return '%d "%s"%s %d,%d' % (
            i, c.label, " " + c.kind if c.kind else "",
            round(c.nx * 100), round(c.ny * 100),
        )

    inside = [line(i, c) for i, c in enumerate(cands, 1) if not c.chrome]
    around = [line(i, c) for i, c in enumerate(cands, 1) if c.chrome]
    out = (
        "\n\nON SCREEN right now, one per line: number, what it says, what"
        " kind of control it is when that is known, then how far across and"
        " down the screen it sits as percentages."
        "\n\nIN THE PAGE OR APP THEY ARE USING - the answer is almost always"
        " one of these:\n" + "\n".join(inside)
    )
    if around:
        out += (
            "\n\nTHE BROWSER'S OWN WINDOW around that page - its tabs, its"
            " address bar, its bookmarks and extensions. These are not part of"
            " what they are looking at, and picking one of them because it"
            " happens to contain the right word is the single most common way"
            " to get this wrong:\n" + "\n".join(around)
        )
    return out
