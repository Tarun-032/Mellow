"""Precision-first visual grounding over one captured monitor."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import re
from dataclasses import dataclass, replace

from PIL import Image, ImageDraw, ImageFont

from mellowd import agents, llm, perf, point

log = logging.getLogger(__name__)

COARSE_COLS = 12
COARSE_ROWS = 8
FINE_COLS = 12
FINE_ROWS = 12
COARSE_EDGE = 1280
FINE_EDGE = 1152
MAX_FINE_ELEMENTS = 48
AGENT_EDGE = 1280
MAX_AGENT_ELEMENTS = 64
# Query matches may not take the whole list: on an inbox the word "spam" in
# subject lines filled it and left no room for the navigation that reveals Spam.
MAX_AGENT_RELEVANT = 24
MAX_AGENT_EXPANDERS = 8
# Labels that reveal more of a list: "More", "More labels", "Show more messages".
_REVEALS_MORE = re.compile(r"^(?:more|showmore|seemore|showall|seeall|viewall|viewmore|expand)")
MAX_AGENT_UIA = 40
MAX_AGENT_OCR = 24
NORMALIZED_EDGE = 1000
# How far past 0 or 1000 a coordinate may be written and still mean the edge.
EDGE_SLACK = 15

_REGION = re.compile(r"\[?(?:REGION|CELL)\s*:\s*(none|[EC]?\s*\d+)\]?", re.IGNORECASE)
_TARGET = re.compile(r"\[?TARGET\s*:\s*(none|[EG]\s*\d+)\]?", re.IGNORECASE)


@dataclass
class GroundedResult:
    target: point.Target | None
    answer: str


class InvalidGrounding(ValueError):
    """The combined API response needs the original strict locator fallback."""


def _json_result(text: str) -> dict | None:
    """A structured agent result, tolerating a surrounding code fence."""
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        start, end = value.find("{"), value.rfind("}")
        if start >= 0:
            # The first complete object, even with prose or a second object
            # after it; slicing first "{" to last "}" rejected both.
            with contextlib.suppress(ValueError):
                parsed, _ = json.JSONDecoder().raw_decode(value, start)
                if isinstance(parsed, dict):
                    return parsed
        if start >= 0 and end > start:
            try:
                parsed = json.loads(value[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except ValueError:
                pass
    return None


def _bare_choice(text: str, stage: str) -> str | None:
    """Normalize strict, bracketed, and bare locator tokens."""
    raw = text.strip()
    pattern = _REGION if stage == "coarse" else _TARGET
    found = pattern.search(raw)
    if found:
        value = found.group(1).replace(" ", "").upper()
        if value.isdigit():
            value = ("C" if stage == "coarse" else "G") + value
        return value
    bare = re.fullmatch(r"(?:REGION|CELL|TARGET)?\s*:?[\s\[]*(none|[ECG]?\s*\d+)\]?", raw, re.IGNORECASE)
    if not bare:
        return None
    value = bare.group(1).replace(" ", "").upper()
    if value.isdigit():
        value = ("C" if stage == "coarse" else "G") + value
    return value


def _grounded_fields(
    raw: str,
    stage: str,
    valid: set[str],
) -> tuple[str | None, str]:
    parsed = _json_result(raw)
    if parsed and not isinstance(parsed.get("answer"), str):
        return None, ""
    if parsed and "selection_kind" in parsed and "selection_index" in parsed:
        kind = str(parsed.get("selection_kind") or "").lower()
        index = parsed.get("selection_index")
        if type(index) is not int or index < 0:
            return None, str(parsed.get("answer", "")).strip()
        if kind == "none":
            if index != 0:
                return None, str(parsed.get("answer", "")).strip()
            selection_raw = "none"
        elif kind == "element":
            selection_raw = f"E{index}"
        elif kind == "cell":
            selection_raw = f"{'C' if stage == 'coarse' else 'G'}{index}"
        else:
            selection_raw = ""
    else:
        selection_raw = str(parsed.get("selection", "")) if parsed else raw
    answer = str(parsed.get("answer", "")).strip() if parsed else raw.strip()
    choice = _bare_choice(selection_raw, stage)
    if choice not in valid:
        return None, answer
    return choice, answer


def _schema(valid: list[str] | None = None) -> dict:
    del valid
    return {
        "type": "object",
        "properties": {
            "selection_kind": {
                "type": "string",
                "enum": ["none", "element", "cell"],
            },
            "selection_index": {"type": "integer", "minimum": 0},
            "answer": {"type": "string"},
        },
        "required": ["selection_kind", "selection_index", "answer"],
        "additionalProperties": False,
    }


AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "selection_kind": {
            "type": "string",
            "enum": ["none", "element", "visual"],
        },
        "selection_index": {"type": "integer", "minimum": 0},
        "visual_left": {"type": "integer", "minimum": 0, "maximum": 1000},
        "visual_top": {"type": "integer", "minimum": 0, "maximum": 1000},
        "visual_right": {"type": "integer", "minimum": 0, "maximum": 1000},
        "visual_bottom": {"type": "integer", "minimum": 0, "maximum": 1000},
        "spoken_answer": {"type": "string"},
    },
    "required": [
        "selection_kind", "selection_index", "visual_left", "visual_top",
        "visual_right", "visual_bottom",
        "spoken_answer",
    ],
    "additionalProperties": False,
}


# One prepared locator worker, one turn per pointer. Codex receives AGENT_SCHEMA
# per turn; Claude is prepared without the CLI schema flag because that flag
# withholds the whole answer until the process exits. The returned JSON passes
# the same local validator either way.
agents.register_profile("locator", agents.VISUAL_LOCATOR_SYSTEM, None)


def _font(size: int):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _jpeg(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("RGB").save(out, "JPEG", quality=90)
    return out.getvalue()


def _fit(image: Image.Image, edge: int, *, enlarge: bool = False) -> tuple[Image.Image, float]:
    longest = max(image.size)
    scale = edge / longest if enlarge or longest > edge else 1.0
    if abs(scale - 1.0) < 0.001:
        return image.copy(), 1.0
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    ), scale


def coarse_image(
    pixels, candidates: list[point.Target] | None = None, mon: dict | None = None
) -> tuple[bytes, list[point.Target]]:
    image, _ = _fit(Image.fromarray(pixels).convert("RGB"), COARSE_EDGE)
    draw = ImageDraw.Draw(image, "RGBA")
    cw, ch = image.width / COARSE_COLS, image.height / COARSE_ROWS
    line = max(1, round(max(image.size) / 900))
    face = _font(max(13, round(min(cw, ch) * 0.20)))
    for col in range(1, COARSE_COLS):
        x = round(col * cw)
        draw.line((x, 0, x, image.height), fill=(25, 210, 255, 210), width=line)
    for row in range(1, COARSE_ROWS):
        y = round(row * ch)
        draw.line((0, y, image.width, y), fill=(25, 210, 255, 210), width=line)
    for row in range(COARSE_ROWS):
        for col in range(COARSE_COLS):
            number = row * COARSE_COLS + col + 1
            x, y = round(col * cw) + 3, round(row * ch) + 3
            text = str(number)
            box = draw.textbbox((x, y), text, font=face, stroke_width=1)
            draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(0, 35, 45, 220))
            draw.text((x, y), text, font=face, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))

    # Put the small set of lexically relevant measured elements directly on the overview.
    available = [c for c in (candidates or []) if c.bounds]
    matched = sorted(
        (c for c in available if c.score > 0),
        key=lambda c: (-c.score, c.chrome, c.source != "uia"),
    )
    # A request such as "my profile" has no lexical match when the screen shows the user's actual name.
    context = sorted(
        (c for c in available if not c.chrome and c not in matched),
        key=lambda c: (
            c.bounds[1],
            c.source != "uia",
            c.bounds[3] * c.bounds[2],
        ),
    )
    measured = (matched + context)[:24]
    tag_face = _font(max(13, round(max(image.size) / 75)))
    native_w = pixels.shape[1]
    scale = image.width / native_w
    if mon:
        for i, cand in enumerate(measured, 1):
            left, top, width, height = cand.bounds
            x0 = round((left - mon["left"]) * scale)
            y0 = round((top - mon["top"]) * scale)
            x1 = round((left + width - mon["left"]) * scale)
            y1 = round((top + height - mon["top"]) * scale)
            draw.rectangle((x0, y0, x1, y1), outline=(255, 70, 185, 245), width=3)
            tag = f"E{i}"
            tb = draw.textbbox((x0 + 2, y0 + 1), tag, font=tag_face, stroke_width=1)
            draw.rectangle((tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1), fill=(80, 0, 48, 235))
            draw.text((x0 + 2, y0 + 1), tag, font=tag_face, fill="white", stroke_width=1, stroke_fill=(0, 0, 0))
    return _jpeg(image), measured


def _crop_for_cell(width: int, height: int, cell: int) -> tuple[int, int, int, int]:
    col = (cell - 1) % COARSE_COLS
    row = (cell - 1) // COARSE_COLS
    cw, ch = width / COARSE_COLS, height / COARSE_ROWS
    # A target can straddle the line the model picked.
    pad_x, pad_y = cw * 0.35, ch * 0.35
    left = max(0, int(col * cw - pad_x))
    top = max(0, int(row * ch - pad_y))
    right = min(width, int((col + 1) * cw + pad_x))
    bottom = min(height, int((row + 1) * ch + pad_y))
    return left, top, right, bottom


def _intersects(bounds, crop, mon) -> bool:
    left = bounds[0] - mon["left"]
    top = bounds[1] - mon["top"]
    right = left + bounds[2]
    bottom = top + bounds[3]
    return right > crop[0] and bottom > crop[1] and left < crop[2] and top < crop[3]


def fine_image(
    pixels, crop: tuple[int, int, int, int], candidates: list[point.Target], mon: dict
) -> tuple[bytes, list[point.Target]]:
    native = Image.fromarray(pixels).convert("RGB").crop(crop)
    image, scale = _fit(native, FINE_EDGE, enlarge=True)
    draw = ImageDraw.Draw(image, "RGBA")
    cw, ch = image.width / FINE_COLS, image.height / FINE_ROWS
    grid_face = _font(max(11, round(min(cw, ch) * 0.18)))
    for col in range(1, FINE_COLS):
        x = round(col * cw)
        draw.line((x, 0, x, image.height), fill=(20, 190, 235, 125), width=1)
    for row in range(1, FINE_ROWS):
        y = round(row * ch)
        draw.line((0, y, image.width, y), fill=(20, 190, 235, 125), width=1)
    for row in range(FINE_ROWS):
        for col in range(FINE_COLS):
            n = row * FINE_COLS + col + 1
            x, y = round(col * cw) + 2, round(row * ch) + 1
            draw.text((x, y), f"G{n}", font=grid_face, fill=(110, 235, 255, 210), stroke_width=1, stroke_fill=(0, 20, 25, 220))

    regional = [c for c in candidates if c.bounds and _intersects(c.bounds, crop, mon)]
    regional.sort(
        key=lambda c: (
            -c.score,
            c.source != "uia",
            (c.bounds[2] * c.bounds[3]) if c.bounds else float("inf"),
        )
    )
    regional = regional[:MAX_FINE_ELEMENTS]
    tag_face = _font(max(14, round(max(image.size) / 70)))
    for i, cand in enumerate(regional, 1):
        left, top, width, height = cand.bounds
        x0 = round((left - mon["left"] - crop[0]) * scale)
        y0 = round((top - mon["top"] - crop[1]) * scale)
        x1 = round((left + width - mon["left"] - crop[0]) * scale)
        y1 = round((top + height - mon["top"] - crop[1]) * scale)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width - 1, x1), min(image.height - 1, y1)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 70, 185, 245), width=3)
        tag = f"E{i}"
        tb = draw.textbbox((x0 + 2, y0 + 1), tag, font=tag_face, stroke_width=1)
        draw.rectangle((tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1), fill=(80, 0, 48, 235))
        draw.text((x0 + 2, y0 + 1), tag, font=tag_face, fill="white", stroke_width=1, stroke_fill=(0, 0, 0))
    return _jpeg(image), regional


def agent_elements(
    candidates: list[point.Target], mon: dict
) -> list[point.Target]:
    """Compact accessible evidence shared by text and visual locators."""
    interactive = set(point.INTERACTIVE.values())
    eligible = [
        candidate
        for candidate in candidates
        if (
            (candidate.source == "uia" and candidate.kind in interactive)
            or candidate.source == "ocr"
        )
        and candidate.bounds
        and candidate.enabled
        and candidate.visible
        and point._shared(candidate.bounds, mon) >= 4
    ]
    # Preserve relevant evidence first, then reserve space for both exact UIA
    # controls and visible OCR in native/canvas applications. The selection is
    # structural and query-scored, never tied to an application or label.
    relevant = sorted(
        (candidate for candidate in eligible if candidate.score > 0),
        key=lambda candidate: (-candidate.score, candidate.chrome, candidate.source != "uia"),
    )
    remaining_uia = sorted(
        (candidate for candidate in eligible if candidate.source == "uia"),
        key=lambda candidate: (candidate.chrome, candidate.bounds[1], candidate.bounds[0]),
    )
    remaining_ocr = sorted(
        (candidate for candidate in eligible if candidate.source == "ocr"),
        key=lambda candidate: (candidate.chrome, candidate.bounds[1], candidate.bounds[0]),
    )
    measured: list[point.Target] = []

    def add(rows, quota):
        added = 0
        for candidate in rows:
            if len(measured) >= MAX_AGENT_ELEMENTS:
                break
            if any(point._same_place(candidate, existing) for existing in measured):
                continue
            measured.append(candidate)
            added += 1
            if added >= quota or len(measured) >= MAX_AGENT_ELEMENTS:
                break

    add(relevant, MAX_AGENT_RELEVANT)
    # A hidden requested item is reached through a control that reveals more,
    # which rarely shares a word with the request, so those get rows of their
    # own. Accessibility state alone is not enough: Gmail marks its toolbar
    # three-dots menu expandable but not the sidebar's "More labels" row, and a
    # slot for only the first sent the bone to the wrong "More". A label that
    # says it reveals more outranks a bare expander: on the real inbox thirteen
    # header and bookmark menus sat above "More labels" and took all eight rows.
    revealing = [row for row in remaining_uia if _REVEALS_MORE.match(point.squash(row.label))]
    add(revealing + [row for row in remaining_uia if row.expands and row not in revealing],
        MAX_AGENT_EXPANDERS)
    if len(measured) < MAX_AGENT_ELEMENTS:
        add(remaining_uia, MAX_AGENT_UIA)
    if len(measured) < MAX_AGENT_ELEMENTS:
        add(remaining_ocr, MAX_AGENT_OCR)

    return measured


def agent_image(
    pixels, candidates: list[point.Target], mon: dict
) -> tuple[bytes, list[point.Target]]:
    """One uncluttered screenshot with measured interactive controls marked."""
    image, scale = _fit(Image.fromarray(pixels).convert("RGB"), AGENT_EDGE)
    measured = agent_elements(candidates, mon)
    draw = ImageDraw.Draw(image, "RGBA")
    tag_face = _font(max(12, round(max(image.size) / 92)))
    line = max(1, round(max(image.size) / 800))
    for index, candidate in enumerate(measured, 1):
        left, top, width, height = candidate.bounds
        x0 = round((left - mon["left"]) * scale)
        y0 = round((top - mon["top"]) * scale)
        x1 = round((left + width - mon["left"]) * scale)
        y1 = round((top + height - mon["top"]) * scale)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width - 1, x1), min(image.height - 1, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle((x0, y0, x1, y1), outline=(255, 70, 185, 230), width=line)
        tag = f"E{index}"
        box = draw.textbbox((0, 0), tag, font=tag_face, stroke_width=1)
        tag_w, tag_h = box[2] - box[0] + 4, box[3] - box[1] + 2
        tx = min(max(0, x0), max(0, image.width - tag_w))
        ty = y0 - tag_h - 1 if y0 >= tag_h + 1 else min(y1 + 1, image.height - tag_h)
        draw.rectangle((tx, ty, tx + tag_w, ty + tag_h), fill=(80, 0, 48, 225))
        draw.text(
            (tx + 2, ty), tag, font=tag_face, fill="white",
            stroke_width=1, stroke_fill=(0, 0, 0),
        )
    return _jpeg(image), measured


def _agent_bounds(candidate: point.Target, mon: dict) -> tuple[int, int, int, int]:
    """One measured hitbox in the normalized space used by model output."""
    left, top, width, height = candidate.bounds
    return (
        max(0, min(1000, round((left - mon["left"]) / mon["width"] * 1000))),
        max(0, min(1000, round((top - mon["top"]) / mon["height"] * 1000))),
        max(0, min(1000, round((left + width - mon["left"]) / mon["width"] * 1000))),
        max(0, min(1000, round((top + height - mon["top"]) / mon["height"] * 1000))),
    )


def _inside(rect: tuple, x: float, y: float, slack: float = 0) -> bool:
    left, top, width, height = rect
    return (
        left - slack <= x <= left + width + slack
        and top - slack <= y <= top + height + slack
    )


def _whole(value) -> int | None:
    """A model's number as an integer: 412, 412.0, 411.6 and "412" all count.

    Claude runs without a native schema (it withholds streaming), so its types
    drift where Codex's never do. A fraction of one 0-1000 unit is sub-pixel.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            value = float(value.strip())
    if isinstance(value, (int, float)) and math.isfinite(value):
        return round(value)
    return None


def _agent_target(raw: str, shot, measured: list[point.Target], candidates: list[point.Target]):
    """Validate one agent result without using its label as a semantic gate.

    Rejections name their reason (invalid_json, invalid_kind, invalid_index,
    invalid_box) so a withheld pointer can be diagnosed from telemetry alone.
    """
    parsed = _json_result(raw)
    if not parsed:
        return None, "invalid_json"
    kind = str(parsed.get("selection_kind") or "").strip().lower()
    index = _whole(parsed.get("selection_index"))
    coordinates = [
        _whole(parsed.get("visual_left")), _whole(parsed.get("visual_top")),
        _whole(parsed.get("visual_right")), _whole(parsed.get("visual_bottom")),
    ]
    # A control on the screen edge (a window's close button) is often written a
    # few units past 1000. That is the edge, not garbage.
    coordinates = [
        min(max(value, 0), NORMALIZED_EDGE)
        if value is not None and -EDGE_SLACK <= value <= NORMALIZED_EDGE + EDGE_SLACK
        else value
        for value in coordinates
    ]
    # How a visual pick was reached, so a wrong bone can be traced to its path.
    via = "visual"
    if kind == "none":
        return None, "none"
    if kind == "element" and (index is None or not 1 <= index <= len(measured)):
        # A row number that does not exist, but the box may still be right:
        # judge the box, which is validated and snapped like any visual pick.
        if any(value is None for value in coordinates) or coordinates == [0, 0, 0, 0]:
            return None, "invalid_index"
        kind, via = "visual", "bad_index"
    if kind == "element":
        chosen = measured[index - 1]
        if not chosen.enabled or not chosen.visible or not chosen.bounds:
            return None, "unsafe"
        chosen_x = chosen.bounds[0] + chosen.bounds[2] / 2
        chosen_y = chosen.bounds[1] + chosen.bounds[3] / 2
        window = getattr(shot, "window", None)
        if window is not None and not _inside(window, chosen_x, chosen_y, 2):
            return None, "outside_window"
        # Element IDs and visual intent must agree. When a model describes the
        # correct region but copies the wrong crowded E label, keep the visual
        # region and run it through the same bounds/snap validation below.
        visual_box = (
            all(value is not None for value in coordinates)
            and 0 <= coordinates[0] < coordinates[2] <= NORMALIZED_EDGE
            and 0 <= coordinates[1] < coordinates[3] <= NORMALIZED_EDGE
        )
        if not visual_box:
            # A bare E number is easy to copy incorrectly on dense screens. The
            # visual box is an independent agreement check, so never drive a
            # pointer from the number alone.
            return None, "element_without_bounds"
        mon = shot.monitor
        visual_x = mon["left"] + (coordinates[0] + coordinates[2]) / 2 / NORMALIZED_EDGE * mon["width"]
        visual_y = mon["top"] + (coordinates[1] + coordinates[3]) / 2 / NORMALIZED_EDGE * mon["height"]
        if _inside(chosen.bounds, visual_x, visual_y, 4):
            return replace(
                chosen,
                source="agent-ocr" if chosen.source == "ocr" else "agent-element",
                score=max(1.0, chosen.score),
                monitor=dict(shot.monitor),
            ), "model_element"
        kind, via = "visual", "element_mismatch"
    if kind != "visual":
        return None, "invalid_kind"
    if any(value is None for value in coordinates):
        return None, "invalid_box"

    left, top, right, bottom = coordinates
    if not (0 <= left < right <= NORMALIZED_EDGE and 0 <= top < bottom <= NORMALIZED_EDGE):
        return None, "invalid_box"
    # A pointer request names a control, not a whole pane or window.
    if right - left > 700 and bottom - top > 500:
        return None, "unsafe"
    mon = shot.monitor
    bounds = (
        mon["left"] + left / NORMALIZED_EDGE * mon["width"],
        mon["top"] + top / NORMALIZED_EDGE * mon["height"],
        (right - left) / NORMALIZED_EDGE * mon["width"],
        (bottom - top) / NORMALIZED_EDGE * mon["height"],
    )
    cx, cy = bounds[0] + bounds[2] / 2, bounds[1] + bounds[3] / 2
    window = getattr(shot, "window", None)
    if window is not None and not _inside(window, cx, cy, 2):
        return None, "outside_window"

    interactive = set(point.INTERACTIVE.values())
    overlaps = [
        candidate
        for candidate in candidates
        if candidate.source == "uia"
        and candidate.kind in interactive
        and candidate.bounds
        and candidate.enabled
        and candidate.visible
        and _inside(candidate.bounds, cx, cy, 3)
    ]
    if overlaps:
        chosen = min(overlaps, key=lambda candidate: candidate.bounds[2] * candidate.bounds[3])
        return replace(
            chosen, source="agent-visual-uia", score=max(1.0, chosen.score),
            monitor=dict(mon),
        ), f"{via}_uia_snap"
    return point.Target(
        nx=(cx - mon["left"]) / mon["width"],
        ny=(cy - mon["top"]) / mon["height"],
        label="target",
        source="agent-visual",
        score=1.0,
        bounds=bounds,
        monitor=dict(mon),
    ), "model_visual" if via == "visual" else f"{via}_visual"


_PRIVATE_SPOKEN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
_PUBLIC_LABEL = re.compile(r"^[\w .&+\-/]{1,48}$", re.UNICODE)


def pointer_reply(target: point.Target) -> str:
    """Safe last-resort guidance without exposing a verbose/private UIA label."""
    label = " ".join((target.label or "").split()).strip()
    kind = " ".join((target.kind or "control").split()).strip()
    if (
        label
        and label.casefold() != "target"
        and _PUBLIC_LABEL.fullmatch(label)
        and not _PRIVATE_SPOKEN.search(label)
    ):
        suffix = "" if kind.casefold() in label.casefold() else f" {kind}"
        return f"The highlighted {label}{suffix} is the control you want."
    return f"The highlighted {kind} is the control you want."


def _agent_spoken(raw: str, target: point.Target | None) -> str:
    """Return a short private spoken instruction, never a raw control label."""
    if target is None:
        return ""
    parsed = _json_result(raw) or {}
    answer = " ".join(str(parsed.get("spoken_answer") or "").split()).strip()
    if (
        len(answer) < 3
        or len(answer.split()) < 4
        or len(answer) > 180
        or _PRIVATE_SPOKEN.search(answer)
        or "```" in answer
        or re.search(r"\[(?:look|point|do|region|target)\b", answer, re.IGNORECASE)
    ):
        return pointer_reply(target)
    return answer


# A selection field whose value is complete: the number or word is followed by
# the next separator, so "visual_bottom":40 is not read while 409 is arriving.
_EARLY_FIELD = re.compile(r'"(selection_kind|selection_index|visual_left|visual_top|visual_right|visual_bottom)"'
                          r'\s*:\s*("[^"]*"|-?\d+(?:\.\d+)?)\s*[,}]')


def early_choice(text: str) -> str | None:
    """The model's choice as JSON once all six selection fields have arrived.

    The spoken sentence is written after them and takes about half the reply, so
    the bone need not wait for it. Returns None until the choice is complete, and
    for "none", which has nothing to show early.
    """
    fields = {name: json.loads(value) for name, value in _EARLY_FIELD.findall(text)}
    if len(fields) < 6 or str(fields["selection_kind"]).strip().lower() not in {"element", "visual"}:
        return None
    return json.dumps(fields)


async def _visual_agent_locate_and_answer(
    query: str, shot, cfg: dict, candidates: list[point.Target], on_choice=None
) -> GroundedResult:
    image, measured = agent_image(shot.pixels, candidates, shot.monitor)
    # Each row carries the box in the same 0-1000 space the answer must use.
    # Listing percent centres while demanding a 0-1000 box put three unit
    # systems in one prompt, and the model answered in image pixels — a correct
    # pick thrown away as element_without_bounds.
    listing = "\n".join(
        f'E{index}|{candidate.source}:{candidate.kind or "text"}|'
        f'{"chrome" if candidate.chrome else "app"}|'
        f'{_agent_bounds(candidate, shot.monitor)}|{candidate.label}'
        # Without this a collapsed "More" read as an ordinary button, so the
        # rule to choose what reveals a hidden item had nothing to act on.
        f'{"|expands" if candidate.expands else ""}'
        for index, candidate in enumerate(measured, 1)
    ) or "(no measured controls)"
    prompt = (
        f'User request: "{query}"\n'
        "Every coordinate you read or write is 0-1000 across the full image "
        "width and 0-1000 down its full height. Never answer in image pixels.\n"
        "The image is the complete current screen. Magenta E boxes are measured "
        "UIA controls or OCR text rectangles; their labels are evidence, not "
        "exact wording the user must say. Prefer element when one E box is the "
        "best visible next control. If the requested item is not visible, choose "
        "the visible control that reveals it where that item belongs - usually a "
        "More, Show more or expand control in the list or group the item would be "
        "in - and say that it reveals the item. A row marked expands opens its own "
        "menu, so choose one only when the item is a command in that menu. Never "
        "name or bound a control that is not visible on this screen. "
        "Otherwise choose visual and tightly bound the visible control using "
        "0-1000 coordinates relative to the full image. Select the control itself, "
        "not surrounding text, a panel, heading, or status. Respect whether the "
        "request refers to app content or window chrome. Write spoken_answer as "
        "one or two natural conversational sentences explaining what the requested "
        "control does and where it is. Use 5-30 words; never answer with one word "
        "or generic filler such as 'click here' or 'right here'. Do not quote "
        "email addresses, account identifiers, or long accessibility labels. Use "
        "none only when no safe visible next step exists, and then use an empty "
        "spoken_answer. For an element, copy that row's four bounds exactly into "
        "the visual fields, so the two selections can be checked against each "
        "other. Set selection_index to zero for visual or none, and set visual "
        "coordinates to zero only for none. Return exactly one "
        "JSON object with these fields: selection_kind, selection_index, "
        "visual_left, visual_top, visual_right, visual_bottom, spoken_answer.\n"
        f"Measured controls:\n{listing}"
    )
    early: list[point.Target] = []

    def on_text(text: str) -> None:
        if early or on_choice is None:
            return
        choice = early_choice(text)
        target = _agent_target(choice, shot, measured, candidates)[0] if choice else None
        if target is not None:
            early.append(target)
            on_choice(target)

    with perf.purpose("locator"), perf.span("grounded_locator"):
        raw = await agents.complete_grounded(
            prompt, cfg, image, [], AGENT_SCHEMA, on_text=on_text
        )
    target, outcome = _agent_target(raw, shot, measured, candidates)
    if target is None and early:
        # The choice validated while streaming but the finished reply did not
        # parse, usually a broken sentence. The bone may already be up; keep it.
        target, outcome = early[0], "early_choice"
    perf.mark("locator." + outcome)
    perf.record_pointer(
        outcome=outcome,
        candidates=len(candidates),
        measured=len(measured),
        source=target.source if target is not None else "",
    )
    log.info("agent locator result: %s from %d measured controls", outcome, len(measured))
    if target is None:
        return GroundedResult(None, "")
    answer = _agent_spoken(raw, target) if outcome != "early_choice" else pointer_reply(target)
    return GroundedResult(target, answer)


async def _call(cfg: dict, prompt: str, image: bytes) -> str:
    if cfg["llm"].get("mode") == "agent":
        return await agents.complete_vision(prompt, cfg, image)
    return await llm.complete_vision(prompt, cfg, image)


async def _strict(cfg: dict, prompt: str, image: bytes, pattern: re.Pattern) -> str | None:
    stage = "coarse" if pattern is _REGION else "fine"
    for attempt in range(2):
        asked = prompt if attempt == 0 else prompt + "\nYour last response was invalid. Return only the required bracketed token."
        with perf.purpose("locator_overview" if stage == "coarse" else "locator_refinement"):
            answer = await _call(cfg, asked, image)
        choice = _bare_choice(answer, stage)
        if choice:
            return choice
        log.info("locator output did not parse: %r", answer[:160])
    return None


def _lexical_guard(chosen: point.Target, regional: list[point.Target]) -> point.Target:
    """Snap a vague nearby selection to a substantially better text match."""
    if not chosen.bounds:
        return chosen
    alternatives = [
        c for c in regional
        if c.bounds
        and c.chrome == chosen.chrome
        and c.score >= 0.45
        and c.score >= chosen.score + 0.25
    ]
    if not alternatives:
        return chosen
    best = max(alternatives, key=lambda c: c.score)
    ax = chosen.bounds[0] + chosen.bounds[2] / 2
    ay = chosen.bounds[1] + chosen.bounds[3] / 2
    bx = best.bounds[0] + best.bounds[2] / 2
    by = best.bounds[1] + best.bounds[3] / 2
    if math.hypot(ax - bx, ay - by) > 180:
        return chosen
    log.info(
        "locator snapped nearby %r (%.2f) to literal %r (%.2f)",
        chosen.label,
        chosen.score,
        best.label,
        best.score,
    )
    return best


@perf.timed("localization")
async def locate(
    query: str, shot, cfg: dict, candidates: list[point.Target]
) -> point.Target | None:
    """Resolve a question to one measured or finely gridded target."""
    mon = shot.monitor
    coarse, overview = coarse_image(shot.pixels, candidates, mon)
    overview_list = "\n".join(
        f"E{i}: {c.label} ({c.kind or c.source}; "
        f"{'browser chrome' if c.chrome else 'inside the app/page'}; "
        f"{round(c.nx * 100)}% across, {round(c.ny * 100)}% down)"
        for i, c in enumerate(overview, 1)
    ) or "(no relevant measured elements)"
    coarse_prompt = (
        f'The user asked: "{query}"\n'
        "Magenta E labels are measured elements. Cyan numbered areas are coarse cells.\n"
        f"Measured elements:\n{overview_list}\n"
        "Return [REGION:E<n>] when a measured element is the exact control. Otherwise return [REGION:C<n>] for the one coarse cell containing it. "
        "Choose the control itself, not a similarly named document, tab title, heading, or status message. "
        "If the requested item is hidden, choose the visible menu, expander, or parent control that reveals it and explain that next step. "
        "A browser URL or browser tab is not an in-page/app command. For 'start a new chat', choose the New button inside the app, not a tab, URL, or existing Chat mode. "
        "For profile/account/avatar requests, choose the username, avatar, or account menu inside the site/app; never choose browser controls such as Ask Gemini or the browser profile. "
        "For an icon, choose its measured element or the cell containing the icon. If no visible control answers the request, return [REGION:none]."
    )
    picked = await _strict(cfg, coarse_prompt, coarse, _REGION)
    log.info("locator overview returned %s from %d measured elements", picked, len(overview))
    if not picked or picked == "NONE":
        return None

    height, width = shot.pixels.shape[:2]
    if picked.startswith("E"):
        index = int(picked[1:])
        if not 1 <= index <= len(overview):
            log.info("locator rejected out-of-range overview target %s", picked)
            return None
        hint = overview[index - 1]
        chosen = _lexical_guard(hint, overview)
        log.info("locator chose overview %s %r", picked, chosen.label)
        # The overview's magenta E rectangles already are native UIA/OCR measurements.
        return replace(chosen, score=max(chosen.score, 1.0), monitor=dict(mon))
    else:
        cell = int(picked[1:] if picked.startswith("C") else picked)
        if not 1 <= cell <= COARSE_COLS * COARSE_ROWS:
            log.info("locator rejected out-of-range coarse cell %s", picked)
            return None
    crop = _crop_for_cell(width, height, cell)
    fine, regional = fine_image(shot.pixels, crop, candidates, mon)
    listing = "\n".join(
        f"E{i}: {c.label} ({c.kind or c.source}; "
        f"{'browser chrome' if c.chrome else 'inside the app/page'})"
        for i, c in enumerate(regional, 1)
    ) or "(no measured element boxes in this crop)"
    fine_prompt = (
        f'The user asked: "{query}"\n'
        "This is an enlarged crop of the region you selected. Magenta E labels are measured UI elements; cyan G labels are fine grid cells.\n"
        f"Measured elements:\n{listing}\n"
        "Return [TARGET:E<n>] when a magenta box is the exact requested control. Prefer this because its hitbox is measured. "
        "Otherwise return [TARGET:G<n>] for the cyan cell containing the visual center of the exact icon/control. "
        "For profile/account/avatar requests, choose the site's username, avatar, or account menu and never a browser control such as Ask Gemini. "
        "Do not choose nearby text, a panel, tab title, breadcrumb, or status message. Return [TARGET:none] if the target is not actually visible."
    )
    target = await _strict(cfg, fine_prompt, fine, _TARGET)
    log.info("locator fine crop returned %s from %d measured elements", target, len(regional))
    if not target or target == "NONE":
        return None
    if target.startswith("E"):
        i = int(target[1:])
        if not 1 <= i <= len(regional):
            return None
        chosen = _lexical_guard(regional[i - 1], regional)
        log.info("locator chose %s %r via cell %d", target, chosen.label, cell)
        return replace(chosen, score=max(chosen.score, 1.0), monitor=dict(mon))

    grid = int(target[1:])
    if not 1 <= grid <= FINE_COLS * FINE_ROWS:
        return None
    col = (grid - 1) % FINE_COLS
    row = (grid - 1) // FINE_COLS
    cell_w = (crop[2] - crop[0]) / FINE_COLS
    cell_h = (crop[3] - crop[1]) / FINE_ROWS
    local_x = crop[0] + (col + 0.5) * cell_w
    local_y = crop[1] + (row + 0.5) * cell_h
    bounds = (
        mon["left"] + crop[0] + col * cell_w,
        mon["top"] + crop[1] + row * cell_h,
        cell_w,
        cell_h,
    )
    log.info("locator chose visual %s via cell %d", target, cell)
    return point.Target(
        nx=local_x / mon["width"],
        ny=local_y / mon["height"],
        label=query[: point.MAX_CHARS],
        source="visual-grid",
        score=0.75,
        bounds=bounds,
        monitor=dict(mon),
    )


async def _agent_pick(
    prompt: str,
    cfg: dict,
    image: bytes,
    messages: list[dict],
    valid: list[str],
    stage: str,
) -> tuple[str | None, str]:
    """Structured selection; API failures fall back to the strict locator."""
    schema = _schema(valid)
    valid_set = {value.upper() for value in valid}
    answer = ""
    agent = cfg["llm"].get("mode") == "agent"
    # Subscription CLIs have high process startup cost. Structured schema output
    # gets one attempt; invalid output is withheld instead of paying for another
    # process and risking a guessed target.
    for attempt in range(1):
        asked = prompt + (
            "\nReturn selection_kind=element with the E number, "
            "selection_kind=cell with the C/G number, or "
            "selection_kind=none with selection_index=0."
        )
        if attempt:
            asked += "\nThe previous selection was invalid. Choose one value from the schema enum."
        with perf.purpose("locator_overview" if stage == "coarse" else "locator_refinement"), perf.span("grounded_locator"):
            complete = agents.complete_grounded if agent else llm.complete_grounded
            raw = await complete(asked, cfg, image, messages, schema)
        if agent:
            choice, answer = _grounded_fields(raw, stage, valid_set)
        else:
            choice, answer = _grounded_fields(raw, stage, valid_set)
            if choice is None:
                raise InvalidGrounding("selection is not an offered element or cell")
            if re.search(r"\[(?:look|point|do|region|target)\b", answer, re.I):
                answer = ""  # Resolve speech separately; never expose control tokens.
        if choice:
            return choice, answer
        log.info("structured agent locator output did not parse: %r", raw[:240])
    return None, "" if agent else answer


@perf.timed("localization_and_answer")
async def locate_and_answer(
    query: str,
    shot,
    cfg: dict,
    candidates: list[point.Target],
    messages: list[dict],
    on_choice=None,
) -> GroundedResult:
    """Grounding and the spoken answer in one model call per image.

    In agent mode `on_choice(target)` is called once, while the model is still
    writing its sentence, with a target that already passed validation.
    """
    if cfg["llm"].get("mode") == "agent":
        return await _visual_agent_locate_and_answer(query, shot, cfg, candidates, on_choice)
    mon = shot.monitor
    coarse, overview = coarse_image(shot.pixels, candidates, mon)
    overview_list = "\n".join(
        f"E{i}: {c.label} ({c.kind or c.source}; "
        f"{'browser chrome' if c.chrome else 'inside the app/page'}; "
        f"{round(c.nx * 100)}% across, {round(c.ny * 100)}% down)"
        for i, c in enumerate(overview, 1)
    ) or "(no relevant measured elements)"
    prompt = (
        f'The user asked: "{query}"\n'
        "Magenta E labels are measured elements. Cyan numbered areas are coarse cells.\n"
        f"Measured elements:\n{overview_list}\n"
        "Choose E<n> when a measured element is the exact control. Otherwise choose C<n> for the one coarse cell containing it. "
        "Choose the control itself, not a similarly named document, tab title, heading, or status message. "
        "If the requested item is hidden, choose the visible menu, expander, or parent control that reveals it and explain that next step. "
        "A browser URL or browser tab is not an in-page/app command. For 'start a new chat', choose the New button inside the app, not a tab, URL, or existing Chat mode. "
        "For profile/account/avatar requests, choose the username, avatar, or account menu inside the site/app; never browser controls such as Ask Gemini. "
        "For an icon, choose its measured element or the cell containing the icon. Choose none only when no visible control answers the request."
    )
    coarse_valid = ["none"] + [f"E{i}" for i in range(1, len(overview) + 1)] + [
        f"C{i}" for i in range(1, COARSE_COLS * COARSE_ROWS + 1)
    ]
    picked, answer = await _agent_pick(
        prompt, cfg, coarse, messages, coarse_valid, "coarse"
    )
    log.info(
        "grounded locator overview returned %s from %d measured elements",
        picked,
        len(overview),
    )
    if not picked or picked == "NONE":
        return GroundedResult(None, answer)
    if picked.startswith("E"):
        index = int(picked[1:])
        if not 1 <= index <= len(overview):
            return GroundedResult(None, answer)
        chosen = _lexical_guard(overview[index - 1], overview)
        if chosen is not overview[index - 1]:
            answer = ""  # The explanation described a different control.
        return GroundedResult(
            replace(chosen, score=max(chosen.score, 1.0), monitor=dict(mon)),
            answer,
        )

    cell = int(picked[1:])
    height, width = shot.pixels.shape[:2]
    if not 1 <= cell <= COARSE_COLS * COARSE_ROWS:
        return GroundedResult(None, answer)
    crop = _crop_for_cell(width, height, cell)
    if cfg["llm"].get("mode") == "agent":
        # The one vision call already narrowed the screen to this region. Use a
        # measured box only when the remaining local evidence is independently
        # safe; otherwise withhold instead of launching a refinement process.
        regional = [
            candidate
            for candidate in candidates
            if candidate.bounds and _intersects(candidate.bounds, crop, mon)
        ]
        chosen = point.confident_match(regional)
        if chosen is None:
            log.info("agent locator region %d had no unique accessible target", cell)
            return GroundedResult(None, "")
        log.info("agent locator region %d resolved locally to %r", cell, chosen.label)
        return GroundedResult(
            replace(chosen, score=max(chosen.score, 1.0), monitor=dict(mon)),
            answer,
        )
    fine, regional = fine_image(shot.pixels, crop, candidates, mon)
    listing = "\n".join(
        f"E{i}: {c.label} ({c.kind or c.source}; "
        f"{'browser chrome' if c.chrome else 'inside the app/page'})"
        for i, c in enumerate(regional, 1)
    ) or "(no measured element boxes in this crop)"
    fine_prompt = (
        f'The user asked: "{query}"\n'
        "This is an enlarged crop of the selected region. Magenta E labels are measured UI elements; cyan G labels are fine grid cells.\n"
        f"Measured elements:\n{listing}\n"
        "Choose E<n> when a magenta box is the exact requested control. Otherwise choose G<n> for the cyan cell containing the visual center of the exact icon/control. "
        "If the requested item is hidden, choose the visible menu, expander, or parent control that reveals it. "
        "For profile/account/avatar requests, choose the site's username, avatar, or account menu and never a browser control such as Ask Gemini. "
        "Do not choose nearby text, a panel, tab title, breadcrumb, or status message. Choose none only if the target is not visible."
    )
    fine_valid = ["none"] + [f"E{i}" for i in range(1, len(regional) + 1)] + [
        f"G{i}" for i in range(1, FINE_COLS * FINE_ROWS + 1)
    ]
    selected, fine_answer = await _agent_pick(
        fine_prompt, cfg, fine, messages, fine_valid, "fine"
    )
    # A coarse-cell explanation cannot stand in for the refined target's answer.
    answer = (fine_answer or answer) if cfg["llm"].get("mode") == "agent" else fine_answer
    log.info(
        "grounded locator fine crop returned %s from %d measured elements",
        selected,
        len(regional),
    )
    if not selected or selected == "NONE":
        return GroundedResult(None, answer)
    if selected.startswith("E"):
        index = int(selected[1:])
        if not 1 <= index <= len(regional):
            return GroundedResult(None, answer)
        chosen = _lexical_guard(regional[index - 1], regional)
        if chosen is not regional[index - 1]:
            answer = ""
        return GroundedResult(
            replace(chosen, score=max(chosen.score, 1.0), monitor=dict(mon)),
            answer,
        )

    grid = int(selected[1:])
    if not 1 <= grid <= FINE_COLS * FINE_ROWS:
        return GroundedResult(None, answer)
    col = (grid - 1) % FINE_COLS
    row = (grid - 1) // FINE_COLS
    cell_w = (crop[2] - crop[0]) / FINE_COLS
    cell_h = (crop[3] - crop[1]) / FINE_ROWS
    local_x = crop[0] + (col + 0.5) * cell_w
    local_y = crop[1] + (row + 0.5) * cell_h
    bounds = (
        mon["left"] + crop[0] + col * cell_w,
        mon["top"] + crop[1] + row * cell_h,
        cell_w,
        cell_h,
    )
    return GroundedResult(
        point.Target(
            nx=local_x / mon["width"],
            ny=local_y / mon["height"],
            label=query[: point.MAX_CHARS],
            source="visual-grid",
            score=0.75,
            bounds=bounds,
            monitor=dict(mon),
        ),
        answer,
    )


@perf.timed("target_verification")
def changed_at(before, after, target: point.Target) -> bool:
    """Whether the localized area moved enough to make the point stale."""
    if before.monitor != after.monitor or not target.bounds:
        return True
    import numpy as np

    mon = before.monitor
    left, top, width, height = target.bounds
    x0 = max(0, int(left - mon["left"] - width))
    y0 = max(0, int(top - mon["top"] - height))
    x1 = min(mon["width"], int(left - mon["left"] + width * 2))
    y1 = min(mon["height"], int(top - mon["top"] + height * 2))
    if x1 <= x0 or y1 <= y0:
        return True
    a = Image.fromarray(before.pixels[y0:y1, x0:x1]).convert("L").resize((64, 64))
    b = Image.fromarray(after.pixels[y0:y1, x0:x1]).convert("L").resize((64, 64))
    delta = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    return bool((delta > 32).mean() > 0.12)


@perf.timed("target_relocation")
def relocate_target(before, after, target: point.Target) -> point.Target | None:
    """Track one changed target locally when its visual patch moved uniquely."""
    if before.monitor != after.monitor or not target.bounds:
        return None
    import numpy as np

    mon = before.monitor
    left, top, width, height = target.bounds
    width, height = round(width), round(height)
    x0, y0 = round(left - mon["left"]), round(top - mon["top"])
    if width < 6 or height < 6:
        return None
    pixels_before, pixels_after = before.pixels, after.pixels
    frame_h, frame_w = pixels_before.shape[:2]
    if (
        x0 < 0 or y0 < 0 or x0 + width > frame_w or y0 + height > frame_h
        or pixels_after.shape[:2] != (frame_h, frame_w)
    ):
        return None

    sample_size = (32, 32)
    needle = np.asarray(
        Image.fromarray(pixels_before[y0:y0 + height, x0:x0 + width])
        .convert("L").resize(sample_size, Image.Resampling.BILINEAR),
        dtype=np.int16,
    )
    radius = min(96, max(24, round(max(width, height) * 0.75)))
    step = max(4, min(12, round(min(width, height) / 3)))
    xs = sorted(set(range(-radius, radius + 1, step)) | {0})
    ys = sorted(set(range(-radius, radius + 1, step)) | {0})
    matches = []
    for dy in ys:
        yy = y0 + dy
        if yy < 0 or yy + height > frame_h:
            continue
        for dx in xs:
            xx = x0 + dx
            if xx < 0 or xx + width > frame_w:
                continue
            sample = np.asarray(
                Image.fromarray(pixels_after[yy:yy + height, xx:xx + width])
                .convert("L").resize(sample_size, Image.Resampling.BILINEAR),
                dtype=np.int16,
            )
            difference = float(np.abs(sample - needle).mean() / 255.0)
            matches.append((difference, dx, dy))
    if not matches:
        return None
    matches.sort()
    best, dx, dy = matches[0]
    separation = max(step * 2, round(min(width, height) * 0.35))
    competitor = next(
        (
            score for score, other_x, other_y in matches[1:]
            if math.hypot(other_x - dx, other_y - dy) >= separation
        ),
        1.0,
    )
    if best > 0.10 or competitor - best < 0.018:
        return None
    moved = (left + dx, top + dy, width, height)
    cx, cy = moved[0] + width / 2, moved[1] + height / 2
    window = getattr(after, "window", None)
    if window is not None and not _inside(window, cx, cy, 2):
        return None
    perf.mark("target_relocated")
    return replace(
        target,
        nx=(cx - mon["left"]) / mon["width"],
        ny=(cy - mon["top"]) / mon["height"],
        bounds=moved,
        monitor=dict(mon),
    )
