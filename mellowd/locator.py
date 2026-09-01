"""Precision-first visual grounding over one captured monitor."""

from __future__ import annotations

import io
import json
import logging
import math
import re
from dataclasses import dataclass, replace

from PIL import Image, ImageDraw, ImageFont

from mellowd import agents, llm, point

log = logging.getLogger(__name__)

COARSE_COLS = 12
COARSE_ROWS = 8
FINE_COLS = 12
FINE_ROWS = 12
COARSE_EDGE = 1280
FINE_EDGE = 1152
MAX_FINE_ELEMENTS = 48

_REGION = re.compile(r"\[?(?:REGION|CELL)\s*:\s*(none|[EC]?\s*\d+)\]?", re.IGNORECASE)
_TARGET = re.compile(r"\[?TARGET\s*:\s*(none|[EG]\s*\d+)\]?", re.IGNORECASE)


@dataclass
class GroundedResult:
    target: point.Target | None
    answer: str


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


def _recover_label(text: str, candidates: list[point.Target]) -> str | None:
    """Recover an exact measured E target explicitly named in prose."""
    body = point.squash(text)
    mentions = []
    for i, candidate in enumerate(candidates, 1):
        label = point.squash(candidate.label)
        if len(label) >= 3 and label in body:
            mentions.append((len(label), label, i))
    if not mentions:
        return None
    mentions.sort(reverse=True)
    longest = mentions[0][0]
    best = [mention for mention in mentions if mention[0] == longest]
    # UIA and OCR commonly expose the exact same control twice.
    if len({mention[1] for mention in best}) > 1:
        return None
    return f"E{min(mention[2] for mention in best)}"


def _grounded_fields(
    raw: str,
    stage: str,
    valid: set[str],
    candidates: list[point.Target],
) -> tuple[str | None, str]:
    parsed = _json_result(raw)
    selection_raw = str(parsed.get("selection", "")) if parsed else raw
    answer = str(parsed.get("answer", "")).strip() if parsed else raw.strip()
    choice = _bare_choice(selection_raw, stage)
    if choice not in valid:
        choice = _recover_label(answer or raw, candidates)
    if choice not in valid:
        return None, answer
    return choice, answer


def _schema(valid: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "selection": {"type": "string", "enum": valid},
            "answer": {"type": "string"},
        },
        "required": ["selection", "answer"],
        "additionalProperties": False,
    }


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


async def _call(cfg: dict, prompt: str, image: bytes) -> str:
    if cfg["llm"].get("mode") == "agent":
        return await agents.complete_vision(prompt, cfg, image)
    return await llm.complete_vision(prompt, cfg, image)


async def _strict(cfg: dict, prompt: str, image: bytes, pattern: re.Pattern) -> str | None:
    stage = "coarse" if pattern is _REGION else "fine"
    for attempt in range(2):
        asked = prompt if attempt == 0 else prompt + "\nYour last response was invalid. Return only the required bracketed token."
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
    candidates: list[point.Target],
) -> tuple[str | None, str]:
    """Structured agent selection, with one compatibility retry."""
    schema = _schema(valid)
    valid_set = {value.upper() for value in valid}
    answer = ""
    for attempt in range(2):
        asked = prompt + (
            "\nIn the JSON selection field use one allowed value exactly, "
            "without brackets or a REGION/TARGET prefix."
        )
        if attempt:
            asked += "\nThe previous selection was invalid. Choose one value from the schema enum."
        raw = await agents.complete_grounded(asked, cfg, image, messages, schema)
        choice, answer = _grounded_fields(raw, stage, valid_set, candidates)
        if choice:
            return choice, answer
        log.info("structured agent locator output did not parse: %r", raw[:240])
    return None, answer


async def locate_and_answer(
    query: str,
    shot,
    cfg: dict,
    candidates: list[point.Target],
    messages: list[dict],
) -> GroundedResult:
    """Agent-mode grounding and the spoken answer in one CLI invocation."""
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
        "A browser URL or browser tab is not an in-page/app command. For 'start a new chat', choose the New button inside the app, not a tab, URL, or existing Chat mode. "
        "For profile/account/avatar requests, choose the username, avatar, or account menu inside the site/app; never browser controls such as Ask Gemini. "
        "For an icon, choose its measured element or the cell containing the icon. Choose none only when no visible control answers the request."
    )
    coarse_valid = ["none"] + [f"E{i}" for i in range(1, len(overview) + 1)] + [
        f"C{i}" for i in range(1, COARSE_COLS * COARSE_ROWS + 1)
    ]
    picked, answer = await _agent_pick(
        prompt, cfg, coarse, messages, coarse_valid, "coarse", overview
    )
    log.info(
        "agent locator overview returned %s from %d measured elements",
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
        return GroundedResult(
            replace(chosen, score=max(chosen.score, 1.0), monitor=dict(mon)),
            answer,
        )

    cell = int(picked[1:])
    height, width = shot.pixels.shape[:2]
    if not 1 <= cell <= COARSE_COLS * COARSE_ROWS:
        return GroundedResult(None, answer)
    crop = _crop_for_cell(width, height, cell)
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
        "For profile/account/avatar requests, choose the site's username, avatar, or account menu and never a browser control such as Ask Gemini. "
        "Do not choose nearby text, a panel, tab title, breadcrumb, or status message. Choose none only if the target is not visible."
    )
    fine_valid = ["none"] + [f"E{i}" for i in range(1, len(regional) + 1)] + [
        f"G{i}" for i in range(1, FINE_COLS * FINE_ROWS + 1)
    ]
    selected, fine_answer = await _agent_pick(
        fine_prompt, cfg, fine, messages, fine_valid, "fine", regional
    )
    answer = fine_answer or answer
    log.info(
        "agent locator fine crop returned %s from %d measured elements",
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
