"""Reminders that outlive the process."""

from __future__ import annotations

import json
import logging
import os
import uuid
import re
from datetime import datetime, timedelta

from mellowd import config

log = logging.getLogger(__name__)

PATH = config.CONFIG_DIR / "reminders.json"

# How late a reminder may still fire.
GRACE = timedelta(minutes=10)

MAX_TEXT = 200
# A pet is not a task manager.
MAX_ITEMS = 50


def _clean(item: object) -> dict | None:
    """One reminder, normalised — or None if it can't be salvaged."""
    if not isinstance(item, dict):
        return None
    text = str(item.get("text", "")).strip()[:MAX_TEXT]
    if not text:
        return None
    try:
        hour, minute = (int(part) for part in str(item.get("time", "")).split(":"))
    except (TypeError, ValueError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return {
        "id": str(item.get("id") or uuid.uuid4()),
        "time": f"{hour:02d}:{minute:02d}",
        "text": text,
        "daily": bool(item.get("daily")),
        # "" means "never fired". Only dailies read it; a one-off is deleted the moment it fires
        "last_fired": str(item.get("last_fired") or ""),
    }


def normalize(items: object) -> list[dict]:
    """Drop what can't be read, keep the rest. Raises only on the wrong type."""
    if not isinstance(items, list):
        raise ValueError("reminders must be a list")
    cleaned = (_clean(item) for item in items)
    return [item for item in cleaned if item is not None][:MAX_ITEMS]


def due(items: list[dict], now: datetime) -> tuple[list[dict], list[dict]]:
    """Which reminders fire at `now`, and the list to persist afterwards."""
    fired: list[dict] = []
    keep: list[dict] = []
    today = now.strftime("%Y-%m-%d")
    for item in items:
        clean = _clean(item)
        if clean is None:
            continue  # one bad entry must not stop the others firing
        hour, minute = (int(part) for part in clean["time"].split(":"))
        when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if clean["last_fired"] == today or now < when or now - when > GRACE:
            keep.append(clean)
            continue
        fired.append(clean)
        if clean["daily"]:
            keep.append({**clean, "last_fired": today})
        # A one-off is simply not kept: firing is what retires it.
    return fired, keep


def load() -> list[dict]:
    """Never raises. An unreadable file must not stop the pet from starting."""
    try:
        return normalize(json.loads(PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return []
    except (ValueError, OSError) as e:
        log.warning("reminders.json is unreadable (%s) — starting empty", e)
        return []


def save(items: object) -> list[dict]:
    """Write atomically and return what was actually stored."""
    checked = normalize(items)
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temp = PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(checked, indent=2), encoding="utf-8")
    os.replace(temp, PATH)
    return checked

# Saying when, out loud Step 15a.

_SPOKEN = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "twentyfive": 25, "thirty": 30, "forty": 40,
    "fortyfive": 45, "fifty": 50, "sixty": 60, "ninety": 90,
}
_COUNT = (
    r"\d+"
    r"|(?:twenty|thirty|forty|fifty|sixty)(?:one|two|three|four|five|six|seven|eight|nine)"
    + "|" + "|".join(sorted(_SPOKEN, key=len, reverse=True))
)

# "in ten minutes", "in 2 hours", "in about 90 seconds".
_AFTER = re.compile(
    rf"\bin\s+(?:about\s+|around\s+)?(?P<n>{_COUNT})[\s-]*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|h\b|m\b)",
    re.IGNORECASE,
)
_HALF = re.compile(r"\bin\s+(?:half\s+an\s+hour|(?:an?\s+)?half\s+hour)\b", re.IGNORECASE)
_QUARTER = re.compile(r"\bin\s+(?:a\s+)?quarter\s+(?:of\s+)?(?:an\s+)?hour\b", re.IGNORECASE)

# "at 9", "at 9:30", "9pm", "at 21:00"
_CLOCK = re.compile(
    r"\b(?:at\s+)?(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)"
    r"|\bat\s+(?P<h2>\d{1,2}):(?P<m2>\d{2})\b"
    r"|\bat\s+(?P<h3>\d{1,2})\b(?!\s*(?::|\d))",
    re.IGNORECASE,
)

_DAILY = re.compile(
    r"\b(?:every\s*day|everyday|daily|each\s+day|every\s+morning"
    r"|every\s+evening|every\s+night)\b",
    re.IGNORECASE,
)

# Words that join the time to the thing and belong to neither.
_JOINER = re.compile(r"^(?:to|that|about|and|for|it|me|please|,|\.|:|-|\s)+", re.IGNORECASE)


# "twenty five" is two words to a speech recogniser and one number to a person.
_TENS = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty)[\s-]+(one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE,
)


def join_numbers(said: str) -> str:
    """"twenty five" -> "twentyfive", so one number reads as one number."""
    return _TENS.sub(lambda m: m.group(1).lower() + m.group(2).lower(), said or "")


_TENS_ONLY = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}
_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


def _count(word: str) -> int | None:
    word = word.strip().lower().replace(" ", "").replace("-", "")
    if word.isdigit():
        return int(word)
    if word in _SPOKEN:
        return _SPOKEN[word]
    # "thirtyfive" rather than an entry per pair.
    for tens, base in _TENS_ONLY.items():
        if word.startswith(tens) and word[len(tens):] in _UNITS:
            return base + _UNITS[word[len(tens):]]
    return None


def at(said: str, now: datetime) -> tuple[str, str, bool] | None:
    """("HH:MM", what to be reminded of, whether it repeats) or None."""
    said = join_numbers((said or "").strip())
    if not said:
        return None
    daily = bool(_DAILY.search(said))
    rest = _DAILY.sub(" ", said)

    when = None
    for pattern, minutes in ((_HALF, 30), (_QUARTER, 15)):
        found = pattern.search(rest)
        if found:
            when = now + timedelta(minutes=minutes)
            rest = rest[: found.start()] + " " + rest[found.end() :]
            break

    if when is None:
        found = _AFTER.search(rest)
        if found:
            size = _count(found.group("n"))
            if size is None:
                return None
            unit = found.group("unit").lower()
            if unit.startswith(("sec", "s")):
                # Nothing shorter than a minute can be stored: the file keeps HH:MM.
                delta = timedelta(minutes=max(1, round(size / 60)))
            elif unit.startswith(("hour", "hr", "h")):
                delta = timedelta(hours=size)
            else:
                delta = timedelta(minutes=size)
            when = now + delta
            rest = rest[: found.start()] + " " + rest[found.end() :]

    if when is None:
        found = _CLOCK.search(rest)
        if found:
            raw = found.group("h") or found.group("h2") or found.group("h3")
            hour = int(raw)
            minute = int(found.group("m") or found.group("m2") or 0)
            half = (found.group("ap") or "").lower().replace(".", "")
            if half.startswith("p") and hour < 12:
                hour += 12
            elif half.startswith("a") and hour == 12:
                hour = 0
            elif not half and hour < 8:
                # Nobody asks for 3am.
                hour += 12
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return None
            when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            rest = rest[: found.start()] + " " + rest[found.end() :]

    if when is None:
        return None
    text = _JOINER.sub("", rest.strip()).strip(" ,.:-")
    return when.strftime("%H:%M"), text[:MAX_TEXT], daily


def add(said: str, now: datetime | None = None) -> tuple[dict, str] | None:
    """Parse and store one spoken reminder."""
    parsed = at(said, now or datetime.now())
    if parsed is None:
        return None
    clock, text, daily = parsed
    item = {"time": clock, "text": text or "reminder", "daily": daily}
    kept = save(load() + [item])
    if not any(x["time"] == clock and x["text"] == item["text"] for x in kept):
        # The cap, almost certainly. Worth saying rather than claiming success.
        return None
    log.info("reminder set for %s: %r (daily=%s)", clock, item["text"], daily)
    return item, f"{'every day at' if daily else 'at'} {clock}"
