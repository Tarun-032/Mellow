"""The session event log — Mellow's memory of what happened, on disk."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mellowd import config

log = logging.getLogger(__name__)

SESSIONS_DIR = config.CONFIG_DIR / "sessions"
INDEX_PATH = SESSIONS_DIR / "index.jsonl"
# Screenshots (step 13) and anything else with bytes rather than text.
MEDIA_DIR = SESSIONS_DIR / "media"

# Bumped when an event's shape changes in a way an old reader couldn't read.
SCHEMA_VERSION = 1

# A stretch of silence longer than this starts a new session.
SEGMENT_AFTER = timedelta(minutes=30)

# Retention, from the roadmap. Text outlives a year of use by single-digit MB
TEXT_KEEP = timedelta(days=365)
IMAGE_KEEP = timedelta(days=7)

# The History list needs something to show.
TITLE_MAX = 48

# How many events may pass before the index is refreshed anyway.
INDEX_EVERY = 20

# Written today: session_start, user_said, assistant_said
FUTURE_TYPES = (
    "screen_captured",
    "tool_call",
    "tool_result",
    "agent_started",
    "agent_finished",
)

# The message-shaped events, in the order a transcript reads them.
SAID = {"user_said": "user", "assistant_said": "assistant"}


def _now() -> str:
    """ISO-8601 UTC with milliseconds. The one timestamp format on the wire."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parse(ts: str) -> datetime:
    """The inverse of _now(). Never raises: a bad stamp reads as the epoch."""
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _title(text: str) -> str:
    """First words of the first thing the user said, on a word boundary."""
    text = " ".join(text.split())
    if len(text) <= TITLE_MAX:
        return text
    cut = text[:TITLE_MAX].rsplit(" ", 1)[0]
    return f"{cut}…"


@dataclass
class _Live:
    """One session currently being written to."""

    id: str
    started: str
    kind: str = "conversation"
    parent: str = ""
    title: str = ""
    events: int = 0
    turns: int = 0
    last_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # `events` as of the last index write
    indexed: int = 0


# ponytail: one lock for the whole log, held across the file append.
_lock = threading.RLock()
_open: dict[str, _Live] = {}
# The session record() writes to when no id is given — the conversation.
_current = ""


def _index_lines() -> list[dict]:
    """The index as it stands."""
    try:
        lines = INDEX_PATH.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    out = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("id"):
            out.append(item)
    return out


def _write_index(entries: list[dict]) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    temp = INDEX_PATH.with_suffix(".tmp")
    temp.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )
    os.replace(temp, INDEX_PATH)


def _entry(live: _Live) -> dict:
    return {
        "id": live.id,
        "kind": live.kind,
        "parent": live.parent,
        "started_at": live.started,
        "ended_at": live.last_ts.isoformat(timespec="milliseconds"),
        "title": live.title,
        # Both, on purpose. `turns` is what a person wants to see in a list
        "turns": live.turns,
        "events": live.events,
    }


def _upsert_index(live: _Live) -> None:
    """Put one open session's current state into the index, atomically."""
    entries = [e for e in _index_lines() if e.get("id") != live.id]
    entries.append(_entry(live))
    _write_index(entries)
    live.indexed = live.events


def _path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.jsonl"


def _append(live: _Live, event: dict, sync: bool = False) -> None:
    """One line onto one session file. The only place that writes events."""
    path = _path(live.id)
    # A power cut can leave a final line with no newline on it. Appending straight onto that would fuse
    try:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
    except OSError:
        torn = False  # missing or empty: nothing to fuse onto
    with path.open("a", encoding="utf-8") as f:
        if torn:
            f.write("\n")
        # default=str so a step-15 tool argument carrying a Path or a datetime degrades to its text instead
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        f.flush()
        if sync:
            os.fsync(f.fileno())


def open_session(kind: str = "conversation", parent: str = "") -> str:
    """Start a session and return its id."""
    global _current
    with _lock:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        live = _Live(id=uuid.uuid4().hex[:12], started=_now(), kind=kind, parent=parent)
        _open[live.id] = live
        if kind == "conversation":
            _current = live.id
        _append(
            live,
            {
                "v": SCHEMA_VERSION,
                "seq": 0,
                "ts": live.started,
                "type": "session_start",
                "kind": kind,
                "parent": parent,
            },
            sync=True,
        )
        _upsert_index(live)
        return live.id


def close(session_id: str = "", reason: str = "user") -> None:
    """Close a session so the next event starts a fresh one."""
    global _current
    with _lock:
        live = _open.pop(session_id or _current, None)
        if session_id in ("", _current):
            _current = ""
        if live is None:
            return
        try:
            live.events += 1
            _append(
                live,
                {
                    "v": SCHEMA_VERSION,
                    "seq": live.events,
                    "ts": _now(),
                    "type": "session_ended",
                    "reason": reason,
                },
                # The one place worth a real sync: nothing more is coming
                sync=True,
            )
            _upsert_index(live)
        except OSError:
            log.exception("could not close session %s", live.id)


def _for_write(session_id: str) -> _Live | None:
    """The session an event belongs to, opening or rolling one over as needed."""
    if session_id:
        return _open.get(session_id)
    live = _open.get(_current)
    if live is None:
        return _open[open_session()]
    if datetime.now(timezone.utc) - live.last_ts >= SEGMENT_AFTER:
        close(live.id, reason="silence")
        return _open[open_session()]
    return live


def record(kind: str, session: str = "", **data) -> None:
    """Append one event to a session."""
    try:
        if not config.load()["remember_conversations"]:
            return
        with _lock:
            live = _for_write(session)
            if live is None:
                log.warning("dropped a %s event for unknown session %r", kind, session)
                return
            live.events += 1
            live.last_ts = datetime.now(timezone.utc)
            if kind == "user_said":
                live.turns += 1
                if not live.title:
                    live.title = _title(str(data.get("text", "")))
                    live.indexed = -1  # a new title is worth writing out now
            _append(
                live,
                {
                    "v": SCHEMA_VERSION,
                    "seq": live.events,
                    "ts": _now(),
                    "type": kind,
                    **data,
                },
            )
            # Everything the History panel shows comes from a said-event
            if kind in SAID or live.indexed < 0 or live.events - live.indexed >= INDEX_EVERY:
                _upsert_index(live)
    except Exception:
        # Deliberately broad. This is the audit trail's own failure mode
        log.exception("could not write to the session log")


def _safe_id(session_id: str) -> bool:
    """A session id is a filename component."""
    return bool(session_id) and session_id.isalnum() and len(session_id) <= 32


def read(session_id: str) -> list[dict] | None:
    """One session's events, oldest first."""
    if not _safe_id(session_id):
        return None
    try:
        lines = _path(session_id).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _rebuild(session_id: str) -> dict | None:
    """One index entry, reconstructed from the session file itself."""
    events = read(session_id)
    if not events:
        return None
    head = events[0]
    title = next(
        (_title(str(e.get("text", ""))) for e in events if e.get("type") == "user_said"),
        "",
    )
    return {
        "id": session_id,
        "kind": str(head.get("kind", "conversation")),
        "parent": str(head.get("parent", "")),
        "started_at": str(head.get("ts", "")),
        "ended_at": str(events[-1].get("ts", "")),
        "title": title,
        "turns": sum(1 for e in events if e.get("type") == "user_said"),
        # The header is seq 0 and is not an event anyone recorded.
        "events": len(events) - 1,
    }


def list_sessions() -> list[dict]:
    """Newest first."""
    with _lock:
        entries = {
            str(e["id"]): e for e in _index_lines() if _safe_id(str(e.get("id", "")))
        }
        try:
            on_disk = {p.stem for p in SESSIONS_DIR.glob("*.jsonl") if p.stem != "index"}
        except OSError:
            on_disk = set()
        recovered = False
        # Missing entirely, or written by a version that didn't have these fields yet
        stale = (on_disk - set(entries)) | {
            i for i, e in entries.items() if "turns" not in e or "kind" not in e
        }
        for session_id in stale & on_disk:
            entry = _rebuild(session_id)
            if entry:
                entries[session_id] = entry
                recovered = True
        # An entry whose file has gone (swept
        for session_id in set(entries) - on_disk:
            del entries[session_id]
            recovered = True
        if recovered:
            log.info("session index rebuilt from disk")
            _write_index(list(entries.values()))
        return sorted(
            entries.values(), key=lambda e: str(e.get("started_at", "")), reverse=True
        )


def resume() -> tuple[list[dict], tuple[str, str] | None]:
    """The current session's recent turns as LLM messages, plus its destination."""
    try:
        entries = list_sessions()
        if not entries:
            return [], None
        newest = entries[0]
        if newest.get("kind", "conversation") != "conversation":
            return [], None
        if datetime.now(timezone.utc) - _parse(str(newest.get("ended_at", ""))) >= SEGMENT_AFTER:
            return [], None  # that conversation is over; the next event opens a new one

        events = read(str(newest["id"])) or []
        if not events:
            return [], None
        # The whole point of writing the ending down
        if events[-1].get("type") == "session_ended":
            return [], None
        messages: list[dict] = []
        destination: tuple[str, str, str] | None = None
        for event in events:
            role = SAID.get(str(event.get("type")))
            text = str(event.get("text", ""))
            if not role or not text:
                continue  # a failed or barged-in-before-a-word turn has nothing to carry
            messages.append({"role": role, "content": text})
            if role == "assistant":
                # Exactly the triple main.answer compares.
                destination = (
                    str(event.get("provider", "")),
                    str(event.get("base_url", "")),
                    str(event.get("model", "")),
                )
        # A question whose answer never landed would make answer() append a second user message in a row
        if messages and messages[-1]["role"] == "user":
            messages.pop()
        _adopt(newest, events)
        return messages, destination
    except Exception:
        log.exception("could not resume the open session")
        return [], None


def _adopt(entry: dict, events: list[dict]) -> None:
    """Re-open a session file for appending, so `resume` writes back into it."""
    global _current
    with _lock:
        session_id = str(entry["id"])
        if session_id in _open:
            _current = session_id
            return
        _open[session_id] = _Live(
            id=session_id,
            started=str(entry.get("started_at", "")),
            kind=str(entry.get("kind", "conversation")),
            parent=str(entry.get("parent", "")),
            title=str(entry.get("title", "")),
            events=max((int(e.get("seq", 0) or 0) for e in events), default=0),
            turns=sum(1 for e in events if e.get("type") == "user_said"),
            last_ts=_parse(str(events[-1].get("ts", ""))),
            indexed=-1,  # write the recovered counts out on the next event
        )
        _current = session_id


def media_path(ext: str = ".png") -> Path:
    """A fresh path under sessions/media/ for a screenshot or other blob."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    return MEDIA_DIR / f"{uuid.uuid4().hex}{ext if ext.startswith('.') else '.' + ext}"


def clear() -> int:
    """Delete every session and everything they refer to."""
    global _current
    with _lock:
        _open.clear()
        _current = ""
        count = sum(1 for p in SESSIONS_DIR.glob("*.jsonl") if p.stem != "index")
        try:
            shutil.rmtree(SESSIONS_DIR, ignore_errors=True)
        except OSError:
            log.exception("could not clear the session log")
            return 0
        return count


def sweep() -> None:
    """Retention pass, run once at sidecar startup."""
    with _lock:
        now = datetime.now(timezone.utc)
        kept = []
        dropped = 0
        for entry in list_sessions():
            stamp = str(entry.get("ended_at") or entry.get("started_at", ""))
            if now - _parse(stamp) > TEXT_KEEP:
                try:
                    _path(str(entry["id"])).unlink(missing_ok=True)
                    dropped += 1
                except OSError:
                    log.exception("could not sweep %s", entry.get("id"))
            else:
                kept.append(entry)
        if dropped:
            _write_index(kept)
            log.info("session log: swept %d session(s) past the one-year mark", dropped)

        images = 0
        cutoff = (now - IMAGE_KEEP).timestamp()
        for f in MEDIA_DIR.glob("*") if MEDIA_DIR.exists() else []:
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    images += 1
            except OSError:
                log.exception("could not sweep %s", f)
        if images:
            log.info("session log: swept %d screenshot(s) past seven days", images)
