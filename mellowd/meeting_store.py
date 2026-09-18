"""Text-only meeting archive, independent of chat retention."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from mellowd import config


class Store:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or config.CONFIG_DIR / "meetings"

    @contextmanager
    def db(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.directory / "meetings.sqlite3", timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meetings (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT NOT NULL,
                    status TEXT NOT NULL, duration REAL NOT NULL DEFAULT 0,
                    warning TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '',
                    notes_status TEXT NOT NULL DEFAULT '', notes_error TEXT NOT NULL DEFAULT '',
                    engine TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY, meeting TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    start REAL NOT NULL, end REAL NOT NULL, speaker TEXT NOT NULL, text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS segment_meeting ON segments(meeting, start);
                CREATE TABLE IF NOT EXISTS meeting_speakers (
                    meeting TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    key TEXT NOT NULL, name TEXT NOT NULL DEFAULT '', merged_into TEXT,
                    PRIMARY KEY (meeting, key)
                );
            """)
            with conn:
                yield conn
        finally:
            conn.close()

    def recover(self):
        with self.db() as db:
            db.execute("UPDATE meetings SET status='interrupted', warning=? WHERE status IN ('starting','recording','paused','finalizing')",
                       ("Mellow closed before this meeting finished. Saved text is intact; unprocessed audio was not retained. Speaker labels were not finalized.",))
            db.execute("UPDATE meetings SET notes_status='error', notes_error='Notes generation was interrupted. You can try again.' WHERE notes_status='generating'")

    def create(self, title: str) -> str:
        mid = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        title = title.strip()[:160] or datetime.now().strftime("Meeting · %b %d, %H:%M")
        with self.db() as db:
            db.execute("INSERT INTO meetings (id,title,created,status) VALUES (?,?,?,'starting')", (mid, title, now))
        return mid

    def update(self, mid: str, **fields):
        allowed = {"title", "status", "duration", "warning", "notes", "notes_status", "notes_error", "engine"}
        if not fields or not fields.keys() <= allowed:
            raise ValueError("Invalid meeting fields")
        with self.db() as db:
            db.execute(f"UPDATE meetings SET {','.join(key + '=?' for key in fields)} WHERE id=?", (*fields.values(), mid))

    def segment(self, mid: str, start: float, end: float, speaker: str, text: str):
        self.append_segments(mid, [{"start": start, "end": end, "speaker": speaker, "text": text}])

    def append_segments(self, mid: str, segments: list[dict]):
        if not segments:
            return []
        with self.db() as db:
            ids = [db.execute("INSERT INTO segments (meeting,start,end,speaker,text) VALUES (?,?,?,?,?)",
                              (mid, s["start"], s["end"], s["speaker"], s["text"])).lastrowid for s in segments]
            db.execute("UPDATE meetings SET duration=MAX(duration,?) WHERE id=?",
                       (max(s["end"] for s in segments), mid))
            return ids

    def relabel(self, mid, labels: dict[int, str], *, final=False):
        with self.db() as db:
            db.executemany("UPDATE segments SET speaker=? WHERE meeting=? AND id=? AND speaker!='You'",
                           [(key, mid, row) for row, key in labels.items()])
            if final:
                db.execute("DELETE FROM meeting_speakers WHERE meeting=?", (mid,))
            db.execute("INSERT OR IGNORE INTO meeting_speakers (meeting,key) SELECT DISTINCT meeting,speaker FROM segments WHERE meeting=? AND speaker NOT IN ('You','Other participants')", (mid,))

    def edit_speakers(self, mid, *, names=None, source=None, target=None):
        with self.db() as db:
            meeting = db.execute("SELECT status,notes_status,notes FROM meetings WHERE id=?", (mid,)).fetchone()
            if not meeting:
                raise KeyError(mid)
            if meeting["status"] != "complete" or meeting["notes_status"] == "generating":
                raise ValueError("Finish the meeting and notes generation before editing speakers.")
            speakers = {r["key"]: dict(r) for r in db.execute("SELECT key,name,merged_into FROM meeting_speakers WHERE meeting=?", (mid,))}
            if names is not None:
                if not names.keys() <= speakers.keys():
                    raise ValueError("Unknown speaker")
                if any(not isinstance(n, str) or len(n) > 80 or any(ord(c) < 32 for c in n) for n in names.values()):
                    raise ValueError("Use a speaker name of at most 80 characters on one line.")
                db.executemany("UPDATE meeting_speakers SET name=? WHERE meeting=? AND key=?",
                               [(name.strip(), mid, key) for key, name in names.items()])
            if source is not None:
                if source not in speakers or (target is not None and target not in speakers):
                    raise ValueError("Choose two known remote speakers")
                speakers[source]["merged_into"] = target
                canonical(source, speakers)  # Reject self-merge and every indirect cycle.
                db.execute("UPDATE meeting_speakers SET merged_into=? WHERE meeting=? AND key=?", (target, mid, source))
            if meeting["notes"]:
                db.execute("UPDATE meetings SET notes_status='stale',notes_error=? WHERE id=?",
                           ("Speaker labels changed. Regenerate notes to use the updated names and merges.", mid))

    def list(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT id,title,created,status,duration,warning,notes_status FROM meetings ORDER BY created DESC")]

    def get(self, mid: str):
        with self.db() as db:
            row = db.execute("SELECT * FROM meetings WHERE id=?", (mid,)).fetchone()
            if row is None:
                raise KeyError(mid)
            return {**dict(row), "speakers": {s["key"]: dict(s) for s in db.execute(
                "SELECT key,name,merged_into FROM meeting_speakers WHERE meeting=?", (mid,))}, "segments": [dict(s) for s in db.execute(
                "SELECT id,start,end,speaker,text FROM segments WHERE meeting=? ORDER BY start,id", (mid,))]}

    def delete(self, mid: str):
        with self.db() as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("DELETE FROM meetings WHERE id=?", (mid,))
        with self.db() as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def canonical(key, speakers=None):
    speakers = speakers or {}
    seen = set()
    while key in speakers and speakers[key].get("merged_into"):
        if key in seen:
            raise ValueError("Speaker merges cannot form a cycle")
        seen.add(key)
        key = speakers[key]["merged_into"]
        if key not in speakers:
            raise ValueError("Unknown merge target")
    return key


def speaker_label(key, speakers=None, *, qualify=False):
    speakers = speakers or {}
    key = canonical(key, speakers)
    name = speakers.get(key, {}).get("name") or key
    if name == "Other participants":
        return "Other participant"
    # Identity survives equal display names in notes and plain-text exports.
    duplicate = sum(1 for k, v in speakers.items() if not v.get("merged_into") and (v.get("name") or k) == name) > 1
    return f"{name} ({key})" if name != key and (qualify or duplicate) else name


def conversation_turns(segments: list[dict], speakers=None) -> list[dict]:
    turns = []
    for segment in sorted(segments, key=lambda s: (s["start"], s.get("id", 0))):
        segment = {**segment, "speaker": canonical(segment["speaker"], speakers)}
        if turns and turns[-1]["speaker"] == segment["speaker"]:
            turns[-1]["text"] += " " + segment["text"]
            turns[-1]["end"] = max(turns[-1]["end"], segment["end"])
        else:
            turns.append(dict(segment))
    return turns


def export(meeting: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(meeting, ensure_ascii=False, indent=2)
    lines = [f"# {meeting['title']}", meeting["created"], ""]
    if meeting["warning"]:
        lines += [f"Note: {meeting['warning']}", ""]
    if meeting["notes"]:
        lines += ["## Notes", meeting["notes"], ""]
    lines += ["## Transcript", ""]
    for turn in conversation_turns(meeting["segments"], meeting.get("speakers")):
        speaker = speaker_label(turn["speaker"], meeting.get("speakers"))
        lines += [f"{speaker}:", turn["text"], ""]
    return "\n".join(lines)
