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
            """)
            with conn:
                yield conn
        finally:
            conn.close()

    def recover(self):
        with self.db() as db:
            db.execute("UPDATE meetings SET status='interrupted', warning=? WHERE status IN ('starting','recording','paused','finalizing')",
                       ("Mellow closed before this meeting finished. Saved text is intact; unprocessed audio was not retained.",))
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
        with self.db() as db:
            db.execute("INSERT INTO segments (meeting,start,end,speaker,text) VALUES (?,?,?,?,?)", (mid, start, end, speaker, text))
            db.execute("UPDATE meetings SET duration=MAX(duration,?) WHERE id=?", (end, mid))

    def list(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT id,title,created,status,duration,warning,notes_status FROM meetings ORDER BY created DESC")]

    def get(self, mid: str):
        with self.db() as db:
            row = db.execute("SELECT * FROM meetings WHERE id=?", (mid,)).fetchone()
            if row is None:
                raise KeyError(mid)
            return {**dict(row), "segments": [dict(s) for s in db.execute(
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


def export(meeting: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(meeting, ensure_ascii=False, indent=2)
    lines = [f"# {meeting['title']}", meeting["created"], ""]
    if meeting["warning"]:
        lines += [f"Note: {meeting['warning']}", ""]
    if meeting["notes"]:
        lines += ["## Notes", meeting["notes"], ""]
    lines += ["## Transcript", ""]
    lines += [f"[{timestamp(s['start'])}] {s['speaker']}: {s['text']}" for s in meeting["segments"]]
    return "\n".join(lines)
