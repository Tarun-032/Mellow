"""Meeting lifecycle and local HTTP API."""

import asyncio
import logging
import re
import time
from collections import deque
from contextlib import suppress

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from mellowd import config, errors, meeting_audio, meeting_notes, stt
from mellowd.meeting_store import Store, export

log = logging.getLogger("mellowd.meetings")
ACTIVE = {"starting", "recording", "paused", "finalizing"}
MAX_PENDING = 24


def deduplicate(previous: str, current: str) -> str:
    before, after = previous.split(), current.split()
    normalize = lambda words: [re.sub(r"\W+", "", word).casefold() for word in words]
    for count in range(min(12, len(before), len(after)), 1, -1):
        if normalize(before[-count:]) == normalize(after[:count]):
            return " ".join(after[count:])
    return current


class Manager:
    def __init__(self, store=None, capture_factory=meeting_audio.Capture, transcribe=stt.transcribe_meeting_segments):
        self.store = store or Store()
        self.capture_factory = capture_factory
        self.transcribe = transcribe
        self.id = None
        self.state = "idle"
        self.warning = ""
        self.capture = None
        self.pending = deque()
        self.worker = None
        self.stop_task = None
        self.notes_tasks = {}
        self.notes_progress = {}
        self.lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.retry = asyncio.Event()
        self.retry.set()
        self.origin = 0.0
        self.duration = 0.0
        self.cfg = None
        self.selection = (None, None)
        self.previous = {}
        self.previous_end = {}
        self.before_start = None
        self.after_stop = None
        self.busy = False

    @property
    def active(self):
        return self.state in ACTIVE

    def elapsed(self):
        """Recorded seconds. Frozen while paused — nothing is being captured then."""
        if self.state == "paused" or not self.active:
            return self.duration
        return time.monotonic() - self.origin

    def status(self):
        return {"id": self.id, "status": self.state, "active": self.active,
                "duration": round(self.elapsed(), 1),
                "warning": self.warning, "pending": len(self.pending) + int(self.busy),
                "levels": dict(self.capture.levels) if self.capture else {"You": 0, "Other participants": 0}}

    def persist(self):
        if self.id:
            self.store.update(self.id, status=self.state, warning=self.warning,
                              duration=self.status()["duration"])

    def enqueue(self, chunk):
        if not self.active:
            return
        # Reserve two tail chunks while the audio threads wind down.
        if len(self.pending) >= MAX_PENDING + 2:
            self.warning = "Capture backlog exceeded its limit. Some audio could not be transcribed; the saved transcript is incomplete."
            self.schedule_pause(self.warning)
            return
        self.pending.append(chunk)
        self.changed.set()
        if len(self.pending) >= MAX_PENDING:
            self.schedule_pause("Transcription is falling behind. Recording paused while queued audio is processed. Resume when the queue is clear.")

    def schedule_pause(self, message):
        if self.state == "finalizing":
            self.warning = "Some final audio could not be processed. This transcript may be incomplete. " + message
            self.persist()
            return
        if self.state not in {"starting", "recording"}:
            return
        self.warning = message
        asyncio.create_task(self.pause())

    async def open_capture(self):
        loop = asyncio.get_running_loop()
        capture = self.capture_factory(
            lambda chunk: loop.call_soon_threadsafe(self.enqueue, chunk),
            lambda message: loop.call_soon_threadsafe(self.schedule_pause, message), self.origin)
        self.capture = capture
        try:
            await asyncio.to_thread(capture.start, *self.selection, self.cfg["stt"].get("input_device"))
        except Exception:
            self.capture = None
            raise

    async def start(self, title, microphone=None, output=None):
        async with self.lock:
            if self.active:
                raise RuntimeError("A meeting is already active. Stop it before starting another.")
            self.cfg = config.load()
            self.selection = (microphone, output)
            self.id = self.store.create(title)
            self.state, self.warning = "starting", ""
            self.origin = time.monotonic()
            self.pending.clear()
            self.previous.clear()
            self.previous_end.clear()
            self.retry.set()
            try:
                if self.before_start:
                    await self.before_start()
                if self.cfg["stt"]["mode"] == "local":
                    await asyncio.to_thread(stt.load)
                await self.open_capture()
                self.state = "recording"
                self.persist()
                self.worker = asyncio.create_task(self.consume())
            except Exception as exc:
                if self.capture:
                    await asyncio.to_thread(self.capture.close)
                    self.capture = None
                self.state = "interrupted"
                self.duration = time.monotonic() - self.origin
                self.warning = "Could not start meeting audio. Check your microphone/output and STT settings. " + errors.message(exc)
                self.persist()
                if self.after_stop:
                    await self.after_stop()
                raise RuntimeError(self.warning) from exc
            return self.status()

    async def pause(self):
        async with self.lock:
            if self.state not in {"starting", "recording"}:
                return self.status()
            self.duration = time.monotonic() - self.origin
            self.state = "paused"
            if not self.warning:
                self.warning = "This meeting includes pauses. Audio during pauses was not recorded."
            if self.capture:
                await asyncio.to_thread(self.capture.close)
                self.capture = None
            self.persist()
            return self.status()

    async def resume(self):
        async with self.lock:
            if self.state != "paused":
                raise RuntimeError("Pause the current meeting before resuming it.")
            self.retry.set()
            self.changed.set()
            if len(self.pending) > MAX_PENDING // 2:
                raise RuntimeError("Still processing queued audio. Wait for the queue to shrink, then resume.")
            try:
                self.origin = time.monotonic() - self.duration
                await self.open_capture()
            except Exception as exc:
                self.warning = "Could not reopen audio. Check your selected devices. " + errors.message(exc)
                self.persist()
                raise RuntimeError(self.warning) from exc
            self.state = "recording"
            # Pauses remain disclosed in the saved transcript.
            self.warning = "This meeting includes pauses. Audio during pauses was not recorded."
            self.persist()
            return self.status()

    async def stop(self):
        async with self.lock:
            if not self.active or self.state == "finalizing":
                return self.status()
            self.duration = self.elapsed()
            self.state = "finalizing"
            if self.capture:
                await asyncio.to_thread(self.capture.close)
                self.capture = None
            self.retry.set()
            self.changed.set()
            self.persist()
            self.stop_task = asyncio.create_task(self.finish())
            return self.status()

    async def finish(self):
        try:
            if self.worker:
                await self.worker
            self.state = "complete"
            self.persist()
        except Exception:
            log.exception("meeting finalization failed")
            self.state = "interrupted"
            self.warning = "Meeting finalization failed. Previously saved transcript is available in Meetings."
            self.persist()
        finally:
            if self.after_stop:
                await self.after_stop()

    async def consume(self):
        while self.active:
            await self.retry.wait()
            if not self.pending:
                if self.state == "finalizing":
                    return
                self.changed.clear()
                try:
                    await asyncio.wait_for(self.changed.wait(), 1)
                except asyncio.TimeoutError:
                    if self.state == "recording" and self.capture and not self.capture.healthy():
                        self.schedule_pause("An audio device disconnected. Check your devices and resume.")
                    self.persist()
                continue
            chunk = self.pending.popleft()
            self.busy = True
            try:
                # A single consumer serializes model inference; eight seconds is below Parakeet's cap.
                parts = await asyncio.to_thread(self.transcribe, chunk.audio, self.cfg)
                if isinstance(parts, str):
                    parts = [{"start": 0, "end": chunk.end - chunk.start, "text": parts}]
                previous = self.previous.get(chunk.speaker, "")
                previous_end = self.previous_end.get(chunk.speaker, -1)
                rows = []
                for part in parts:
                    start = chunk.start + part["start"]
                    end = min(chunk.end, chunk.start + part["end"])
                    text = part["text"].strip()
                    if chunk.overlap and start < previous_end:
                        text = deduplicate(previous, text)
                    if text:
                        rows.append({"start": start, "end": end, "speaker": chunk.speaker, "text": text})
                        previous, previous_end = text, end
                if rows:
                    self.store.append_segments(self.id, rows)
                    self.previous[chunk.speaker] = previous
                    self.previous_end[chunk.speaker] = previous_end
            except Exception as exc:
                if self.state == "finalizing":
                    self.warning = "Some queued audio could not be transcribed. This transcript is incomplete. " + errors.message(exc)
                else:
                    self.pending.appendleft(chunk)
                    self.retry.clear()
                    self.warning = "Transcription failed; recording paused. Check the speech provider or connectivity, then Resume to retry. " + errors.message(exc)
                    await self.pause()
                self.persist()
            finally:
                self.busy = False

    async def shutdown(self):
        if self.capture:
            await asyncio.to_thread(self.capture.close)
            self.capture = None
        if self.active:
            self.duration = self.elapsed()
            self.state = "interrupted"
            self.warning = "Mellow closed during the meeting. Saved text is intact; pending audio was discarded."
            self.persist()
        for task in [self.worker, self.stop_task, *self.notes_tasks.values()]:
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        self.pending.clear()

    async def notes(self, mid):
        meeting = self.store.get(mid)
        if meeting["status"] in ACTIVE:
            raise RuntimeError("Stop the meeting and wait for transcription to finish before generating notes.")
        if mid in self.notes_tasks and not self.notes_tasks[mid].done():
            raise RuntimeError("Notes are already being generated for this meeting.")
        cfg = config.load()
        if not cfg.get("ai_enabled", True):
            raise RuntimeError("Choose an answer engine in Settings before generating notes.")
        self.store.update(mid, notes_status="generating", notes_error="")

        async def run():
            try:
                text = await meeting_notes.generate(meeting, cfg, lambda message: self.notes_progress.update({mid: message}))
                self.store.update(mid, notes=text, notes_status="ready", notes_error="",
                                  engine=f"{cfg['llm']['provider']} / {cfg['llm']['model'] or 'agent default'}")
            except asyncio.CancelledError:
                self.store.update(mid, notes_status="error", notes_error="Notes generation was interrupted. Try again.")
                raise
            except Exception as exc:
                self.store.update(mid, notes_status="error", notes_error=errors.message(exc))
            finally:
                self.notes_progress.pop(mid, None)

        self.notes_tasks[mid] = asyncio.create_task(run())
        return {"status": "generating"}


manager = Manager()


async def trusted(request: Request):
    allowed = {"http://localhost:1420", "http://127.0.0.1:1420", "http://tauri.localhost", "tauri://localhost"}
    origin = request.headers.get("origin")
    if request.url.hostname not in {"localhost", "127.0.0.1", "testserver"} or (origin and origin not in allowed):
        raise HTTPException(403, "Untrusted meeting request")
    if request.method != "GET" and origin not in allowed:
        raise HTTPException(403, "Start meeting actions from Mellow.")


router = APIRouter(prefix="/meetings", dependencies=[Depends(trusted)])


class Start(BaseModel):
    title: str = Field(default="", max_length=160)
    microphone: int | None = Field(default=None, ge=0)
    output: int | None = Field(default=None, ge=0)


@router.get("/devices")
async def audio_devices():
    try:
        return await asyncio.to_thread(meeting_audio.devices)
    except Exception as exc:
        raise HTTPException(400, "Could not list meeting audio devices. " + errors.message(exc)) from exc


@router.get("/status")
async def status():
    return manager.status()


@router.post("/start")
async def start(body: Start):
    try:
        return await manager.start(body.title, body.microphone, body.output)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/levels")
async def check_levels(body: Start):
    async with manager.lock:
        if manager.active:
            raise HTTPException(409, "Use the active meeting's audio meters instead.")
        failures = []
        capture = meeting_audio.Capture(lambda chunk: None, failures.append, time.monotonic())
        peaks = {"You": 0.0, "Other participants": 0.0}
        try:
            await asyncio.to_thread(capture.start, body.microphone, body.output, config.load()["stt"].get("input_device"))
            for _ in range(20):
                await asyncio.sleep(0.1)
                for name in peaks:
                    peaks[name] = max(peaks[name], capture.levels[name])
            if failures:
                raise RuntimeError(failures[0])
            return {"levels": peaks}
        except Exception as exc:
            raise HTTPException(400, "Could not check audio levels. " + errors.message(exc)) from exc
        finally:
            await asyncio.to_thread(capture.close)


@router.post("/pause")
async def pause():
    return await manager.pause()


@router.post("/resume")
async def resume():
    try:
        return await manager.resume()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/stop")
async def stop():
    return await manager.stop()


@router.get("")
async def listing():
    return {"meetings": await asyncio.to_thread(manager.store.list)}


def require(mid):
    try:
        return manager.store.get(mid)
    except KeyError as exc:
        raise HTTPException(404, "Meeting not found") from exc


@router.get("/{mid}")
async def detail(mid: str):
    meeting = await asyncio.to_thread(require, mid)
    return {**meeting, "notes_progress": manager.notes_progress.get(mid, "")}


class Title(BaseModel):
    title: str = Field(min_length=1, max_length=160)


@router.put("/{mid}")
async def rename(mid: str, body: Title):
    require(mid)
    if not body.title.strip():
        raise HTTPException(400, "Enter a meeting title")
    manager.store.update(mid, title=body.title.strip())
    return {"ok": True}


@router.post("/{mid}/delete")
async def delete(mid: str):
    meeting = require(mid)
    if meeting["status"] in ACTIVE or meeting["notes_status"] == "generating":
        raise HTTPException(409, "Stop recording and wait for notes generation before deleting this meeting.")
    await asyncio.to_thread(manager.store.delete, mid)
    return {"ok": True}


@router.get("/{mid}/export")
async def download(mid: str, format: str = "md"):
    if format not in {"md", "txt", "json"}:
        raise HTTPException(400, "Choose md, txt or json")
    return {"text": export(require(mid), format), "filename": f"mellow-meeting-{mid}.{format}"}


@router.post("/{mid}/notes")
async def notes(mid: str):
    require(mid)
    try:
        return await manager.notes(mid)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
