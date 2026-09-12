"""Mellow sidecar: the AI half of the app."""

import asyncio
import json
import logging
import re
import sys
import threading
import time
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import sounddevice as sd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from mellowd import (
    act, agents, capture, config, errors, llm, locator, meetings, perf, point, remind, sessions, stt, transport, tts, writing,
)
from mellowd.version import PROTOCOL, SERVICE, VERSION

log = logging.getLogger("mellowd")

HOST = "127.0.0.1"
PORT = 8765

# Last N turns kept for context.
HISTORY_TURNS = 10
_transcription_lock = threading.Lock()

# Voice probe.
TTS_PROBE = "hi, this is how mellow sounds."

# Microphone retry timing.
WARM_RETRY_SECONDS = 2.0
WARM_SLOW_AFTER = 30
WARM_SLOW_SECONDS = 10.0

# Reminders are set to the minute
REMINDER_TICK_SECONDS = 20.0

# Meter tuning.
METER_INTERVAL = 0.05
METER_ATTACK = 0.65
METER_RELEASE = 0.20


def set_dpi_aware() -> None:
    """Measure in real pixels, and settle it before anything measures anything."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def standby() -> bool:
    """True while the AI half must stay completely off."""
    if not config.CONFIG_PATH.exists():
        return True
    return not config.load().get("ai_enabled", True)


# Pet-only reply.
PET_ONLY_LINE = "I'm just the pet right now. Settings can turn my brain back on."

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the speech model at boot instead of inside the first keypress."""

    set_dpi_aware()
    cfg = config.load()
    # Runtime summary.
    log.info(
        "mellowd on %s | llm %s/%s | prompt %s",
        sys.executable,
        cfg["llm"]["provider"],
        cfg["llm"]["model"],
        "default" if cfg["system_prompt"] == config.DEFAULTS["system_prompt"] else "custom",
    )

    # Run retention at boot.
    try:
        await asyncio.to_thread(sessions.sweep)
    except Exception:
        log.exception("session log sweep failed")

    await asyncio.to_thread(meetings.manager.store.recover)
    task = asyncio.create_task(warm_models())
    async with transport.lifespan():
        try:
            yield
        finally:
            await meetings.manager.shutdown()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def warm_models() -> None:
    """Warm the speech engines at boot."""
    # Skip pet-only mode.
    if standby():
        log.info("no brain configured; skipping model warm-up")
        return
    cfg = config.load()
    # Warm local engines only.
    for name, loader in (("stt", stt.load), ("tts", tts.load)):
        if cfg[name]["mode"] != "local":
            continue
        try:
            # Loader signatures differ.
            await asyncio.to_thread(loader, progress=_progress_cb(name))
        except Exception:
            # First use retries.
            log.exception("%s warm-up failed", name)


app = FastAPI(title="mellowd", lifespan=lifespan)
app.include_router(meetings.router)
TRUSTED_ORIGINS = {
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://tauri.localhost",
    "tauri://localhost",
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(TRUSTED_ORIGINS),
    allow_methods=["GET", "PUT", "POST"],
    allow_headers=["content-type"],
)


@app.get("/health")
async def health():
    # Shell identity fields.
    return {
        "ok": True,
        "service": SERVICE,
        "protocol": PROTOCOL,
        "version": VERSION,
    }


def _merge_section(name: str, current: dict, submitted: dict) -> dict:
    """Merge one capability's form without leaking its key to a different host."""
    merged = {**current, **submitted}
    merged.pop("has_api_key", None)
    # Agent mode has no HTTP destination.
    if merged.get("mode") == "agent" or current.get("mode") == "agent":
        # Keep a saved key when blank.
        if not str(merged.get("api_key") or "").strip():
            merged["api_key"] = current["api_key"]
        return merged
    if submitted.get("api_key"):
        return merged

    preset = config.PRESETS[name].get(merged.get("provider"), {})
    base = merged.get("base_url") or preset.get("base_url") or current["base_url"]
    try:
        same_destination = (
            merged.get("provider") == current["provider"]
            and config.normalize_base_url(str(base)) == current["base_url"]
        )
    except ValueError:
        same_destination = False  # Invalid destinations never match.
    merged["api_key"] = current["api_key"] if same_destination else ""
    return merged


def _candidate(body: dict) -> dict:
    """Merge a settings form over the saved config, capability by capability."""
    current = config.load()
    submitted = dict(body)
    merged = {**current, **submitted}
    for name in config.CAPABILITIES:
        section = submitted.get(name)
        merged[name] = _merge_section(
            name, current[name], section if isinstance(section, dict) else {}
        )
    return config.validate(merged)


def _engine_signature(cfg: dict) -> tuple[str, ...]:
    """The settings that define which brain owns a conversation."""
    if not cfg.get("ai_enabled", True):
        return ("pet",)
    section = cfg["llm"]
    if section.get("mode") == "agent":
        return (
            "ai",
            "agent",
            str(section.get("provider", "")),
            str(section.get("model", "")),
        )
    return (
        "ai",
        str(section.get("mode", "")),
        str(section.get("provider", "")),
        str(section.get("base_url", "")),
        str(section.get("model", "")),
    )


@app.get("/config")
async def get_config():
    return {
        "settings": config.redacted(config.load()),
        "presets": config.PRESETS,
        "stt_models": config.STT_MODELS,
        "tts_voices": config.KOKORO_VOICES,
        # Keep the form in sync.
        "reasoning_efforts": list(config.REASONING_EFFORTS),
        "vision_modes": list(config.VISION_MODES),
        # Default prompt.
        "default_prompt": config.DEFAULTS["system_prompt"],
    }


@app.put("/config")
async def put_config(body: dict):
    try:
        previous = config.load()
        cfg = _candidate(body)
        section = cfg["llm"]
        if section.get("mode") == "agent":
            # Validate the saved engine.
            await asyncio.to_thread(
                agents.require_exact_model,
                str(section.get("provider", "")),
                str(section.get("model", "")),
            )
        engine_changed = _engine_signature(previous) != _engine_signature(cfg)
        config.save(cfg)
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if engine_changed:
        await _reset_for_engine_change()
    elif previous.get("writing_enabled") != cfg.get("writing_enabled"):
        for session in list(_active_sessions.values()):
            await session.abort()
            session.writer.reset()
            await writing.status(session, send, "idle")
            await send(session.ws, type="state", state="idle")
    return {
        "settings": config.redacted(cfg),
        "engine_changed": engine_changed,
    }


@app.post("/config/test")
async def test_config(body: dict):
    try:
        cfg = _candidate(body)
        # Probe agent setup.
        probe = agents.test if cfg["llm"]["mode"] == "agent" else llm.test
        answer = await asyncio.wait_for(probe(cfg), 30.0)
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except asyncio.TimeoutError as e:
        raise HTTPException(status_code=504, detail="provider test timed out") from e
    except Exception as e:
        log.warning("provider test failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)[:300]) from e
    return {"ok": True, "reply": answer}


@app.get("/agents")
async def get_agents(refresh: bool = False):
    """Which coding-agent CLIs exist on this machine, detection and models."""
    return {"agents": await asyncio.to_thread(agents.catalog, refresh)}


@app.post("/agents/login")
async def agent_login(body: dict):
    """Connect to an agent: probe first, console only when it must be."""
    agent_id = str(body.get("agent", "")).strip()
    model = str(body.get("model", "")).strip()
    agent_speed = str(body.get("agent_speed") or "fast").strip().lower()
    if agent_id not in config.AGENT_PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown agent: {agent_id or '(empty)'}")
    if agent_speed not in config.AGENT_SPEEDS:
        raise HTTPException(
            status_code=400,
            detail=f"agent speed must be one of {', '.join(config.AGENT_SPEEDS)}",
        )
    if agents.find(agent_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"{config.AGENT_PRESETS[agent_id]['label']} is not installed",
        )
    try:
        # Validate the model first.
        await asyncio.to_thread(agents.require_exact_model, agent_id, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    signed, detail = await asyncio.to_thread(agents.auth_status, agent_id)
    if signed:
        try:
            verified, capability = await asyncio.wait_for(
                agents.check_capabilities(agent_id, model, agent_speed), 75.0
            )
        except asyncio.TimeoutError:
            verified, capability = False, "vision verification timed out"
        return {
            "ok": verified,
            "installed": True,
            "signed_in": True,
            "model_ok": verified,
            "vision_ok": verified,
            "detail": capability,
        }
    await asyncio.to_thread(agents.login, agent_id)
    return {
        "ok": True,
        "installed": True,
        "signed_in": False,
        "model_ok": False,
        "vision_ok": False,
        "detail": detail,
    }


@app.get("/audio/devices")
async def audio_devices():
    try:
        devices = await asyncio.to_thread(stt.input_devices)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not list microphones: {e}") from e
    return {"devices": devices}


@app.post("/stt/test")
async def test_stt(body: dict):
    if meetings.manager.active:
        raise HTTPException(409, "Stop the meeting before testing speech input.")
    try:
        # Include the STT key.
        cfg = _candidate({"stt": body.get("stt", {})})
        recorder = stt.Recorder(cfg)
        recorder.start()
        try:
            await asyncio.sleep(5)
        finally:
            audio = recorder.stop()
            # Release the test recorder.
            recorder.close()
        transcript = await asyncio.to_thread(stt.transcribe, audio, cfg)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("microphone test failed")
        raise HTTPException(status_code=500, detail=f"microphone test failed: {e}") from e
    section = cfg["stt"]
    return {
        **recorder.last_stats,
        "transcript": transcript,
        "model": section["model"] if section["mode"] == "cloud" else section["local_model"],
        "backend": stt.backend(),
    }


@app.post("/tts/voices")
async def tts_voices(body: dict):
    """List the ElevenLabs account's voices, from the form rather than the file."""
    section = body.get("tts") or {}
    try:
        # Supply a temporary voice.
        cfg = _candidate({"tts": {**section, "voice": section.get("voice") or "-"}})
        found = await asyncio.to_thread(tts.voices, cfg)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.warning("voice list failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)[:300]) from e
    return {"voices": found}


@app.post("/tts/test")
async def test_tts(body: dict):
    if meetings.manager.active:
        raise HTTPException(409, "Stop the meeting before playing a test voice.")
    """Say one line out loud with the submitted voice, saved or not."""
    try:
        cfg = _candidate({"tts": body.get("tts", {})})
        samples, rate = await asyncio.to_thread(tts.synth, TTS_PROBE, cfg)
        await asyncio.to_thread(sd.play, samples, rate, blocking=True)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.warning("voice test failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)[:300]) from e
    section = cfg["tts"]
    return {
        "ok": True,
        "backend": tts.backend(cfg),
        "voice": section["voice"] if section["mode"] == "cloud" else section["local_voice"],
        "seconds": round(len(samples) / rate, 2),
    }


# Model download progress.

_download_progress: dict[str, dict] = {
    "stt": {"state": "idle", "name": "", "done": 0, "total": 0, "error": "", "base": 0},
    "tts": {"state": "idle", "name": "", "done": 0, "total": 0, "error": "", "base": 0},
}
_download_tasks: dict[str, asyncio.Task] = {}


@app.get("/models/available")
async def available_models():
    """Cheap readiness used before offering an on-device voice preview."""
    return {"tts": tts.local_available()}


def _progress_cb(which: str):
    def cb(name: str, done: int, total: int) -> None:
        s = _download_progress[which]
        if name != s["name"]:
            s["base"] = s["done"] if s["name"] else 0
            s["name"] = name
        s["state"] = "running"
        s["done"] = s["base"] + done
        s["total"] = s["base"] + total
    return cb


async def _run_download(which: str, cfg: dict) -> None:
    s = _download_progress[which]
    try:
        if which == "stt":
            await asyncio.to_thread(stt.load, cfg, _progress_cb(which))
        else:
            await asyncio.to_thread(
                tts.load,
                progress=_progress_cb(which),
                cfg=cfg,
            )
        s["state"] = "done"
        log.info("%s model ready", which)
    except Exception as e:
        # Show a readable error.
        s["state"] = "failed"
        s["error"] = errors.message(e)
        log.warning("%s download failed: %s", which, e)


@app.post("/models/download")
async def start_model_download(body: dict):
    """Start loading one capability's on-device model, in the background."""
    which = str(body.get("which", "")).strip()
    if which not in _download_progress:
        raise HTTPException(status_code=400, detail="which must be 'stt' or 'tts'")
    try:
        # First-run has no config yet.
        cfg = _candidate(body.get("settings", {}))
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if cfg[which]["mode"] != "local":
        raise HTTPException(
            status_code=409,
            detail=f"{which} is configured for the cloud; there is nothing to download",
        )
    task = _download_tasks.get(which)
    if task and not task.done():
        return {"ok": True, "already": True}
    _download_progress[which].update(
        state="running", name="", done=0, total=0, error="", base=0
    )
    _download_tasks[which] = asyncio.create_task(_run_download(which, cfg))
    return {"ok": True}


@app.get("/models/progress")
async def model_progress():
    out = {}
    for which, s in _download_progress.items():
        entry = {k: v for k, v in s.items() if k != "base"}
        task = _download_tasks.get(which)
        if task and task.done() and not task.exception() and s["state"] == "running":
            entry["state"] = "done"
        out[which] = entry
    return out


@app.get("/reminders")
async def get_reminders():
    return {"reminders": remind.load()}


@app.put("/reminders")
async def put_reminders(body: dict):
    """The whole list, every time."""
    try:
        return {"reminders": await asyncio.to_thread(remind.save, body.get("reminders"))}
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/history")
async def get_history():
    """The session list, newest first."""
    return {"sessions": await asyncio.to_thread(sessions.list_sessions)}


@app.get("/history/{session_id}")
async def get_session(session_id: str):
    events = await asyncio.to_thread(sessions.read, session_id)
    if events is None:
        raise HTTPException(status_code=404, detail="no such session")
    return {"events": events}


@app.post("/history/new")
async def new_session():
    """Close the open conversation so the next turn starts a fresh one."""
    await asyncio.to_thread(sessions.close)
    return {"ok": True}


@app.post("/history/clear")
async def clear_history():
    n = await asyncio.to_thread(sessions.clear)
    return {"cleared": n}


async def send(ws: WebSocket, **msg) -> None:
    await ws.send_text(json.dumps(msg))
    if msg.get("type") == "reply_chunk" and msg.get("text"):
        perf.mark("first_text_emitted")


@dataclass
class Session:
    """Per-connection state."""

    ws: WebSocket
    recorder: stt.Recorder = field(default_factory=stt.Recorder)
    speaker: tts.Speaker = None
    history: list[dict] = field(default_factory=list)
    # Active model destination.
    destination: tuple[str, str, str] | None = None
    # In-flight turn.
    turn: asyncio.Task | None = None
    awake: bool = False
    warmup: asyncio.Task | None = None
    meter: asyncio.Task | None = None
    # Push-to-talk readiness.
    mic_ready: bool = False
    ptt_pending: asyncio.Task | None = None
    ptt_cancelled: threading.Event = field(default_factory=threading.Event)
    # Connection lifetime.
    alive: bool = True
    reminders: asyncio.Task | None = None
    # Capture visibility signal.
    hidden: asyncio.Event = field(default_factory=asyncio.Event)
    # Turn monitor.
    turn_monitor: dict | None = None
    writer: writing.Writer = field(default_factory=writing.Writer)

    def __post_init__(self) -> None:
        self.speaker = tts.Speaker(self.ws, send)

    def wake_mic(self) -> None:
        """Grab the microphone as soon as Windows allows, and hold it."""
        if meetings.manager.active:
            return
        self.awake = True
        # Pet-only mode skips the mic.
        if standby():
            self.mic_ready = False
            asyncio.create_task(self._send_mic("off"))
            return
        if self.mic_ready:
            return
        if self.warmup is None or self.warmup.done():
            self.warmup = asyncio.create_task(self._warm_mic())

    async def _send_mic(self, state: str) -> None:
        """Best-effort microphone state; disconnect owns any failed socket."""
        with suppress(WebSocketDisconnect, RuntimeError):
            await send(self.ws, type="microphone", state=state)

    async def _warm_mic(self) -> None:
        self.mic_ready = False
        await self._send_mic("warming")
        ready = await asyncio.to_thread(self._warm_open)
        self.mic_ready = bool(ready and self.awake)
        if self.mic_ready:
            await self._send_mic("ready")
        else:
            await self._send_mic("off")

    def start_meter(self) -> None:
        """Publish smoothed local microphone levels while push-to-talk is held."""
        if self.meter is None or self.meter.done():
            self.meter = asyncio.create_task(self._meter_levels())

    async def _meter_levels(self) -> None:
        shown = 0.0
        try:
            while self.alive and self.recorder.active:
                current = self.recorder.live_level
                strength = METER_ATTACK if current > shown else METER_RELEASE
                shown += (current - shown) * strength
                await send(self.ws, type="mic_level", level=round(shown, 3))
                await asyncio.sleep(METER_INTERVAL)
        except (WebSocketDisconnect, RuntimeError):
            return

    async def stop_meter(self) -> None:
        task, self.meter = getattr(self, "meter", None), None
        if task is None:
            return
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        with suppress(WebSocketDisconnect, RuntimeError):
            await send(self.ws, type="mic_level", level=0.0)

    def _warm_open(self) -> bool:
        # Runs in a worker thread.
        started = time.monotonic()
        probes = 0
        refreshed = False
        while self.awake:
            # Recheck config while probing.
            if standby() or meetings.manager.active:
                return False
            try:
                self.recorder.open(quiet=True)
            except Exception as e:
                probes += 1
                if probes == 1:
                    # Expected startup delay.
                    log.info(
                        "microphone busy at startup (%s) — retrying quietly", e
                    )
                elif probes == WARM_SLOW_AFTER:
                    # Report prolonged failure.
                    log.warning(
                        "microphone still refusing after %d tries (%s)", probes, e
                    )
                if probes >= 2 and not refreshed and stt.refresh_devices():
                    # Refresh a stale device cache once.
                    refreshed = True
                    log.info("refreshed portaudio device list")
                    continue
                time.sleep(
                    WARM_RETRY_SECONDS if probes < WARM_SLOW_AFTER else WARM_SLOW_SECONDS
                )
                continue
            if not self.awake:
                # Close after a mid-open nap.
                self.recorder.close()
            elif probes:
                log.info("microphone ready after %.1fs", time.monotonic() - started)
            return self.awake
        return False

    def watch_reminders(self) -> None:
        """Start the clock that keeps promises made before this connection."""
        if self.reminders is None or self.reminders.done():
            self.reminders = asyncio.create_task(self._tick_reminders())

    async def _tick_reminders(self) -> None:
        # Background tasks handle errors.
        while self.alive:
            await asyncio.sleep(REMINDER_TICK_SECONDS)
            if not self.alive:
                return
            try:
                items = await asyncio.to_thread(remind.load)
                fired, keep = remind.due(items, datetime.now())
                if not fired:
                    continue
                # Persist before sending.
                await asyncio.to_thread(remind.save, keep)
                for item in fired:
                    log.info("reminder fired: %s", item["text"])
                    await send(self.ws, type="remind", text=item["text"], id=item["id"])
            except (WebSocketDisconnect, RuntimeError):
                return  # Disconnect cleanup owns this.
            except Exception:
                log.exception("reminder tick failed")

    async def cancel_ptt_start(self) -> None:
        cancelled = getattr(self, "ptt_cancelled", None)
        if cancelled is not None:
            self.recorder.cancel_start(cancelled)
        task, self.ptt_pending = self.ptt_pending, None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def abort(self) -> None:
        """Stop whatever the pet is doing, right now."""
        self.writer.cancel()
        await self.cancel_ptt_start()
        self.recorder.stop()
        await self.stop_meter()
        if self.turn and not self.turn.done():
            self.turn.cancel()
            with suppress(asyncio.CancelledError):
                await self.turn
        self.turn = None
        await self.speaker.stop()


# Per-shell histories.
_active_sessions: dict[int, Session] = {}
_engine_revision = 0


async def _meeting_started():
    for session in list(_active_sessions.values()):
        session.awake = False
        session.mic_ready = False
        await session.abort()
        session.writer.reset()
        await writing.status(session, send, "idle")
        if session.warmup and not session.warmup.done():
            await session.warmup
        await asyncio.to_thread(session.recorder.close)
        await session._send_mic("off")
        with suppress(WebSocketDisconnect, RuntimeError):
            await send(session.ws, type="state", state="idle")


async def _meeting_stopped():
    for session in list(_active_sessions.values()):
        if session.alive and not meetings.manager.active:
            session.wake_mic()


meetings.manager.before_start = _meeting_started
meetings.manager.after_stop = _meeting_stopped


async def _reset_for_engine_change() -> None:
    """End the current conversation after a committed engine change."""
    global _engine_revision
    _engine_revision += 1
    for session in list(_active_sessions.values()):
        try:
            await session.abort()
        except Exception:
            log.exception("could not stop a session during engine change")
        finally:
            session.history.clear()
            session.destination = None
            session.writer.reset()
            await writing.status(session, send, "idle")
        try:
            await send(session.ws, type="state", state="idle")
        except (WebSocketDisconnect, RuntimeError):
            pass
    await asyncio.to_thread(sessions.close, reason="engine_changed")
    log.info("engine changed; current conversation closed")


def _said(cfg: dict, reply: str, aborted: bool) -> dict:
    """The fields an assistant_said event carries."""
    return {
        "text": reply,
        "model": cfg["llm"]["model"],
        "provider": cfg["llm"]["provider"],
        # Answering endpoint.
        "base_url": cfg["llm"]["base_url"],
        "aborted": aborted,
    }


class Shot(NamedTuple):
    """A screenshot and its physical-monitor coordinate space."""

    data: bytes
    width: int
    height: int
    # Full frame for local OCR.
    pixels: object
    # Source monitor.
    monitor: dict
    # Source window.
    hwnd: int


def _shot(
    max_edge: int = capture.MAX_EDGE, monitor: dict | None = None
) -> tuple[Shot | None, str, str]:
    """Capture + audit metadata in one blocking call. Never raises."""
    monitor = capture.known_monitor(monitor) or capture.active_monitor()
    hwnd, app, title = capture.window_on_monitor(monitor) if monitor else (0, "", "")
    grabbed = capture.grab(max_edge, monitor)
    shot = Shot(*grabbed, monitor, hwnd) if grabbed and monitor else None
    if not app and not title:
        app, title = capture.foreground()
    return shot, app, title


# Capture-hide timeout.
HIDE_TIMEOUT = 0.4


@perf.timed("capture")
async def _unseen_shot(
    session: Session, max_edge: int = capture.MAX_EDGE
) -> tuple[Shot | None, str, str]:
    """`_shot`, with Mellow's own windows out of the picture."""
    session.hidden.clear()
    await send(session.ws, type="capture", phase="begin")
    try:
        try:
            await asyncio.wait_for(session.hidden.wait(), HIDE_TIMEOUT)
        except asyncio.TimeoutError:
            # Capture despite a hide timeout.
            log.warning("shell did not confirm the hide in %.1fs, capturing anyway", HIDE_TIMEOUT)
        return await asyncio.to_thread(
            _shot, max_edge, getattr(session, "turn_monitor", None)
        )
    finally:
        # Always restore Mellow.
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(send(session.ws, type="capture", phase="end"))


# Defensive scan for action replies and screen-request preambles.
LOOK_SCAN = 64

_LOOK_PREAMBLES = (
    "sure", "let me", "let's", "i'll", "i will", "i need to",
    "one moment", "just a moment", "give me a moment",
)


def _hold_look_opening(text: str) -> bool:
    """Hold a possible screen request, but release ordinary prose immediately.

    The model contract puts [look] first. Keep the old short scan for common
    preambles as well, since some models say 'Let me look' before the marker.
    """
    head = text.lstrip().lower()
    if len(head) >= LOOK_SCAN:
        return False
    if not head or (head.startswith("[") and "]" not in head):
        return True
    return any(prefix.startswith(head) or head.startswith(prefix) for prefix in _LOOK_PREAMBLES)

# Screen marker.
_LOOK_TOKEN = re.compile(re.escape(llm.LOOK) + r"(?![0-9A-Za-z])", re.IGNORECASE)

# Point marker.
_POINT_TOKEN = re.compile(
    re.escape(llm.POINT) + r"\s*([^\]\n]{0,60}?)\s*\]",
    re.IGNORECASE,
)

# Marker tail limit.
POINT_HOLD = 120

# One-based target row.
Pick = int | str

# Point veto.
NONE = "none"


# Action marker.
_DO_TOKEN = re.compile(
    re.escape(llm.DO) + r"\s*([^\]\n]{0,160}?)\s*\]",
    re.IGNORECASE,
)

# Action and argument.
Deed = tuple[int | str, str]


def _split_point(text: str, token=None) -> tuple[str, str, Pick | Deed | None]:
    """(what to emit now, what to keep holding, a marker if one completed)."""
    token = token or _POINT_TOKEN
    doing = token is _DO_TOKEN
    point = None
    match = token.search(text)
    if match:
        body = match.group(1).strip()
        body, _, argument = body.partition("|")
        body, argument = body.strip(), argument.strip()
        if body.lower() == NONE or not body:
            point = NONE  # Explicit veto.
        elif body.isdigit():
            point = int(body)
        else:
            # Label target.
            point = body
        if doing and point is not NONE:
            point = (point, argument)
        text = text[: match.start()] + text[match.end() :]
    cut = text.rfind("[")
    if cut < 0 or "]" in text[cut:] or len(text) - cut > POINT_HOLD:
        return text, "", point
    return text[:cut], text[cut:], point


@perf.timed("answer_pass")
async def _pass(
    session: Session,
    cfg: dict,
    speak: bool,
    *,
    image: bytes | None = None,
    look: str = "",
    partial: dict | None = None,
    on_point=None,
    token=None,
) -> tuple[str, bool, Pick | Deed | None]:
    """One streaming pass of the model, into the bubble and the voice."""
    ws = session.ws
    sentences = tts.SentenceBuffer()
    held = ""
    settled = not look
    reply = ""
    # Hold a possible point marker.
    tail = ""
    point: Pick | None = None

    async def emit(text: str, final: bool = False) -> None:
        nonlocal reply, tail, point
        # Hide internal markers.
        if not text and not final:
            return
        # Combine the held tail first: [look] can arrive across several chunks.
        text, tail, found = _split_point(_LOOK_TOKEN.sub("", tail + text), token)
        if found and point is None:
            point = found
            if on_point is not None:
                await on_point(found)
        if final and tail:
            # Flush an unfinished tail.
            text, tail = text + tail, ""
        if not text:
            return
        reply += text
        if partial is not None:
            partial["text"] += text
        await send(ws, type="reply_chunk", text=text)
        if speak:
            for sentence in sentences.feed(text):
                await session.speaker.speak(sentence)

    def resolve(text: str) -> tuple[str, bool]:
        """(what to emit, whether the model asked for eyes) for a held opening."""
        if not _LOOK_TOKEN.search(text):
            return text, False
        if look == "ask":
            # Drop pre-capture filler.
            return "", True
        # Strip phase-two markers.
        return _LOOK_TOKEN.sub("", text), False

    try:
        # Both engines share a stream shape.
        stream = (
            agents.chat(session.history, cfg, image=image)
            if cfg.get("llm", {}).get("mode") == "agent"
            else llm.chat(session.history, cfg, image=image)
        )
        async with aclosing(stream):
            async for chunk in stream:
                if not settled:
                    held += chunk
                    text, asked = resolve(held)
                    if asked:
                        return "", True, None
                    if token is _DO_TOKEN and _declined(held):
                        # Respect an action veto.
                        return "", False, NONE
                    if not (token or _POINT_TOKEN).search(held):
                        if token is _DO_TOKEN or look not in ("ask", "strip"):
                            waiting = len(held.lstrip()) < LOOK_SCAN
                        else:
                            waiting = look == "ask" and _hold_look_opening(held)
                        if waiting:
                            continue
                    settled = True
                    chunk, held = text, ""
                await emit(chunk)
    except asyncio.CancelledError:
        # Preserve cancelled text.
        if held and partial is not None:
            text, asked = resolve(held)
            if not asked:
                partial["text"] += text
        raise

    if not settled and held:
        # Flush a short answer.
        text, asked = resolve(held)
        if asked:
            return "", True, None
        chunk, held = text, ""
        await emit(chunk)

    await emit("", final=True)

    if speak:
        for sentence in sentences.flush():
            await session.speaker.speak(sentence)
    return reply, False, point


async def _deliver(
    session: Session, text: str, speak: bool, partial: dict | None = None
) -> str:
    """Deliver an already-grounded answer without another model call."""
    reply = text.strip()
    if not reply:
        reply = "I couldn't lock onto a safe target on this screen."
    if partial is not None:
        partial["text"] += reply
    await send(session.ws, type="reply_chunk", text=reply)
    perf.mark("first_text_emitted")
    if speak:
        sentences = tts.SentenceBuffer()
        for sentence in sentences.feed(reply):
            await session.speaker.speak(sentence)
        for sentence in sentences.flush():
            await session.speaker.speak(sentence)
    return reply


def _seen_cfg(
    cfg: dict,
    shot: Shot,
    pointing: bool,
    guide: bool = False,
    items: str = "",
    target: str = "",
) -> dict:
    """cfg for a pass that holds a screenshot."""
    return {
        **cfg,
        "llm": {
            **cfg["llm"],
            "screen": "guide" if guide else "seen",
            "shot": (shot.width, shot.height),
            "point": pointing,
            "items": items,
            "target": target,
        },
    }


async def _hide_point(session: Session) -> None:
    """Take the bone away."""
    await send(session.ws, type="point", nx=None)


async def _aim(session: Session, target: point.Target) -> None:
    """Put the bone on a row of the list."""
    log.info("pointing at %r via %s", target.label, target.source)
    await send(
        session.ws,
        type="point",
        nx=target.nx,
        ny=target.ny,
        label=target.label,
        monitor=target.monitor,
    )
    perf.mark("pointer_dispatched")


def _declined(text: str) -> bool:
    """Did it open with [DO:none]? Checked before a word has been emitted."""
    found = _DO_TOKEN.search(text)
    return bool(found and found.group(1).strip().lower() == NONE)


def _chosen(deed, things: list[act.Thing]) -> tuple[act.Thing | None, str]:
    """The row the model chose and its argument, or nothing at all."""
    if not deed or deed is NONE:
        return None, ""
    which, argument = deed
    if isinstance(which, int):
        return (things[which - 1] if 1 <= which <= len(things) else None), argument
    # Match labels to offered rows.
    wanted = point.terms(which)
    best = None
    for thing in things:
        value = point.score(thing.label, wanted)
        if value >= point.THRESHOLD and (best is None or value > best[0]):
            best = (value, thing)
    return (best[1] if best else None), argument


def _picked(pick: Pick | None, cands: list[point.Target]) -> point.Target | None:
    """The row the model chose, or None if it chose nothing that exists."""
    if pick is None or pick is NONE:
        return None
    if isinstance(pick, int):
        return cands[pick - 1] if 1 <= pick <= len(cands) else None
    # Match a label.
    wanted = point.terms(pick)
    best = None
    for cand in cands:
        value = point.score(cand.label, wanted)
        if value >= point.THRESHOLD and (best is None or value > best[0]):
            best = (value, cand)
    if best:
        log.info("pick %r matched row %r (%.2f)", pick, best[1].label, best[0])
        return best[1]
    log.info("pick %r is on no row that was offered", pick)
    return None


def _act_cfg(cfg: dict, things: list[act.Thing]) -> dict:
    """cfg for a turn that is about doing rather than seeing or saying."""
    return {**cfg, "llm": {**cfg["llm"], "doing": act.describe(things)}}


@perf.timed("action")
async def _act(
    session: Session, cfg: dict, speak: bool, partial: dict, prompt: str
) -> tuple[str, bool]:
    """Try to do what they asked."""
    things = await asyncio.to_thread(act.catalog, prompt)
    # Exact names beat fuzzy scores.
    if not things or (things[0].score < act.THRESHOLD and not act.direct(prompt, things)):
        log.info("act: nothing on this machine matches %r", prompt)
        return "", False

    done: list[str] = []

    async def execute(thing: act.Thing, argument: str) -> None:
        try:
            said = await asyncio.to_thread(act.run, thing, argument)
            log.info("act: %s", said)
            done.append(said)
            if thing.kind in act.ON_SCREEN:
                # Pomodoro runs in the frontend.
                await send(
                    session.ws,
                    type="pomodoro",
                    action="stop" if thing.kind.endswith("stop") else "start",
                    minutes=act.minutes(argument),
                )
            # Avoid the reserved `kind` argument.
            await asyncio.to_thread(
                sessions.record,
                "acted",
                what=thing.kind,
                name=thing.label,
                detail=argument,
            )
        except Exception:
            # Keep failures contained.
            log.exception("act: %s failed", thing.label)

    async def fire(deed) -> None:
        thing, argument = _chosen(deed, things)
        if thing is not None:
            if thing.kind in ("youtube", "spotify"):
                argument = act.media_argument(prompt, argument) or argument
            await execute(thing, argument)

    # Run exact requests directly.
    immediate = act.direct(prompt, things)
    if immediate is not None:
        thing, argument = immediate
        await execute(thing, argument)
        if done:
            outcome = done[-1].strip().rstrip(".")
            confirmation = outcome[:1].upper() + outcome[1:] + "."
        else:
            confirmation = f"I couldn't open {thing.label}."
        reply = await _deliver(session, confirmation, speak, partial=partial)
        return reply, True

    reply, _, _ = await _pass(
        session,
        _act_cfg(cfg, things),
        speak,
        # Actions need no screenshot.
        look="pick",
        partial=partial,
        on_point=fire,
        token=_DO_TOKEN,
    )
    return reply, bool(done)


async def _prepare_pointing(session: Session, prompt: str):
    """Read-only work overlapped with writing classification; never call a model."""
    try:
        with perf.span("point_preparation"):
            shot, app, title = await _unseen_shot(session, capture.POINT_EDGE)
            if shot is None:
                return None
            cands = await asyncio.to_thread(
                point.candidates, prompt, shot.pixels, shot.monitor, None, shot.hwnd
            )
            return shot, app, title, cands
    except Exception:
        log.exception("early screen preparation failed; reading normally")
        return None


async def _discard_preparation(task) -> None:
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task


async def _resolve_point(prompt, shot, cfg, cands, history):
    if cfg["llm"]["mode"] in ("cloud", "agent"):
        try:
            return await locator.locate_and_answer(prompt, shot, cfg, cands, history)
        except locator.InvalidGrounding:
            perf.mark("grounded_fallback")
            log.info("combined output invalid; using the strict locator and separate answer")
    return locator.GroundedResult(await locator.locate(prompt, shot, cfg, cands), "")


async def answer(session: Session, prompt: str, prepared=None) -> None:
    """Stream one reply, speaking it sentence by sentence as it arrives."""
    ws = session.ws
    if not prompt:
        await send(ws, type="state", state="idle")
        return

    # Clear the previous point.
    await _hide_point(session)

    cfg = config.load()
    # Active model.
    destination = (
        cfg["llm"]["provider"],
        cfg["llm"]["base_url"],
        cfg["llm"]["model"],
    )
    if session.destination is not None and destination != session.destination:
        session.history.clear()
    session.destination = destination
    speak = cfg["tts"]["speak"]
    if speak:
        session.speaker.begin()

    session.history.append({"role": "user", "content": prompt})
    # Keep disk I/O off-loop.
    await asyncio.to_thread(sessions.record, "user_said", text=prompt)
    reply = ""
    # Preserve partial output.
    partial = {"text": ""}
    # Probe local vision.
    await asyncio.to_thread(llm.probe_vision, cfg["llm"])
    # Check model fit.
    await asyncio.to_thread(llm.check_fit, cfg["llm"])
    sighted = llm.vision_ok(cfg["llm"])
    # Route screen requests.
    pointing = sighted and capture.wants_pointing(prompt)
    asked = sighted and (capture.wants_screen(prompt) or pointing)
    # Selected target.
    aimed: point.Target | None = None
    # Answer generated alongside target selection.
    grounded_answer = ""
    # Actions require cloud or agent mode.
    doing = cfg["llm"]["mode"] in ("cloud", "agent") and capture.wants_action(prompt)
    try:
        if doing:
            reply, did = await _act(session, cfg, speak, partial, prompt)
            if did:
                await asyncio.to_thread(
                    sessions.record, "assistant_said", **_said(cfg, reply, False)
                )
                session.history.append({"role": "assistant", "content": reply})
                del session.history[: max(0, len(session.history) - HISTORY_TURNS * 2)]
                if speak:
                    await session.speaker.finish()
                await send(ws, type="state", state="idle")
                return
        if not asked:
            reply, asked, _ = await _pass(
                session,
                cfg,
                speak,
                # Vision-off has no marker.
                look="ask" if sighted else "",
                partial=partial,
            )
        if asked:
            # Show screen processing.
            await send(ws, type="state", state="looking")
            # Pointing uses a smaller frame.
            early = await prepared if pointing and prepared is not None else None
            if early is not None:
                shot, app, title, cands = early
                perf.mark("point_preparation_reused")
            else:
                shot, app, title = await _unseen_shot(
                    session, capture.POINT_EDGE if pointing else capture.MAX_EDGE
                )
                cands = []
            if shot:
                saved = await asyncio.to_thread(capture.media_bytes, shot.data)
                await asyncio.to_thread(
                    sessions.record,
                    "screen_captured",
                    app=app,
                    title=title,
                    file=saved or "",
                )
                log.info("screen turn: %s | %s", app or "?", title[:80])
                if pointing:
                    if early is None:
                        cands = await asyncio.to_thread(
                            point.candidates, prompt, shot.pixels,
                            shot.monitor, None, shot.hwnd,
                        )

                    # Resolve a measured target.
                    grounded = await _resolve_point(prompt, shot, cfg, cands, session.history)
                    aimed, grounded_answer = grounded.target, grounded.answer
                    if aimed:
                        fresh, _, _ = await _unseen_shot(session, capture.POINT_EDGE)
                        if fresh is None:
                            log.info("could not verify the localized target; withholding the bone")
                            perf.mark("pointer_withheld")
                            aimed = None
                            grounded_answer = ""
                        elif locator.changed_at(shot, fresh, aimed):
                            perf.mark("target_changed_retry")
                            log.info("localized area changed; resolving once on the fresh frame")
                            shot = fresh
                            cands = await asyncio.to_thread(
                                point.candidates,
                                prompt,
                                shot.pixels,
                                shot.monitor,
                                None,
                                shot.hwnd,
                            )
                            grounded = await _resolve_point(prompt, shot, cfg, cands, session.history)
                            aimed, grounded_answer = grounded.target, grounded.answer
                            if aimed:
                                verified, _, _ = await _unseen_shot(
                                    session, capture.POINT_EDGE
                                )
                                if verified is None or locator.changed_at(
                                    shot, verified, aimed
                                ):
                                    log.info(
                                        "localized area moved twice; withholding the bone"
                                    )
                                    perf.mark("pointer_withheld")
                                    aimed = None
                                    grounded_answer = ""
                                else:
                                    shot = verified
                        else:
                            shot = fresh
                        if aimed:
                            await _aim(session, aimed)

                if pointing and grounded_answer:
                    reply = await _deliver(
                        session, grounded_answer, speak, partial=partial
                    )
                else:
                    reply, _, _ = await _pass(
                        session,
                        _seen_cfg(
                            cfg,
                            shot,
                            False,
                            target=(
                                f'"{aimed.label}" ({aimed.kind or aimed.source})'
                                if aimed
                                else ""
                            ),
                        ),
                        speak,
                        image=shot.data,
                        look="strip",
                        partial=partial,
                    )
            else:
                # Report capture failure.
                blind = {**cfg, "llm": {**cfg["llm"], "screen": "failed"}}
                reply, _, _ = await _pass(session, blind, speak, partial=partial)
    except asyncio.CancelledError:
        # Preserve interrupted speech.
        text = partial["text"] or reply
        if text.strip():
            await asyncio.to_thread(sessions.record, "assistant_said", **_said(cfg, text, True))
        # Clear the point despite cancellation.
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(_hide_point(session))
        raise

    await asyncio.to_thread(
        sessions.record, "assistant_said", **_said(cfg, reply, False)
    )
    session.history.append({"role": "assistant", "content": reply})
    del session.history[: max(0, len(session.history) - HISTORY_TURNS * 2)]

    if speak:
        # Wait for playback.
        await session.speaker.finish()

    # The frontend retires the bone.
    await send(ws, type="state", state="idle")


async def run_turn(session: Session, prompt: str) -> None:
    """A turn owns its own error handling, because as a separate task it's outside the message loop's"""
    prepared = None
    try:
        if config.load().get("writing_enabled") and prompt:
            await _hide_point(session)
            cfg = config.load()
            if (cfg.get("ai_enabled") and cfg["llm"]["mode"] == "cloud"
                    and llm.vision_ok(cfg["llm"]) and capture.wants_pointing(prompt)
                    and not capture.wants_action(prompt)):
                prepared = asyncio.create_task(_prepare_pointing(session, prompt))
            options = {"discard_preparation": lambda: _discard_preparation(prepared)} if prepared is not None else {}
            if await writing.handle(session, prompt, send, **options):
                return
        if prepared is None:
            await answer(session, prompt)
        else:
            await answer(session, prompt, prepared=prepared)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("turn failed")
        perf.outcome("failed")
        # Record failed turns.
        await asyncio.to_thread(
            sessions.record, "turn_failed", reason=errors.message(e)
        )
        # Clear failed points.
        with suppress(Exception):
            await _hide_point(session)
        await session.speaker.stop()
        await send(session.ws, type="error", message=errors.message(e))
        await send(session.ws, type="state", state="idle")
    finally:
        await _discard_preparation(prepared)


@perf.timed("transcription")
def _transcribe_voice(audio, cancelled):
    # Serialize local transcription.
    with _transcription_lock:
        return "" if cancelled.is_set() else stt.transcribe(audio)


async def _voice_turn(session: Session, audio) -> None:
    try:
        text = await asyncio.to_thread(_transcribe_voice, audio, session.writer.cancelled)
        if meetings.manager.active or session.writer.cancelled.is_set():
            perf.outcome("cancelled")
            return
        if not text and session.recorder.last_stats["peak"] < stt.MIN_PEAK:
            await asyncio.to_thread(session.recorder.reopen)
            text_shown = "that was too quiet — say it again"
        else:
            text_shown = text or "…didn't catch that"
        # Show only recognition failures.
        if not text:
            perf.outcome("no_speech")
            await send(session.ws, type="transcript", text=text_shown)
        await run_turn(session, text)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        perf.outcome("failed")
        await send(session.ws, type="error", message=errors.message(exc))
        await send(session.ws, type="state", state="idle")


async def _writing_start(session: Session) -> None:
    session.writer.begin()
    await writing.status(session, send, "idle")
    if config.load().get("writing_enabled"):
        # Resolve the writing target.
        session.writer.finding = asyncio.create_task(
            asyncio.to_thread(writing.desktop.resolve, writing.FIND_SECONDS)
        )


async def _listen_when_ready(session: Session) -> None:
    """Honor this held press after wake-up without blocking key-release messages."""
    cancelled = session.ptt_cancelled
    try:
        if not session.mic_ready:
            session.wake_mic()
            if session.warmup is not None:
                # A release cancels this press, not the shared microphone warm-up.
                await asyncio.shield(session.warmup)
        if (cancelled.is_set() or not session.alive or not session.awake
                or not session.mic_ready or meetings.manager.active or standby()):
            return
        await _writing_start(session)
        await asyncio.to_thread(session.recorder.start, cancelled)
        if cancelled.is_set():
            return
        await send(session.ws, type="state", state="listening")
        session.start_meter()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("starting push-to-talk failed")
        session.mic_ready = False
        session.recorder.stop()
        session.writer.cancel()
        await session.stop_meter()
        await send(session.ws, type="error", message=errors.message(exc))
        await send(session.ws, type="state", state="idle")


async def handle(session: Session, msg: dict) -> None:
    ws = session.ws
    kind = msg.get("type")
    received = time.perf_counter()
    if meetings.manager.active and kind in {"ptt_start", "ptt_end", "text", "writing_retry"}:
        await send(ws, type="error", message="Meeting transcription is active. Stop the meeting before talking to Mellow.")
        return

    if kind == "ping":
        await send(ws, type="pong", echo=msg.get("text", ""))

    elif kind == "capture_ready":
        # Capture is ready.
        session.hidden.set()

    elif kind == "awake":
        # Wake the mic keeper.
        if msg.get("value"):
            session.wake_mic()
        else:
            session.awake = False
            session.mic_ready = False
            await session.cancel_ptt_start()
            await asyncio.to_thread(session.recorder.close)
            await session._send_mic("off")

    elif kind == "set_speak":
        cfg = config.load()
        cfg["tts"]["speak"] = bool(msg.get("value"))
        await asyncio.to_thread(config.save, cfg)
        if not cfg["tts"]["speak"]:
            await session.speaker.stop()
        await send(ws, type="speak", value=cfg["tts"]["speak"])

    elif kind == "ptt_start":
        # Stop current speech first.
        await session.abort()
        session.turn_monitor = None
        # Pet-only mode cannot listen.
        if standby():
            await send(ws, type="reply_chunk", text=PET_ONLY_LINE)
            await send(ws, type="state", state="idle")
            return
        session.ptt_cancelled = threading.Event()
        session.ptt_pending = asyncio.create_task(_listen_when_ready(session))

    elif kind == "ptt_end":
        await session.cancel_ptt_start()
        # Ignore unmatched releases.
        if not session.recorder.active:
            session.writer.cancel()
            return
        session.turn_monitor = capture.known_monitor(msg.get("monitor"))
        if msg.get("monitor") is not None and session.turn_monitor is None:
            log.warning("ignored an invalid cursor monitor on ptt_end")
        audio = session.recorder.stop()
        await session.stop_meter()
        await send(ws, type="state", state="thinking")
        # Keep transcription cancellable.
        session.turn = asyncio.create_task(perf.run(
            _voice_turn(session, audio), perf.Turn("voice", received)
        ))

    elif kind == "text":
        await session.abort()
        session.turn_monitor = capture.known_monitor(msg.get("monitor"))
        if msg.get("monitor") is not None and session.turn_monitor is None:
            log.warning("ignored an invalid cursor monitor on text submission")
        # Match pet-only hotkey behavior.
        if standby():
            await send(ws, type="reply_chunk", text=PET_ONLY_LINE)
            await send(ws, type="state", state="idle")
            return
        await send(ws, type="state", state="thinking")
        await _writing_start(session)
        session.turn = asyncio.create_task(
            perf.run(run_turn(session, msg.get("text", "").strip()), perf.Turn("text", received))
        )

    elif kind == "cancel":
        await session.abort()
        session.recorder.stop()
        await writing.status(session, send, "idle")
        await send(ws, type="state", state="idle")

    elif kind == "writing_retry":
        if session.turn is None or session.turn.done():
            session.turn = asyncio.create_task(writing.retry(session, send, str(msg.get("id", ""))))

    elif kind == "writing_dismiss":
        if msg.get("id") == session.writer.id:
            await session.abort()
            await writing.status(session, send, "idle")
            await send(ws, type="state", state="idle")

    elif kind == "new_conversation":
        # Reset both histories.
        await session.abort()
        session.history.clear()
        session.destination = None
        session.writer.reset()
        await writing.status(session, send, "idle")
        await asyncio.to_thread(sessions.close)
        await send(ws, type="state", state="idle")

    else:
        await send(ws, type="error", message=f"unknown message: {kind!r}")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    origin = ws.headers.get("origin")
    if origin and origin not in TRUSTED_ORIGINS:
        await ws.close(code=1008, reason="untrusted origin")
        return
    await ws.accept()
    log.info("shell connected")
    try:
        await send(ws, type="state", state="idle")
        # Initialize the tray voice state.
        await send(ws, type="speak", value=config.load()["tts"]["speak"])
    except WebSocketDisconnect:
        # Ignore the abandoned dev socket.
        log.info("shell left during the greeting")
        return
    session = Session(ws)
    connected_revision = _engine_revision
    # Resume conversation state.
    session.history, session.destination = await asyncio.to_thread(sessions.resume)
    # Reject a stale resume.
    if connected_revision != _engine_revision:
        session.history.clear()
        session.destination = None
    _active_sessions[id(session)] = session
    if session.history:
        log.info("resumed %d message(s) from the open session", len(session.history))
    # Reminders stay independent.
    session.watch_reminders()

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            try:
                await handle(session, msg)
            except Exception as e:
                # Guard each message.
                if type(e) is RuntimeError:
                    log.warning("handler %r refused: %s", msg.get("type"), e)
                else:
                    log.exception("handler %r failed", msg.get("type"))
                await session.abort()
                await send(ws, type="error", message=errors.message(e))
                await send(ws, type="state", state="idle")

    except WebSocketDisconnect:
        log.info("shell disconnected")
    finally:
        _active_sessions.pop(id(session), None)
        session.awake = False  # Stop the mic keeper.
        session.mic_ready = False
        session.alive = False  # Stop reminders.
        session.recorder.close()
        # Stop audio on close.
        await session.abort()


def run() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
