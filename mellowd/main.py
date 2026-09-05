"""Mellow sidecar: the AI half of the app."""

import asyncio
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import sounddevice as sd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from mellowd import (
    act, agents, capture, config, errors, llm, locator, meetings, point, remind, sessions, stt, tts,
)
from mellowd.version import PROTOCOL, SERVICE, VERSION

log = logging.getLogger("mellowd")

HOST = "127.0.0.1"
PORT = 8765

# Last N turns kept for context.
HISTORY_TURNS = 10

# Short, and it names the thing being tested, so a wrong voice is obvious.
TTS_PROBE = "hi, this is how mellow sounds."

# After app launch the OS refuses this microphone to everyone in the process for a while (a capped
WARM_RETRY_SECONDS = 2.0
WARM_SLOW_AFTER = 30
WARM_SLOW_SECONDS = 10.0

# Reminders are set to the minute
REMINDER_TICK_SECONDS = 20.0


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


# What the bubble shows when someone talks to a pet that has no brain.
PET_ONLY_LINE = "I'm just the pet right now. Settings can turn my brain back on."

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the speech model at boot instead of inside the first keypress."""

    set_dpi_aware()
    cfg = config.load()
    # Which interpreter, which model, whose prompt.
    log.info(
        "mellowd on %s | llm %s/%s | prompt %s",
        sys.executable,
        cfg["llm"]["provider"],
        cfg["llm"]["model"],
        "default" if cfg["system_prompt"] == config.DEFAULTS["system_prompt"] else "custom",
    )

    # Retention runs at boot, not on a timer: it rarely deletes anything
    try:
        await asyncio.to_thread(sessions.sweep)
    except Exception:
        log.exception("session log sweep failed")

    await asyncio.to_thread(meetings.manager.store.recover)
    task = asyncio.create_task(warm_models())
    yield
    await meetings.manager.shutdown()
    task.cancel()


async def warm_models() -> None:
    """Warm the speech engines at boot."""
    # First run, or just the pet: neither download may fire.
    if standby():
        log.info("no brain configured; skipping model warm-up")
        return
    cfg = config.load()
    # Only what actually runs here. A cloud engine has nothing to load
    for name, loader in (("stt", stt.load), ("tts", tts.load)):
        if cfg[name]["mode"] != "local":
            continue
        try:
            # By keyword, because the two loaders do not have the same shape
            await asyncio.to_thread(loader, progress=_progress_cb(name))
        except Exception:
            # Not fatal. Both retry their load on first use and report properly
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
    # The shell checks all four fields before trusting an existing listener on Mellow's fixed
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
    # Agent mode speaks no HTTP transport, so "same place" has no meaning here
    if merged.get("mode") == "agent" or current.get("mode") == "agent":
        # Blank still means "keep the saved one" here
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
        same_destination = False  # unparseable is not "the same place"
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
        # Served rather than hardcoded in the form, so the two can't drift.
        "reasoning_efforts": list(config.REASONING_EFFORTS),
        "vision_modes": list(config.VISION_MODES),
        # The shipped prompt
        "default_prompt": config.DEFAULTS["system_prompt"],
    }


@app.put("/config")
async def put_config(body: dict):
    try:
        previous = config.load()
        cfg = _candidate(body)
        section = cfg["llm"]
        if section.get("mode") == "agent":
            # A saved engine is a promise about what will run.
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
    return {
        "settings": config.redacted(cfg),
        "engine_changed": engine_changed,
    }


@app.post("/config/test")
async def test_config(body: dict):
    try:
        cfg = _candidate(body)
        # Agent probes cover installation and sign-in failures.
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
        # Do this before the capability probe
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
        # The whole stt section, key included
        cfg = _candidate({"stt": body.get("stt", {})})
        recorder = stt.Recorder(cfg)
        recorder.start()
        try:
            await asyncio.sleep(5)
        finally:
            audio = recorder.stop()
            # This recorder is a throwaway
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
        # ElevenLabs validation needs a temporary voice before listing voices.
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


# - Model downloads (step 8) The wizard's download screen polls this twice a second.

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
        # A sentence, not a traceback: this goes in the wizard's failed state.
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
        # First-run downloads deliberately happen before config.json exists.
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


@dataclass
class Session:
    """Per-connection state."""

    ws: WebSocket
    recorder: stt.Recorder = field(default_factory=stt.Recorder)
    speaker: tts.Speaker = None
    history: list[dict] = field(default_factory=list)
    # (provider, base_url, model) — see answer(). All three
    destination: tuple[str, str, str] | None = None
    # The in-flight turn. Held as a task so the message loop can keep reading while the pet talks
    turn: asyncio.Task | None = None
    awake: bool = False
    warmup: asyncio.Task | None = None
    # The shell uses this to gate push-to-talk during Windows' brief startup refusal.
    mic_ready: bool = False
    # Cleared on disconnect so the reminder tick stops at its next wake-up
    alive: bool = True
    reminders: asyncio.Task | None = None
    # Set when the shell confirms Mellow's windows are out of the frame.
    hidden: asyncio.Event = field(default_factory=asyncio.Event)
    # Physical monitor containing the cursor when this turn was submitted.
    turn_monitor: dict | None = None

    def __post_init__(self) -> None:
        self.speaker = tts.Speaker(self.ws, send)

    def wake_mic(self) -> None:
        """Grab the microphone as soon as Windows allows, and hold it."""
        if meetings.manager.active:
            return
        self.awake = True
        # A pet with no brain never touches the microphone: first run
        if standby():
            self.mic_ready = False
            asyncio.create_task(self._send_mic("off"))
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

    def _warm_open(self) -> bool:
        # Runs in a thread. Never raises and logs one warning total
        started = time.monotonic()
        probes = 0
        refreshed = False
        while self.awake:
            # The config can change while the keeper probes (setup finished
            if standby() or meetings.manager.active:
                return False
            try:
                self.recorder.open(quiet=True)
            except Exception as e:
                probes += 1
                if probes == 1:
                    # Expected, and documented at WARM_RETRY_SECONDS
                    log.info(
                        "microphone busy at startup (%s) — retrying quietly", e
                    )
                elif probes == WARM_SLOW_AFTER:
                    # Past the window it is no longer a warm-up
                    log.warning(
                        "microphone still refusing after %d tries (%s)", probes, e
                    )
                if probes >= 2 and not refreshed and stt.refresh_devices():
                    # Once: if this process's device cache was poisoned by initialising mid-churn
                    refreshed = True
                    log.info("refreshed portaudio device list")
                    continue
                time.sleep(
                    WARM_RETRY_SECONDS if probes < WARM_SLOW_AFTER else WARM_SLOW_SECONDS
                )
                continue
            if not self.awake:
                # Napped while the open was in flight
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
        # Its own error handling, because a create_task runs outside the message loop's guard
        while self.alive:
            await asyncio.sleep(REMINDER_TICK_SECONDS)
            if not self.alive:
                return
            try:
                items = await asyncio.to_thread(remind.load)
                fired, keep = remind.due(items, datetime.now())
                if not fired:
                    continue
                # Persisted before sending: if this frame never lands
                await asyncio.to_thread(remind.save, keep)
                for item in fired:
                    log.info("reminder fired: %s", item["text"])
                    await send(self.ws, type="remind", text=item["text"], id=item["id"])
            except (WebSocketDisconnect, RuntimeError):
                return  # the socket went away; the disconnect handler owns cleanup
            except Exception:
                log.exception("reminder tick failed")

    async def abort(self) -> None:
        """Stop whatever the pet is doing, right now."""
        if self.turn and not self.turn.done():
            self.turn.cancel()
            with suppress(asyncio.CancelledError):
                await self.turn
        self.turn = None
        await self.speaker.stop()


# Every live shell has its own in-memory model history.
_active_sessions: dict[int, Session] = {}
_engine_revision = 0


async def _meeting_started():
    for session in list(_active_sessions.values()):
        session.awake = False
        session.mic_ready = False
        await session.abort()
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
        # Which endpoint actually answered
        "base_url": cfg["llm"]["base_url"],
        "aborted": aborted,
    }


class Shot(NamedTuple):
    """A screenshot and its physical-monitor coordinate space."""

    data: bytes
    width: int
    height: int
    # The unshrunk frame, for point.find's OCR tier. Never sent anywhere.
    pixels: object
    # The physical desktop monitor these pixels came from.
    monitor: dict
    # Top-level application underneath Mellow on that monitor.
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


# How long to wait for the shell to take Mellow out of the frame.
HIDE_TIMEOUT = 0.4


async def _unseen_shot(
    session: Session, max_edge: int = capture.MAX_EDGE
) -> tuple[Shot | None, str, str]:
    """`_shot`, with Mellow's own windows out of the picture."""
    session.hidden.clear()
    await send(session.ws, type="capture", phase="begin")
    try:
        await asyncio.wait_for(session.hidden.wait(), HIDE_TIMEOUT)
    except asyncio.TimeoutError:
        # Degraded, not broken: the shot still happens, Mellow is just in it.
        log.warning("shell did not confirm the hide in %.1fs, capturing anyway", HIDE_TIMEOUT)
    try:
        return await asyncio.to_thread(
            _shot, max_edge, getattr(session, "turn_monitor", None)
        )
    finally:
        # Always, including on barge-in. Shielded because a cancelled task raises at its next await
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(send(session.ws, type="capture", phase="end"))


# How much of a pass to inspect before committing it to the bubble and the voice.
LOOK_SCAN = 64

# The marker as a standalone token: bracketed
_LOOK_TOKEN = re.compile(re.escape(llm.LOOK) + r"(?![0-9A-Za-z])", re.IGNORECASE)

# The other marker. [POINT:7] picking a row off the list
_POINT_TOKEN = re.compile(
    re.escape(llm.POINT) + r"\s*([^\]\n]{0,60}?)\s*\]",
    re.IGNORECASE,
)

# Longest a held-back tail may grow before it is released as ordinary text.
POINT_HOLD = 120

# Which row of point.candidates() to fly to: a 1-based index
Pick = int | str

# [POINT:none] as a value
NONE = "none"


# The third marker. [DO:7] or [DO:7|back in black]
_DO_TOKEN = re.compile(
    re.escape(llm.DO) + r"\s*([^\]\n]{0,160}?)\s*\]",
    re.IGNORECASE,
)

# What to do and what to do it to: a row of act.catalog
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
            point = NONE  # it would not help, and it said so
        elif body.isdigit():
            point = int(body)
        else:
            # A label rather than a number.
            point = body
        if doing and point is not NONE:
            point = (point, argument)
        text = text[: match.start()] + text[match.end() :]
    cut = text.rfind("[")
    if cut < 0 or "]" in text[cut:] or len(text) - cut > POINT_HOLD:
        return text, "", point
    return text[:cut], text[cut:], point


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
    # The tail of the stream, held back only while it could still be growing a [POINT:...].
    tail = ""
    point: Pick | None = None

    async def emit(text: str, final: bool = False) -> None:
        nonlocal reply, tail, point
        # The marker is machinery, never something to read.
        text = _LOOK_TOKEN.sub("", text)
        if not text and not final:
            return
        text, tail, found = _split_point(tail + text, token)
        if found and point is None:
            point = found
            if on_point is not None:
                await on_point(found)
        if final and tail:
            # The stream is over, so nothing is still arriving to close that bracket.
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
            # Whatever sits around it is throat-clearing before a question the model cannot answer yet
            return "", True
        # Phase 2 already has the screenshot, so the marker is noise here
        return _LOOK_TOKEN.sub("", text), False

    try:
        # One of two brains: an agent CLI streams the same shape llm.chat yields
        stream = (
            agents.chat(session.history, cfg, image=image)
            if cfg.get("llm", {}).get("mode") == "agent"
            else llm.chat(session.history, cfg, image=image)
        )
        async for chunk in stream:
            if not settled:
                held += chunk
                text, asked = resolve(held)
                if asked:
                    return "", True, None
                if token is _DO_TOKEN and _declined(held):
                    # It read the catalog and said none of it was the point.
                    return "", False, NONE
                # Hold the whole window, not just until a bracket shows up
                if not (token or _POINT_TOKEN).search(held) and len(held.lstrip()) < LOOK_SCAN:
                    continue
                settled = True
                chunk, held = text, ""
            await emit(chunk)
    except asyncio.CancelledError:
        # Barge-in inside the scan window. The words never reached the bubble or the voice
        if held and partial is not None:
            text, asked = resolve(held)
            if not asked:
                partial["text"] += text
        raise

    if not settled and held:
        # The stream ended inside the scan window. What is left is the whole answer
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
    """Deliver an already-grounded agent answer without another model call."""
    reply = text.strip()
    if not reply:
        reply = "I couldn't lock onto a safe target on this screen."
    if partial is not None:
        partial["text"] += reply
    await send(session.ws, type="reply_chunk", text=reply)
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
    # It answered with words. Matched against the rows it was shown and nothing else
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
    # It answered with words instead of a number.
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


async def _act(
    session: Session, cfg: dict, speak: bool, partial: dict, prompt: str
) -> tuple[str, bool]:
    """Try to do what they asked."""
    things = await asyncio.to_thread(act.catalog, prompt)
    if not things or things[0].score < act.THRESHOLD:
        log.info("act: nothing on this machine matches %r", prompt)
        return "", False

    done: list[str] = []

    async def execute(thing: act.Thing, argument: str) -> None:
        try:
            said = await asyncio.to_thread(act.run, thing, argument)
            log.info("act: %s", said)
            done.append(said)
            if thing.kind in act.ON_SCREEN:
                # The pomodoro lives in the frontend on purpose
                await send(
                    session.ws,
                    type="pomodoro",
                    action="stop" if thing.kind.endswith("stop") else "start",
                    minutes=act.minutes(argument),
                )
            # `what`, not `kind`: sessions.record's own first parameter is called kind
            await asyncio.to_thread(
                sessions.record,
                "acted",
                what=thing.kind,
                name=thing.label,
                detail=argument,
            )
        except Exception:
            # All of it inside the try now.
            log.exception("act: %s failed", thing.label)

    async def fire(deed) -> None:
        thing, argument = _chosen(deed, things)
        if thing is not None:
            if thing.kind in ("youtube", "spotify"):
                argument = act.media_argument(prompt, argument) or argument
            await execute(thing, argument)

    # Exact folders/sites/apps and explicit play requests do not need semantic arbitration.
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
        # No screenshot anywhere in here. Opening an app is not a question about what is on screen
        look="pick",
        partial=partial,
        on_point=fire,
        token=_DO_TOKEN,
    )
    return reply, bool(done)


async def answer(session: Session, prompt: str) -> None:
    """Stream one reply, speaking it sentence by sentence as it arrives."""
    ws = session.ws
    if not prompt:
        await send(ws, type="state", state="idle")
        return

    # Whatever the last turn pointed at, it is not what this one is about. The clear lives here
    await _hide_point(session)

    cfg = config.load()
    # The model belongs in here
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
    # Off the loop, like every other blocking write.
    await asyncio.to_thread(sessions.record, "user_said", text=prompt)
    reply = ""
    # What was actually said before a cancellation, across both passes
    partial = {"text": ""}
    # Ollama can say outright whether a local model takes images
    await asyncio.to_thread(llm.probe_vision, cfg["llm"])
    # Same trip out to Ollama, different question
    await asyncio.to_thread(llm.check_fit, cfg["llm"])
    sighted = llm.vision_ok(cfg["llm"])
    # Decided here, before the model gets a say
    pointing = sighted and capture.wants_pointing(prompt)
    asked = sighted and (capture.wants_screen(prompt) or pointing)
    # Which row of the on-screen list the bone ended up on, if any.
    aimed: point.Target | None = None
    # Agent mode returns this beside the selection
    grounded_answer = ""
    # Step 15a. Cloud and agent brains only, and that is a measured limit rather than caution
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
                # Vision off means the marker rule was never sent
                look="ask" if sighted else "",
                partial=partial,
            )
        if asked:
            # The pet needs something honest to do while the second round trip runs
            await send(ws, type="state", state="looking")
            # A pointing turn gets the smaller frame: it is choosing off a list
            shot, app, title = await _unseen_shot(
                session, capture.POINT_EDGE if pointing else capture.MAX_EDGE
            )
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
                cands = []
                if pointing:
                    cands = await asyncio.to_thread(
                        point.candidates,
                        prompt,
                        shot.pixels,
                        shot.monitor,
                        None,
                        shot.hwnd,
                    )

                    # Semantic vision chooses a region, then an exact measured hitbox or a fine-grid cell.
                    if cfg["llm"]["mode"] == "agent":
                        grounded = await locator.locate_and_answer(
                            prompt, shot, cfg, cands, session.history
                        )
                        aimed, grounded_answer = grounded.target, grounded.answer
                    else:
                        aimed = await locator.locate(prompt, shot, cfg, cands)
                    if aimed:
                        fresh, _, _ = await _unseen_shot(session, capture.POINT_EDGE)
                        if fresh is None:
                            log.info("could not verify the localized target; withholding the bone")
                            aimed = None
                        elif locator.changed_at(shot, fresh, aimed):
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
                            if cfg["llm"]["mode"] == "agent":
                                grounded = await locator.locate_and_answer(
                                    prompt, shot, cfg, cands, session.history
                                )
                                aimed, grounded_answer = (
                                    grounded.target,
                                    grounded.answer,
                                )
                            else:
                                aimed = await locator.locate(prompt, shot, cfg, cands)
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
                                    aimed = None
                                else:
                                    shot = verified
                        else:
                            shot = fresh
                        if aimed:
                            await _aim(session, aimed)

                if pointing and cfg["llm"]["mode"] == "agent":
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
                # No picture this turn, so no pretending
                blind = {**cfg, "llm": {**cfg["llm"], "screen": "failed"}}
                reply, _, _ = await _pass(session, blind, speak, partial=partial)
    except asyncio.CancelledError:
        # Barge-in. What Mellow managed to say is still what happened
        text = partial["text"] or reply
        if text.strip():
            await asyncio.to_thread(sessions.record, "assistant_said", **_said(cfg, text, True))
        # Shielded for the same reason the capture-end send is: a cancelled task raises at its next await
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(_hide_point(session))
        raise

    await asyncio.to_thread(
        sessions.record, "assistant_said", **_said(cfg, reply, False)
    )
    session.history.append({"role": "assistant", "content": reply})
    del session.history[: max(0, len(session.history) - HISTORY_TURNS * 2)]

    if speak:
        # Blocks until the audio genuinely stops
        await session.speaker.finish()

    # The bone is left where it is. The frontend retires it after ten quiet seconds
    await send(ws, type="state", state="idle")


async def run_turn(session: Session, prompt: str) -> None:
    """A turn owns its own error handling, because as a separate task it's outside the message loop's"""
    try:
        await answer(session, prompt)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("turn failed")
        # A log that shows the question and then nothing is worse than one that says the turn died
        await asyncio.to_thread(
            sessions.record, "turn_failed", reason=errors.message(e)
        )
        # A turn that died mid-point must not leave the bone standing there pointing at something nobody
        with suppress(Exception):
            await _hide_point(session)
        await session.speaker.stop()
        await send(session.ws, type="error", message=errors.message(e))
        await send(session.ws, type="state", state="idle")


async def handle(session: Session, msg: dict) -> None:
    ws = session.ws
    kind = msg.get("type")
    if meetings.manager.active and kind in {"ptt_start", "ptt_end", "text"}:
        await send(ws, type="error", message="Meeting transcription is active. Stop the meeting before talking to Mellow.")
        return

    if kind == "ping":
        await send(ws, type="pong", echo=msg.get("text", ""))

    elif kind == "capture_ready":
        # The shell has taken Mellow's windows out of the frame
        session.hidden.set()

    elif kind == "awake":
        # Waking starts the keeper (see WARM_RETRY_SECONDS)
        if msg.get("value"):
            session.wake_mic()
        else:
            session.awake = False
            session.mic_ready = False
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
        # Barge-in: talking over the user is the worst failure mode a pet has.
        await session.abort()
        session.turn_monitor = None
        # A pet with no brain cannot listen: the recorder never opens
        if standby():
            await send(ws, type="reply_chunk", text=PET_ONLY_LINE)
            await send(ws, type="state", state="idle")
            return
        # The normal frontend prevents this press
        if not session.mic_ready:
            await session._send_mic("warming")
            return
        # Off the loop: the first press after a wake does the full device open now
        await asyncio.to_thread(session.recorder.start)
        await send(ws, type="state", state="listening")

    elif kind == "ptt_end":
        # A release can race a press rejected during warm-up (or arrive from an older renderer).
        if not session.recorder.active:
            return
        session.turn_monitor = capture.known_monitor(msg.get("monitor"))
        if msg.get("monitor") is not None and session.turn_monitor is None:
            log.warning("ignored an invalid cursor monitor on ptt_end")
        audio = session.recorder.stop()
        await send(ws, type="state", state="thinking")
        # Blocking C call — off the event loop or the socket stalls.
        text = await asyncio.to_thread(stt.transcribe, audio)
        if meetings.manager.active:
            return
        # Show *something* when nothing was heard.
        if not text and session.recorder.last_stats["peak"] < stt.MIN_PEAK:
            await asyncio.to_thread(session.recorder.reopen)
            text_shown = "that was too quiet — say it again"
        else:
            text_shown = text or "…didn't catch that"
        await send(ws, type="transcript", text=text_shown)
        session.turn = asyncio.create_task(run_turn(session, text))

    elif kind == "text":
        await session.abort()
        session.turn_monitor = capture.known_monitor(msg.get("monitor"))
        if msg.get("monitor") is not None and session.turn_monitor is None:
            log.warning("ignored an invalid cursor monitor on text submission")
        # Same answer as the hotkey: no brain, one polite line, no turn.
        if standby():
            await send(ws, type="reply_chunk", text=PET_ONLY_LINE)
            await send(ws, type="state", state="idle")
            return
        await send(ws, type="state", state="thinking")
        session.turn = asyncio.create_task(
            run_turn(session, msg.get("text", "").strip())
        )

    elif kind == "cancel":
        await session.abort()
        session.recorder.stop()
        await send(ws, type="state", state="idle")

    elif kind == "new_conversation":
        # Both halves, or neither is worth doing
        await session.abort()
        session.history.clear()
        session.destination = None
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
        # The shell labels its menu "Mute" or "Unmute" from this, so it has to know before the user
        await send(ws, type="speak", value=config.load()["tts"]["speak"])
    except WebSocketDisconnect:
        # The dev webview connects twice at launch and abandons one instantly
        log.info("shell left during the greeting")
        return
    session = Session(ws)
    connected_revision = _engine_revision
    # The log already has this conversation
    session.history, session.destination = await asyncio.to_thread(sessions.resume)
    # A config save can land while resume() is reading from disk.
    if connected_revision != _engine_revision:
        session.history.clear()
        session.destination = None
    _active_sessions[id(session)] = session
    if session.history:
        log.info("resumed %d message(s) from the open session", len(session.history))
    # Reminders are not tied to the microphone or to being awake
    session.watch_reminders()

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            try:
                await handle(session, msg)
            except Exception as e:
                # One guard for every handler
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
        session.awake = False  # a keeper in flight stops at its next probe
        session.mic_ready = False
        session.alive = False  # and the reminder tick stops at its next wake-up
        session.recorder.close()
        # Closing the window must kill the audio
        await session.abort()


def run() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
