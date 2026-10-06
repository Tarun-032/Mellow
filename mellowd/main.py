"""Mellow sidecar: the AI half of the app."""

import asyncio
import json
import logging
import re
import sys
import threading
import time
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import NamedTuple

import sounddevice as sd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from mellowd import (
    act, agents, capture, config, drawing, errors, guide, llm, locator, meetings, memory, perf, point, remind, sessions, stt, transport, tts, visual_explanation, writing,
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
WARM_FAST_DELAYS = (0.1, 0.25, 0.5)

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
    try:
        await asyncio.to_thread(memory.sweep_ledger)
    except Exception:
        log.exception("memory could not be prepared")
    task = asyncio.create_task(warm_models())
    memory.learner.start(busy=_memory_busy)
    async with transport.lifespan():
        try:
            yield
        finally:
            await memory.learner.stop()
            await meetings.manager.shutdown()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await agents.stop()
            point.stop_ocr()


async def warm_models() -> None:
    """Warm the speech engines at boot."""
    # Skip pet-only mode.
    if standby():
        log.info("no brain configured; skipping model warm-up")
        return
    cfg = config.load()
    point.warm_ocr()
    agent_task = None
    if cfg["llm"].get("mode") == "agent":
        agent_task = asyncio.create_task(agents.warm(cfg))
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
    if agent_task is not None:
        try:
            await agent_task
        except Exception:
            log.exception("agent runtime warm-up failed")


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
    if name == "llm":
        # One set of active transport fields serves both API and agent modes.
        # Preserve each non-secret destination separately so changing modes
        # cannot combine `mode=cloud` with provider `codex` or `claude`.
        current_mode = current.get("mode")
        if current_mode == "cloud":
            merged.update(
                api_provider=current.get("provider", ""),
                api_base_url=current.get("base_url", ""),
                api_model=current.get("model", ""),
            )
        elif current_mode == "agent":
            merged.update(
                agent_provider=current.get("provider", ""),
                agent_model=current.get("model", ""),
            )

        mode = merged.get("mode")
        provider = str(merged.get("provider") or "").strip().lower()
        if mode == "cloud" and (
            provider not in config.LLM_PRESETS
            or config.LLM_PRESETS[provider].get("local")
        ):
            remembered = str(merged.get("api_provider") or "").strip().lower()
            if remembered in config.LLM_PRESETS and not config.LLM_PRESETS[remembered].get("local"):
                provider = remembered
                merged["base_url"] = merged.get("api_base_url") or merged.get("base_url")
                merged["model"] = merged.get("api_model") or merged.get("model")
            else:
                base = str(merged.get("base_url") or "").strip()
                provider = next(
                    (
                        key for key, preset in config.LLM_PRESETS.items()
                        if base
                        and not preset.get("local")
                        and preset.get("base_url")
                        and config.normalize_base_url(base) == config.normalize_base_url(preset["base_url"])
                    ),
                    "custom" if base else "openai",
                )
            merged["provider"] = provider
        if mode == "local" and (
            provider not in config.LLM_PRESETS
            or not config.LLM_PRESETS[provider].get("local")
        ):
            merged.update(
                provider="ollama",
                base_url=config.LLM_PRESETS["ollama"]["base_url"],
            )
        if mode == "agent" and provider not in config.AGENT_PRESETS:
            remembered = str(merged.get("agent_provider") or "").strip().lower()
            merged["provider"] = remembered if remembered in config.AGENT_PRESETS else "claude"
            merged["model"] = merged.get("agent_model") or ""

        if mode == "cloud":
            merged.update(
                api_provider=merged.get("provider", ""),
                api_base_url=merged.get("base_url", ""),
                api_model=merged.get("model", ""),
            )
        elif mode == "agent":
            merged.update(
                agent_provider=merged.get("provider", ""),
                agent_model=merged.get("model", ""),
            )
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
    # Owned by PUT /memory/settings. A settings form loaded before the Memory
    # page changed it would otherwise quietly switch it back.
    submitted.pop("memory_enabled", None)
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
            str(section.get("agent_speed", "fast")),
            str(cfg.get("system_prompt", "")),
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
    if previous.get("drawing_enabled") and not cfg.get("drawing_enabled"):
        for session in list(_active_sessions.values()):
            if getattr(session, "drawing_active", False) or getattr(session, "drawing_scene", None):
                await session.abort()
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
    # Summaries came from those sessions, and a digest in flight must not
    # recreate them. Saved notes stay; they are cleared in Memory.
    await asyncio.to_thread(memory.on_history_cleared)
    return {"cleared": n}


def _turn_running(exclude: asyncio.Task | None = None) -> bool:
    return any(
        s.turn is not None and not s.turn.done() and s.turn is not exclude
        for s in list(_active_sessions.values())
    )


def _memory_busy() -> bool:
    """Background learning waits for this: a live turn or a meeting always wins."""
    return _turn_running() or meetings.manager.active


async def _reset_conversations(reason: str) -> None:
    """End the conversation in progress everywhere, like New conversation."""
    for session in list(_active_sessions.values()):
        try:
            await session.abort()
        except Exception:
            log.exception("could not stop a session while resetting (%s)", reason)
        finally:
            session.history.clear()
            session.destination = None
        with suppress(WebSocketDisconnect, RuntimeError):
            await send(session.ws, type="state", state="idle")
    await asyncio.to_thread(sessions.close, reason=reason)


@app.post("/history/delete")
async def delete_history(body: dict):
    """Delete the chosen conversations, screenshots included."""
    ids = body.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise HTTPException(status_code=400, detail="ids must be a list of session ids")
    current = sessions.current()
    gone = await asyncio.to_thread(sessions.delete, ids)
    if current and current in gone:
        # Deleting the conversation in progress also takes it out of Mellow's context.
        await _reset_conversations("deleted")
    # Their summaries go too, and a digest in flight over them is discarded.
    # Saved memories stay; they are managed in Personalization.
    await asyncio.to_thread(memory.on_sessions_deleted, gone)
    return {"deleted": gone}


def _memory_view() -> dict:
    today = memory.usage_today()
    return {
        **memory.view(),
        "enabled": memory.enabled(),
        "learning_queued": today["pending"],
        "remember_conversations": bool(config.load().get("remember_conversations", True)),
    }


@app.get("/memory")
async def get_memory():
    return await asyncio.to_thread(_memory_view)


@app.put("/memory/document")
async def put_memory_document(body: dict):
    """The user's own edit of their memory summary."""
    try:
        await asyncio.to_thread(memory.save_document, body.get("sections"))
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return await asyncio.to_thread(_memory_view)



@app.put("/memory/settings")
async def put_memory_settings(body: dict):
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be true or false")
    cfg = config.load()
    if cfg.get("memory_enabled") != enabled:
        cfg["memory_enabled"] = enabled
        await asyncio.to_thread(config.save, cfg)
        await asyncio.to_thread(memory.set_enabled_changed, enabled)
        if not enabled:
            memory.learner.turn_started()  # stops a digest in flight; the rev bump discards it
        memory.learner.nudge()
    return await asyncio.to_thread(_memory_view)


@app.post("/memory/clear")
async def clear_memory(body: dict | None = None):
    """Forget everything; with {"disable": true}, also switch memory off."""
    await asyncio.to_thread(memory.clear_all)
    if (body or {}).get("disable"):
        cfg = config.load()
        cfg["memory_enabled"] = False
        await asyncio.to_thread(config.save, cfg)
        memory.learner.turn_started()
    await _reset_conversations("memory_cleared")
    return await asyncio.to_thread(_memory_view)



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
    guide_pending: guide.Pending | None = None
    guide_task: asyncio.Task | None = None
    presentation: guide.Presentation | None = None
    drawing_presentation: guide.Presentation | None = None
    drawing_task: asyncio.Task | None = None
    drawing_scene: dict | None = None
    drawing_occlusions: list = field(default_factory=list)
    drawing_pen_occlusions: list = field(default_factory=list)
    drawing_deadline: float = 0
    drawing_completion_deadline: float = 0
    drawing_observed_regions: tuple = ()
    drawing_diagnostic: object = None
    drawing_animations: tuple = ()
    drawing_complete: asyncio.Event = field(default_factory=asyncio.Event)
    drawing_failure_reason: str | None = None
    drawing_frame: drawing.Frame | None = None
    drawing_active: bool = False
    drawing_pen_active: bool = False
    drawing_scale: float = 1
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
    capture_id: str | None = None
    capture_safe: bool = False
    capture_active: bool = False
    capture_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
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
        last_error: Exception | None = None

        def opened(label: str) -> bool:
            if not self.awake:
                # Sleep may win while PortAudio is inside stream.start().
                self.recorder.close(immediate=True)
                return False
            log.info(
                "microphone %s in %.0fms",
                label,
                (time.monotonic() - started) * 1000,
            )
            return True

        # Normal sleep wake-up: reopen only the last working microphone. If
        # PortAudio kept stale state, refresh once and retry the current primary
        # before falling back to the exhaustive compatibility path.
        try:
            route = self.recorder.open_preferred(quiet=True)
        except Exception as exc:
            last_error = exc
        else:
            return opened(f"{route} reopen")

        if self.awake and not standby() and not meetings.manager.active:
            refreshed = stt.refresh_devices()
            if refreshed:
                log.info("refreshed portaudio device list after fast reopen failed")
            try:
                route = self.recorder.open_preferred(
                    quiet=True, prefer_cached=not refreshed
                )
            except Exception as exc:
                last_error = exc
            else:
                return opened("refreshed reopen")

        for delay in WARM_FAST_DELAYS:
            if not self.awake:
                return False
            time.sleep(delay)
            try:
                route = self.recorder.open_preferred(
                    quiet=True, prefer_cached=False
                )
            except Exception as exc:
                last_error = exc
                continue
            return opened("short-retry reopen")

        log.info("fast microphone reopen failed (%s); using device fallback", last_error)
        while self.awake:
            # Recheck config while probing.
            if standby() or meetings.manager.active:
                return False
            try:
                # Retry scheduling belongs here; do not multiply it by the
                # recorder's general-purpose retry loop.
                self.recorder.open(quiet=True, attempts=1)
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
                if not refreshed and stt.refresh_devices():
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
                self.recorder.close(immediate=True)
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

    async def abort(self, *, preserve_guide: bool = False) -> None:
        """Stop whatever the pet is doing, right now."""
        _retire_point(self)
        await _clear_drawing(self)
        self.writer.cancel()
        if self.turn and not self.turn.done():
            self.turn.cancel()
        await _stop_guide(self, preserve=preserve_guide)
        await self.cancel_ptt_start()
        self.recorder.stop()
        await self.stop_meter()
        if self.turn and not self.turn.done():
            self.turn.cancel()
            with suppress(asyncio.CancelledError):
                await self.turn
        self.turn = None
        await self.speaker.stop()
        if self.alive:
            await _hide_point(self)


# Per-shell histories.
_active_sessions: dict[int, Session] = {}
_engine_revision = 0

async def _meeting_started():
    await agents.stop()
    point.stop_ocr()
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
    cfg = config.load()
    if cfg.get("ai_enabled"):
        point.warm_ocr()
    if cfg.get("ai_enabled") and cfg["llm"].get("mode") == "agent":
        asyncio.create_task(agents.warm(cfg))
    for session in list(_active_sessions.values()):
        if session.alive and not meetings.manager.active:
            session.wake_mic()


meetings.manager.before_start = _meeting_started
meetings.manager.after_stop = _meeting_stopped


async def _reset_for_engine_change() -> None:
    """End the current conversation after a committed engine change."""
    global _engine_revision
    _engine_revision += 1
    await agents.stop()
    point.stop_ocr()
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
    cfg = config.load()
    if cfg.get("ai_enabled") and cfg["llm"].get("mode") == "agent":
        asyncio.create_task(agents.warm(cfg))
    if cfg.get("ai_enabled"):
        point.warm_ocr()
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
    # Physical source-window bounds; optional for old fixtures/headless runs.
    window: tuple[int, int, int, int] | None = None
    frame: drawing.Frame | None = None


def _shot(
    max_edge: int = capture.MAX_EDGE, monitor: dict | None = None
) -> tuple[Shot | None, str, str]:
    """Capture + audit metadata in one blocking call. Never raises."""
    monitor = capture.known_monitor(monitor) or capture.active_monitor()
    hwnd, app, title = capture.window_on_monitor(monitor) if monitor else (0, "", "")
    captured_at = time.monotonic()
    grabbed = capture.grab(max_edge, monitor)
    window = capture.window_rect(hwnd)
    shot = (
        Shot(*grabbed, monitor, hwnd, window,
             drawing.Frame.capture(monitor, hwnd, window, grabbed[1], grabbed[2], grabbed[3], captured=captured_at))
        if grabbed and monitor else None
    )
    if not app and not title:
        app, title = capture.foreground()
    return shot, app, title


# Capture-hide timeout.
HIDE_TIMEOUT = 1.0


async def _prepare_capture(session):
    # A receipt belongs to this capture, not a cancelled/previous one.
    import uuid
    session.capture_id = uuid.uuid4().hex
    session.capture_safe = False
    session.capture_active = True
    session.hidden.clear()
    await send(session.ws, type="capture", phase="begin", capture_id=session.capture_id)
    try:
        await asyncio.wait_for(session.hidden.wait(), HIDE_TIMEOUT)
    except TimeoutError:
        log.warning("shell did not confirm capture preparation")
        return False
    return session.capture_safe


@perf.timed("capture")
async def _unseen_shot(
    session: Session, max_edge: int = capture.MAX_EDGE
) -> tuple[Shot | None, str, str]:
    """`_shot`, with Mellow's own windows out of the picture."""
    result = await _clean_capture(session, lambda: _shot(max_edge, getattr(session, "turn_monitor", None)))
    return result if result is not None else (None, "", "")


async def _clean_capture(session, sample):
    """Serialize clean local samples; restore the matching receipt even on cancel."""
    if not hasattr(session, "capture_lock"):
        session.capture_lock = asyncio.Lock()
    async with session.capture_lock:
        try:
            if not await _prepare_capture(session):
                return None
            return await asyncio.to_thread(sample)
        finally:
            capture_id = session.capture_id
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(send(session.ws, type="capture", phase="end", capture_id=capture_id))
            session.capture_active = False


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
    presentation: guide.Presentation | None = None,
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
    pointer_failed = False

    async def before_speech():
        nonlocal pointer_failed
        if presentation is not None and not await presentation.wait():
            pointer_failed = True
            return False
        return True

    async def queue(sentence):
        if presentation is None:
            await session.speaker.speak(sentence)
        else:
            await session.speaker.speak_beat(sentence, before_speech)

    if not speak and presentation is not None and not await presentation.wait():
        await _stop_guide(session)
        await _hide_point(session)
        return await _deliver(session, _POINTER_UNAVAILABLE, False, partial), False, None

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
                await queue(sentence)

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
            await queue(sentence)
        if presentation is not None:
            await _played(session)
            if pointer_failed:
                await _stop_guide(session)
                await _hide_point(session)
                reply += await _deliver(session, _POINTER_UNAVAILABLE, True, partial)
    return reply, False, point


async def _deliver(
    session: Session, text: str, speak: bool, partial: dict | None = None
) -> str:
    """Deliver an already-grounded answer without another model call.

    Beats accumulate in the bubble, so it ends holding the whole explanation as
    one paragraph.
    """
    reply = text.strip()
    if not reply:
        reply = "I couldn't lock onto a safe target on this screen."
    if partial is not None:
        # Beats of one narration accumulate, and neither useSocket nor `partial`
        # inserts anything between them. Without this a cancelled multi-beat
        # turn was recorded as "...your media.Then drag...".
        if partial["text"]:
            reply = " " + reply
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


# Said when the screen moved out from under a later bone. This one stays: it
# reports a real failure, which is not a voice decision. The old _CLOSING is
# gone - a stock sign-off is exactly what llm.CORE forbids, and a handover that
# belongs is written by the model as a narration beat.
_CUT_SHORT = "That's moved, so I'll stop there rather than point at the wrong thing."

_POINTER_UNAVAILABLE = "I couldn't get the pointer into place, so I'll stop pointing there."


def _dwell(sentence: str) -> float:
    """How long a step holds when there is no voice to pace against.

    ponytail: flat reading speed, no per-word timing. Raise it if muted users
    report the bone outrunning them.
    """
    return min(6.0, 1.0 + len(sentence.split()) / 3)


async def _played(session: Session) -> None:
    """Wait for the queued speech to be heard, and let a barge-in through.

    `Speaker.finish` suppresses CancelledError while it waits on the playback
    task, so a cancel arriving mid-step is swallowed and the loop would go on to
    show the next bone after the user had already interrupted. The suppression
    leaves the request recorded on this task, so ask.
    """
    await session.speaker.finish()
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError
    # finish() cleared the speaker's task; without this the next sentence would
    # be queued for a consumer that no longer exists and never be heard.
    session.speaker.begin()


async def _still_there(session: Session, shot, target: point.Target):
    """Re-verify a later step on a fresh frame. Local pixels only, no model call.

    Steps are chosen together on one frame, but a later bone is shown seconds
    after it - long enough for the app to repaint or the user to start clicking.
    """
    fresh, _, _ = await _unseen_shot(session, capture.POINT_EDGE)
    if fresh is None:
        perf.mark("step_unverified")
        return None, shot
    if not locator.changed_at(shot, fresh, target):
        return target, fresh
    moved = await asyncio.to_thread(locator.relocate_target, shot, fresh, target)
    perf.mark("step_relocated" if moved is not None else "step_lost")
    if moved is None:
        log.info("step %r is no longer where it was measured; ending the sequence", target.label)
    return moved, fresh


async def _narrate(
    session: Session, beats: list[tuple[point.Target | None, str]], speak: bool,
    partial: dict, shot,
) -> str:
    """Speak one narration, moving the bone as each beat that has one comes up.

    Speech shares one synthesis/playback pipeline across every beat. Each beat
    verifies its target and emits its text at playback, never during lookahead.
    """
    said = []
    cut = False
    shown = 0      # bones actually put on screen: what latency.md counts
    up = True      # answer() already showed and verified the first bone
    spoke_bone = False
    presentation = getattr(session, "presentation", None)
    obstacle = _CUT_SHORT
    async def present(target, sentence):
        nonlocal shown, up, spoke_bone, shot, cut, presentation, obstacle
        if target is None:
            # Narration. Take the bone away first, or unrelated prose is read
            # beside a control it is no longer about. Before the first bone
            # there is nothing to hide, and the bone answer() dispatched is
            # already flying to the control this beat is leading up to.
            if up and spoke_bone and not (
                (pending := getattr(session, "guide_pending", None)) and pending.observer
            ):
                await _hide_point(session)
                up = False
        else:
            shown += 1
            if shown > 1 or not up:
                target, shot = await _still_there(session, shot, target)
                if target is None:
                    cut = True
                    return False
                presentation = await _aim(session, target)
                # _aim's own mark keeps the first dispatch, which latency.md
                # keys on; number the rest by bone, never by beat.
                perf.mark(f"pointer_dispatched_{shown}")
                up = True
            if presentation is None or not await presentation.wait():
                cut = True
                obstacle = _POINTER_UNAVAILABLE
                return False
            spoke_bone = True
            await _arm_guide(session, target, shot)
        said.append(await _deliver(session, sentence, False, partial))
        return True

    if speak:
        # Queue the full narration so LOOKAHEAD can synthesize later clauses
        # while this one is playing. The callback gates both pointer and text;
        # a changed target or cancellation cannot release prefetched speech.
        for target, sentence in beats:
            async def before(target=target, sentence=sentence):
                return await present(target, sentence)
            await session.speaker.speak_beat(sentence, before)
        await _played(session)
    else:
        for index, (target, sentence) in enumerate(beats):
            if not await present(target, sentence):
                break
            # The frontend gives the final silent reply its existing reading
            # timeout; dwell here only while another pointer beat is waiting.
            if index < len(beats) - 1:
                await asyncio.sleep(_dwell(sentence))
    if cut:
        await _stop_guide(session)
        await _hide_point(session)
        said.append(await _deliver(session, obstacle, speak, partial))
    # A narration that ends on a bone leaves it up; the turn's `idle` retires it.
    return "".join(said)


async def _stop_guide(session, *, preserve=False):
    pending = getattr(session, "guide_pending", None)
    current = asyncio.current_task()
    tasks = [getattr(session, "guide_task", None), pending.observer if pending else None]
    session.guide_task = None
    if not preserve:
        session.guide_pending = None
    for task in tasks:
        if task and task is not current and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    if pending:
        pending.observer = None
        await _hide_point(session)
        if getattr(session, "alive", True):
            await send(session.ws, type="guide", waiting=False)


async def _arm_guide(session, target, shot):
    pending = getattr(session, "guide_pending", None)
    if pending is None or pending.observer is not None or not point._same_place(target, pending.target):
        return
    pending.target, pending.shot = target, shot
    pending.observer = asyncio.create_task(guide.interaction(
        target, shot.hwnd, pending.spec.trigger,
        max(0, pending.deadline - time.monotonic()),
    ))
    await send(session.ws, type="guide", waiting=True)


def _pending_guide(session, goal, result, beats, shot, partial, previous=None):
    bones = [target for target, _ in beats if target is not None]
    spec = result.continuation
    if spec is None or spec.after_bone != len(bones) or not bones:
        session.guide_pending = None
        return None
    pending = guide.Pending(goal, spec, bones[-1], shot, partial,
                            segments=previous.segments + 1 if previous else 1,
                            explained=[*previous.explained, previous.target.label] if previous else [])
    session.guide_pending = pending
    return pending


async def _guide_handover(session, pending):
    cfg = config.load()
    speak = cfg["tts"]["speak"]
    if speak:
        session.speaker.begin()
    line = await _deliver(session, pending.spec.handover, speak, pending.partial)
    if speak:
        await session.speaker.finish()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
    session.history.append({"role": "assistant", "content": line.strip()})
    await asyncio.to_thread(sessions.record, "assistant_said", **_said(cfg, line.strip(), False))


async def _guide_loop(session, *, force=False):
    """One bounded observation per hidden-control boundary, never a click agent."""
    try:
        while (pending := session.guide_pending) is not None:
            await pending.ready.wait()
            if pending.segments >= guide.MAX_SEGMENTS:
                await _guide_handover(session, pending)
                break
            if not force:
                if pending.observer is None:
                    await _arm_guide(session, pending.target, pending.shot)
                interacted = await pending.observer
                pending.observer = None
                if not interacted:
                    await _guide_handover(session, pending)
                    break
            force = False
            fresh = None
            settling = None
            # Give menus and dialogs time to settle, without an LLM polling loop.
            for _ in range(10):
                await asyncio.sleep(.25)
                candidate, _, _ = await _unseen_shot(session, capture.POINT_EDGE)
                if candidate is not None and guide.changed(pending.shot, candidate):
                    if settling is not None and not guide.changed(settling, candidate):
                        fresh = candidate
                        break
                    settling = candidate
                else:
                    settling = None
            if fresh is None:
                if time.monotonic() >= pending.deadline:
                    await _guide_handover(session, pending)
                    break
                # A spoken 'next' may resume without the menu having opened.
                # Put the verified pending target back before observing again.
                target, shot = await _still_there(session, pending.shot, pending.target)
                if target is None:
                    await _guide_handover(session, pending)
                    break
                pending.target, pending.shot = target, shot
                presentation = await _aim(session, target)
                if not await presentation.wait():
                    await _guide_handover(session, pending)
                    break
                await _arm_guide(session, pending.target, pending.shot)
                await send(session.ws, type="state", state="idle")
                continue
            await send(session.ws, type="guide", waiting=False)
            await _hide_point(session)
            await send(session.ws, type="state", state="thinking")
            query = (
                f"Continue guiding this goal: {pending.goal}\n"
                f"Still unresolved: {pending.spec.remaining}\n"
                f"Expected UI change to VERIFY now: {pending.spec.expected_change}\n"
                f"The user interacted with {pending.target.label!r}; that alone does not prove completion.\n"
                f"Controls already explained, not necessarily completed: {[*pending.explained, pending.target.label]}\n"
                f"Already spoken (do not repeat): {pending.partial['text'][-1200:]}\n"
                "Use only the fresh screen. If the expected UI is not open, choose none and explain why. "
                "Otherwise continue the same explanation with the newly visible controls. "
                "Answer only the still-unresolved part. The original goal is context, "
                "not a request to start over. Do not repeat the introduction or reopen "
                "a menu or form that is already open. Point individually to the newly "
                "visible requested controls; do not replace their beats with a verbal summary."
            )
            cfg = config.load()
            cands = await asyncio.to_thread(point.candidates, pending.spec.remaining, fresh.pixels, fresh.monitor, None, fresh.hwnd)
            result = await locator.locate_and_answer(query, fresh, cfg, cands, session.history)
            target = result.target
            if target is not None:
                target, fresh = await _still_there(session, fresh, target)
            beats = locator.narration(result, target, result.answer)
            speak = cfg["tts"]["speak"]
            if speak:
                session.speaker.begin()
            if not beats:
                session.guide_pending = None
                line = result.answer if target is None and result.target is None and result.answer else pending.spec.handover
                reply = await _deliver(session, line, speak, pending.partial)
            else:
                next_pending = _pending_guide(session, pending.goal, result, beats, fresh, pending.partial, pending)
                await _aim(session, target)
                reply = await _narrate(session, beats, speak, pending.partial, fresh)
                if next_pending and session.guide_pending is next_pending:
                    next_pending.ready.set()
            if speak:
                await session.speaker.finish()
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
            session.history.append({"role": "assistant", "content": reply.strip()})
            del session.history[:max(0, len(session.history) - HISTORY_TURNS * 2)]
            await asyncio.to_thread(sessions.record, "assistant_said", **_said(cfg, reply.strip(), False))
            await send(session.ws, type="state", state="idle")
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("walkthrough stopped")
        if pending := session.guide_pending:
            await _guide_handover(session, pending)
    finally:
        # A newer turn can own a different guide; never clear its pointer.
        if getattr(session, "guide_task", None) is asyncio.current_task():
            await _stop_guide(session)
            await send(session.ws, type="guide", waiting=False)
            await send(session.ws, type="state", state="idle")


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


def _retire_point(session: Session) -> None:
    if presentation := getattr(session, "presentation", None):
        presentation.retire()
    session.presentation = None


async def _clear_drawing(session, *, preserve_pen=False) -> None:
    owned_pen = getattr(session, "drawing_pen_active", False)
    if not preserve_pen: session.drawing_pen_active = False
    presentation = getattr(session, "drawing_presentation", None)
    if presentation:
        presentation.retire()
    session.drawing_presentation = None
    if complete := getattr(session, "drawing_complete", None):
        complete.set()
    watcher = getattr(session, "drawing_task", None)
    session.drawing_task = None
    if watcher and watcher is not asyncio.current_task():
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher
    scene = getattr(session, "drawing_scene", None)
    session.drawing_scene = None
    session.drawing_occlusions = []
    session.drawing_pen_occlusions = []
    session.drawing_deadline = 0
    session.drawing_completion_deadline = 0
    session.drawing_animations = ()
    session.drawing_observed_regions = ()
    session.drawing_scale = 1
    if (scene or owned_pen) and getattr(session, "alive", True):
        await send(session.ws, type="drawing", scene=None, preserve_pen=preserve_pen)


async def _draw(session, shot: Shot, marks: list[dict], reveal_from=0, caption="", *, animations=(), observed_regions=()) -> guide.Presentation:
    """Present validated geometry; called at playback, never at synthesis."""
    await _clear_drawing(session, preserve_pen=True)
    frame = shot.frame
    if frame is None or not frame.hwnd or frame.window is None:
        raise ValueError("drawing needs a captured source window")
    # Speech synthesis/planning can outlive the screen it describes. Recheck a
    # clean local capture before showing old geometry, not just after display.
    # A tooltip, hover card or caret may be gone a moment later: look twice.
    for attempt in range(2):
        fresh, _, _ = await _unseen_shot(session)
        if (fresh is None or fresh.hwnd != frame.hwnd or fresh.window != frame.window
                or fresh.monitor != frame.monitor):
            raise ValueError("drawing source changed since capture")
        # Renew only AFTER the original source has matched a clean fresh capture.
        # Keep its crop transform: the model's coordinates belong to that image.
        renewed = replace(fresh.frame, transform=frame.transform)
        scene = renewed.scene(marks, reveal_from)
        changed = frame.content_changed(fresh.pixels, targets=_drawing_regions(scene, observed_regions),
                                        cursor=capture.cursor_position(), animations=animations)
        if not changed:
            break
        perf.visual_recorder()("preflight_changed", changed)
        if attempt:
            raise ValueError("drawing source changed since capture")
        await asyncio.sleep(.3)
    frame = renewed
    scene["caption"] = caption[:60]
    if capture.window_rect(frame.hwnd) != frame.window or capture.known_monitor(frame.monitor) != frame.monitor:
        raise ValueError("drawing source moved or disappeared")
    marker = perf.marker()
    presentation = guide.Presentation(mark=lambda name: marker(name.replace("pointer_", "drawing_")))
    scene["presentation_id"] = presentation.id
    session.drawing_presentation = presentation
    session.drawing_scene = scene
    session.drawing_pen_active = True
    session.drawing_frame = frame
    session.drawing_deadline = time.monotonic() + drawing.ACTIVE_LIFETIME
    session.drawing_diagnostic = perf.visual_recorder()
    session.drawing_animations = tuple(animations)
    session.drawing_observed_regions = tuple(observed_regions)
    session.drawing_complete = asyncio.Event()
    # Older shells do not report their animation clock. Use a bounded estimate
    # until the ready receipt supplies the actual remaining stroke time.
    shapes = sum(mark["kind"] != "label" for mark in marks[reveal_from:])
    session.drawing_completion_deadline = time.monotonic() + min(57.0, 2.85 + shapes * 2.1)
    session.drawing_failure_reason = None
    perf.mark("drawing_dispatched")
    await send(session.ws, type="drawing", scene=scene)
    return presentation


def _drawing_regions(scene, observed_regions=()):
    """Physical source targets, never masks hiding their underlying content."""
    # A short measured arrow watches the entire original control, not just
    # its shaft. The host keeps those bounds outside the model/native scene.
    regions = list(observed_regions)
    for mark in scene["marks"]:
        if mark["kind"] == "label":
            continue
        xs, ys = zip(*mark["points"])
        regions.append((min(xs), min(ys), max(1, max(xs)-min(xs)), max(1, max(ys)-min(ys))))
    return regions


async def _invalidate_drawing(session, reason="native_failure") -> None:
    """A changed source must also stop speech that still describes its marks."""
    # Capture ownership before clearing ink: cleanup yields to playback and may
    # overlap a fresh push-to-talk turn. Never reset that replacement's state.
    turn = getattr(session, "turn", None)
    interrupted = bool(turn and not turn.done() and turn is not asyncio.current_task())
    if record := getattr(session, "drawing_diagnostic", None):
        record("retired", reason)
    log.info("drawing retired: %s", reason)
    await _clear_drawing(session)
    if getattr(session, "turn", None) is not turn:
        return
    if interrupted:
        turn.cancel()
    await session.speaker.stop()
    if (interrupted and getattr(session, "turn", None) is turn
            and getattr(session, "alive", True)):
        # Successful answer() sends idle; its cancellation path cannot reach
        # that line. Also retire interrupted dialogue, including muted replies.
        await send(session.ws, type="state", state="idle", interrupted=True)


async def _watch_drawing(session, frame, scene):
    # Clean underlying source samples; no screenshot is uploaded or saved.
    # Input is not a cancellation event. Only persistent content/layout changes
    # matter, and annotations themselves must not hide changed target pixels.
    baseline = frame
    changed_samples = 0
    reason = "source_changed"
    try:
        while session.drawing_scene is scene:
            await asyncio.sleep(.5)
            if getattr(session, "capture_active", False):
                continue
            if time.monotonic() >= session.drawing_deadline:
                reason = "expired"
                break
            if capture.window_rect(frame.hwnd) != frame.window or capture.known_monitor(frame.monitor) != frame.monitor:
                reason = "display_changed"
                break
            sampled = await _clean_capture(session, lambda: capture.frame(frame.monitor))
            if sampled is None:
                reason = "capture_unavailable"
                break
            _, pixels, _ = sampled
            if session.drawing_scene is not scene:
                return
            changed = baseline.content_changed(pixels, targets=_drawing_regions(scene, getattr(session, "drawing_observed_regions", ())), cursor=capture.cursor_position(),
                                                animations=getattr(session, "drawing_animations", ()))
            changed_samples = changed_samples + 1 if changed else 0
            if changed_samples >= 2:
                break
        if session.drawing_scene is scene:
            await _invalidate_drawing(session, reason)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("drawing observation failed")
        if session.drawing_scene is scene:
            await _invalidate_drawing(session, "observer_error")


async def _begin_drawing_beat(session, shot, marks, sentence, *, reveal_from=0, animations=(), observed_regions=()):
    """Verify and show ink; return "ready" or why its readiness gate failed.

    The caller delivers the sentence when its audio starts. The first beat runs
    this while its speech is still being synthesized, so the bone is already
    drawing by the time Mellow begins to talk.
    """
    try:
        caption = next((m["text"] for m in marks[reveal_from:] if m["kind"] == "label"), "")
        if not caption:
            caption = sentence.split(".", 1)[0][:60]
        ticket = await _draw(session, shot, marks, reveal_from, caption, animations=animations, observed_regions=observed_regions)
        if not await ticket.wait():
            reason = getattr(session, "drawing_failure_reason", None)
            await _clear_drawing(session)
            return "renderer_failed" if reason in {None, "presentation_failed", "native_failure"} else "withheld"
        ticket.mark("drawing_playback_ready")
        scene = session.drawing_scene
        session.drawing_task = asyncio.create_task(_watch_drawing(session, session.drawing_frame, scene))
        return "ready"
    except ValueError as error:
        reason = "source_changed" if str(error) in {
            "drawing source changed since capture", "drawing source moved or disappeared"
        } else "invalid_scene"
        perf.visual_recorder()("preflight_rejected", reason)
        perf.mark("drawing_preflight_rejected_" + reason)
        log.info("drawing preflight withheld: %s", reason)
        await _clear_drawing(session)
        return reason if reason == "source_changed" else "withheld"


# Never end in silence: an explanation still answers the question without ink.
_SCREEN_MOVED = "The screen changed before I could mark it, so here it is in words. "


async def _drawing_stroke_done(session):
    """Use the renderer's bounded clock, while still detecting lost receipts."""
    ticket = session.drawing_presentation
    if ticket is None or ticket.retired:
        raise asyncio.CancelledError
    remaining = max(.5, getattr(session, "drawing_completion_deadline", time.monotonic() + 6) - time.monotonic())
    try:
        await asyncio.wait_for(session.drawing_complete.wait(), remaining)
    except TimeoutError:
        perf.outcome("drawing_failed")
        perf.visual_recorder()("completion_timeout", "renderer_unresponsive")
        await _clear_drawing(session)
        return False
    if ticket.retired or session.drawing_presentation is not ticket:
        raise asyncio.CancelledError
    return True


async def _drawing_beat(session, shot: Shot, marks, sentence, speak, partial, *, reveal_from=0, wait_complete=False, animations=(), fallback_target=None, observed_regions=(), early=None, first=False):
    """Visible ink gates speech; the final stroke also gates the next beat.

    `early` is this beat's _begin_drawing_beat task, started during synthesis.
    Only a `first` beat (nothing shown yet) falls back to words on a screen change.
    """
    delivered = ""
    readiness = ""

    async def before():
        nonlocal delivered, readiness
        readiness = await (early if early is not None else _begin_drawing_beat(
            session, shot, marks, sentence, reveal_from=reveal_from, animations=animations,
            observed_regions=observed_regions))
        if readiness == "ready":
            await _deliver(session, sentence, False, partial)
            delivered = sentence
        return readiness == "ready"

    if speak:
        await session.speaker.speak_beat(sentence, before)
        await _played(session)
    else:
        await before()
    if not delivered:
        if readiness == "renderer_failed" and fallback_target is not None:
            return await _visual_control_point(session, shot, fallback_target, sentence, speak, partial)
        perf.outcome("drawing_failed")
        if readiness == "source_changed" and first:
            return await _deliver(session, _SCREEN_MOVED + sentence, speak, partial)
        return await _deliver(session, "I couldn't place the drawing reliably, so I'll stop there.", speak, partial)
    if wait_complete:
        if not await _drawing_stroke_done(session):
            return delivered + " " + await _deliver(session, "The drawing stopped responding, so I'll stop here.", speak, partial)
        if not speak:
            # Muted explanations remain visible long enough to read.
            await asyncio.sleep(min(6.0, max(1.2, len(sentence) / 28)))
    return delivered


async def _visual_reasoning_fresh(session, shot, animations):
    """Reasoning without new ink still needs current screen evidence."""
    fresh, _, _ = await _unseen_shot(session)
    frame = shot.frame
    scene = session.drawing_scene
    targets = _drawing_regions(scene, getattr(session, "drawing_observed_regions", ())) if scene is not None else ()
    return (fresh is not None and fresh.hwnd == frame.hwnd and fresh.window == frame.window
            and fresh.monitor == frame.monitor
            and not frame.content_changed(fresh.pixels, targets=targets,
                                          cursor=capture.cursor_position(), animations=animations))


async def _visual_sequence(session, plan, partial, *, animations=(), early=None):
    """Prebuffer narration across beats; all screen changes happen at playback.

    Audio lookahead overlaps synthesis with the preceding explanation, while
    the readiness and completion gates keep the bone and speech in order.
    `early` is the first beat's _begin_drawing_beat task, started during synthesis.
    """
    reply = []
    failure = ""

    async def present(beat):
        nonlocal failure, early
        readiness = "ready"
        if early is not None:
            readiness, early = await early, None
        else:
            if session.drawing_scene is not None and not await _drawing_stroke_done(session):
                failure = "The drawing stopped responding, so I'll stop here."
                return False
            new_ink = beat.marks and (beat.reveal_from != len(beat.marks) or session.drawing_scene is None)
            if not new_ink:
                # Retained reasoning and a plain conclusion still refer to the
                # captured page. Do not let audio lookahead bypass its freshness
                # check merely because this beat adds no shapes.
                if not await _visual_reasoning_fresh(session, plan.shot, animations):
                    await _clear_drawing(session)
                    failure = "The screen changed before I could show that, so I'll stop here."
                    return False
            if not beat.marks:
                await _clear_drawing(session)
            elif beat.reveal_from == len(beat.marks) and session.drawing_scene is not None:
                session.drawing_deadline = time.monotonic() + drawing.ACTIVE_LIFETIME
                await send(session.ws, type="drawing_keepalive",
                           presentation_id=session.drawing_scene["presentation_id"])
            else:
                await send(session.ws, type="state", state="looking")
                readiness = await _begin_drawing_beat(session, plan.shot, beat.marks, beat.say,
                                                     reveal_from=beat.reveal_from, animations=animations,
                                                     observed_regions=getattr(beat, "observed_regions", ()))
        if readiness != "ready":
            # Nothing shown yet: the explanation still answers the question in words.
            failure = (_SCREEN_MOVED + " ".join(b.say for b in plan.beats)
                       if readiness == "source_changed" and not reply else
                       "I couldn't place the drawing reliably, so I'll stop there.")
            return False
        await _deliver(session, beat.say, False, partial)
        reply.append(beat.say)
        return True

    for beat in plan.beats:
        # Binding the beat here avoids the last-loop-value callback bug.
        await session.speaker.speak_beat(beat.say, lambda beat=beat: present(beat))
    await _played(session)
    if not failure and session.drawing_scene is not None and not await _drawing_stroke_done(session):
        failure = "The drawing stopped responding, so I'll stop here."
    if failure:
        perf.outcome("drawing_failed")
        reply.append(await _deliver(session, failure, True, partial))
        # Do not clear the last ink while a failure sentence is still playing.
        await _played(session)
    return " ".join(text.strip() for text in reply)


def _visual_images(shot, candidates):
    """Measured visible media, not guessed diagram boxes or control regions."""
    wx, wy, ww, wh = shot.window
    return tuple(row.bounds for row in candidates
                 if row.source == "uia" and getattr(row, "is_image", False)
                 and not row.chrome and row.bounds is not None
                 and wx <= row.bounds[0] < row.bounds[0] + row.bounds[2] <= wx + ww
                 and wy <= row.bounds[1] < row.bounds[1] + row.bounds[3] <= wy + wh)


async def _visual_sample(session, shot, samples):
    """Keep small local motion evidence; never upload or store these frames."""
    hwnd, _, _ = capture.window_on_monitor(shot.monitor)
    if (hwnd != shot.hwnd or capture.window_rect(shot.hwnd) != shot.window
            or capture.known_monitor(shot.monitor) != shot.monitor):
        return False
    sampled = await _clean_capture(session, lambda: capture.frame(shot.monitor))
    if sampled is None:
        return False
    _, pixels, _ = sampled
    hwnd, _, _ = capture.window_on_monitor(shot.monitor)
    if hwnd != shot.hwnd or capture.window_rect(shot.hwnd) != shot.window:
        return False
    samples.append(await asyncio.to_thread(drawing.observation, pixels))
    return True


async def _visual_motion_samples(session, shot, samples):
    # Observe while the existing planner runs; six small local samples are the
    # entire motion budget. A paused GIF may only begin moving during planning.
    while len(samples) < 6:
        await asyncio.sleep(.5)
        if not await _visual_sample(session, shot, samples):
            return False
    return True


async def _visual_point(session, plan, speak, partial, *, animations=()):
    """No ink survived validation: the ordinary arriving bone points at each beat's location."""
    beats = [(beat.pointer, beat.say) for beat in plan.beats]
    frame = plan.shot.frame
    fresh, _, _ = await _unseen_shot(session)
    if (fresh is None or fresh.hwnd != frame.hwnd or fresh.window != frame.window
            or fresh.monitor != frame.monitor
            or frame.content_changed(fresh.pixels, targets=[t.bounds for t, _ in beats if t is not None],
                                     cursor=capture.cursor_position(), animations=animations)):
        perf.outcome("drawing_failed")
        return await _deliver(session, "The screen changed before I could show that, so I'll stop here.", speak, partial)
    # _narrate expects the first bone already in flight, as answer() does.
    await _aim(session, next(target for target, _ in beats if target is not None))
    perf.mark("visual_presentation_point_fallback")
    return await _narrate(session, beats, speak, partial, fresh)


async def _visual_control_point(session, shot, target, sentence, speak, partial):
    """An ink failure can still point the bone at the beat's location.

    Never use this path after a source/layout change: the unchanged source is
    checked again at playback, after the failed renderer has been cleared.
    """
    delivered = ""
    obstacle = "The screen changed before I could show that, so I'll stop here."

    async def before():
        nonlocal delivered, obstacle
        # Check at playback, after slow synthesis, just like the drawing path.
        fresh, _, _ = await _unseen_shot(session)
        frame = shot.frame
        if (fresh is None or frame is None or fresh.hwnd != frame.hwnd
                or fresh.window != frame.window or fresh.monitor != frame.monitor
                or frame.content_changed(fresh.pixels, targets=[target.bounds], cursor=capture.cursor_position())):
            return False
        ticket = await _aim(session, target)
        if not await ticket.wait():
            await _hide_point(session)
            obstacle = _POINTER_UNAVAILABLE
            return False
        perf.mark("visual_control_point_fallback")
        delivered = await _deliver(session, "I'll point to it instead. " + sentence, False, partial)
        return True

    if speak:
        await session.speaker.speak_beat("I'll point to it instead. " + sentence, before)
        await _played(session)
    else:
        await before()
    if delivered:
        return delivered
    perf.outcome("drawing_failed")
    return await _deliver(session, obstacle, speak, partial)


async def _visual_answer(session, prompt, cfg, speak, partial):
    if not cfg.get("drawing_enabled"):
        return await _deliver(session, "Turn on screen drawings in Settings, Screen guidance, to use this.", speak, partial)
    if not cfg.get("ai_enabled") or not llm.vision_ok(cfg["llm"]):
        return await _deliver(session, "Screen drawings need an engine that can see images. Check vision in Settings.", speak, partial)
    await send(session.ws, type="state", state="looking")
    session.drawing_active = True
    early = None
    try:
        await _clear_drawing(session)
        shot, app, _ = await _unseen_shot(session)
        if shot is None or shot.frame is None or not shot.hwnd or shot.window is None:
            return await _deliver(session, "I couldn't capture the screen safely, so I can't draw on it.", speak, partial)
        candidates = await asyncio.to_thread(point.candidates, prompt, shot.pixels, shot.monitor, None, shot.hwnd)
        browser = app.lower() in {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe",
                                  "opera.exe", "vivaldi.exe", "arc.exe"}
        page = next((row.page_bounds for row in candidates if row.page_bounds is not None), None) if browser else None
        images = _visual_images(shot, candidates)
        samples, motion = [], None
        animations = ()
        if images:
            if not await _visual_sample(session, shot, samples):
                return await _deliver(session, "The screen changed before I could explain it. Ask again once it is still.", speak, partial)
            await asyncio.sleep(.25)
            if not await _visual_sample(session, shot, samples):
                return await _deliver(session, "The screen changed before I could explain it. Ask again once it is still.", speak, partial)
            animations = shot.frame.animation_regions(samples, images)
            motion = asyncio.create_task(_visual_motion_samples(session, shot, samples))
        try:
            try:
                plan = await visual_explanation.generate(prompt, shot, cfg, candidates, page_bounds=page,
                                                        history=session.history, dynamic_regions=animations)
            finally:
                if motion is not None:
                    motion.cancel()
                    with suppress(asyncio.CancelledError):
                        await motion
        except (ValueError, TimeoutError) as error:
            log.info("visual explanation withheld: %s", str(error) or "planning timed out")
            perf.visual_recorder()("plan_rejected", visual_explanation.rejection_reason(error))
            perf.outcome("drawing_failed")
            return await _deliver(session, "I couldn't make a reliable drawing plan for this screen. Try a simpler drawing request.", speak, partial)
        if plan.status == "unavailable":
            perf.outcome("drawing_unavailable")
            return await _deliver(session, plan.message, speak, partial)
        if plan.status == "plain":
            perf.mark("visual_presentation_plain")
            return await _deliver(session, plan.message, speak, partial)
        if images:
            animations = shot.frame.animation_regions(samples, images)
            plan = visual_explanation.stabilize(plan, animations)
            if animations:
                perf.mark("visual_animation_observed")
        perf.mark("drawing_plan_validated")
        if plan.status == "point":
            return await _visual_point(session, plan, speak, partial, animations=animations)
        perf.mark("visual_presentation_drawing")
        first = plan.beats[0]
        if speak and first.marks:
            # Recheck, dispatch and fly the bone while the first sentence is
            # still being synthesized; its audio still waits for paint readiness.
            early = asyncio.create_task(_begin_drawing_beat(
                session, plan.shot, first.marks, first.say, reveal_from=first.reveal_from,
                animations=animations, observed_regions=first.observed_regions))
        if speak and len(plan.beats) > 1:
            return await _visual_sequence(session, plan, partial, animations=animations, early=early)
        reply = []
        for beat in plan.beats:
            if (not beat.marks or (beat.reveal_from == len(beat.marks) and session.drawing_scene is not None)):
                if not await _visual_reasoning_fresh(session, plan.shot, animations):
                    await _clear_drawing(session)
                    reply.append(await _deliver(session, "The screen changed before I could show that, so I'll stop here.", speak, partial))
                    break
            if speak:
                # Each drawing beat has its own audio preparation. Keep the
                # previous speaking pose from lingering through that wait;
                # Speaker switches back only after this beat is ready to play.
                await send(session.ws, type="state", state="looking")
            if beat.marks:
                if (beat.reveal_from == len(beat.marks)
                        and getattr(session, "drawing_scene", None) is not None):
                    # Further reasoning may need no new ink. Preserve the marks
                    # and observer; do not make the bone trace them a second time.
                    session.drawing_deadline = time.monotonic() + drawing.ACTIVE_LIFETIME
                    await send(session.ws, type="drawing_keepalive",
                               presentation_id=session.drawing_scene["presentation_id"])
                    reply.append((await _deliver(session, beat.say, speak, partial)).strip())
                    if speak:
                        await _played(session)
                    else:
                        await asyncio.sleep(_dwell(beat.say))
                    continue
                heard = await _drawing_beat(session, plan.shot, beat.marks, beat.say, speak, partial,
                                             reveal_from=beat.reveal_from, wait_complete=True, animations=animations,
                                             fallback_target=beat.pointer, observed_regions=getattr(beat, "observed_regions", ()),
                                             early=early if beat is first else None, first=not reply)
                reply.append(heard.strip())
                if heard.strip() != beat.say: break
            else:
                await _clear_drawing(session)
                reply.append((await _deliver(session, beat.say, speak, partial)).strip())
                if speak: await _played(session)
        return " ".join(text.strip() for text in reply)
    except BaseException:
        await asyncio.shield(_clear_drawing(session))
        raise
    finally:
        session.drawing_active = False
        if early is not None and not early.done():
            early.cancel()
        # Every beat has finished its audio (or muted reading interval) and its
        # final stroke. Retire ink and return the borrowed bone together now.
        await asyncio.shield(_clear_drawing(session))


async def _hide_point(session: Session) -> None:
    """Take the bone away."""
    _retire_point(session)
    if getattr(session, "alive", True):
        await send(session.ws, type="point", nx=None)


async def _aim(session: Session, target: point.Target) -> guide.Presentation:
    """Put the bone on a row of the list."""
    log.info("pointing at %r via %s", target.label, target.source)
    _retire_point(session)
    # Register before send: an event can arrive before send_text returns.
    presentation = guide.Presentation(mark=perf.marker())
    session.presentation = presentation
    perf.mark("pointer_dispatched")
    await send(
        session.ws,
        type="point",
        nx=target.nx,
        ny=target.ny,
        label=target.label,
        monitor=target.monitor,
        presentation_id=presentation.id,
    )
    return presentation


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
            try:
                if not await _prepare_capture(session):
                    return None
                monitor = capture.known_monitor(getattr(session, "turn_monitor", None)) or capture.active_monitor()
                hwnd, app, title = capture.window_on_monitor(monitor) if monitor else (0, "", "")
                # Capture native pixels once. UIA, OCR and JPEG encoding then
                # proceed together over the same requested screen state.
                uia_task = asyncio.create_task(asyncio.to_thread(point.uia_candidates, hwnd))
                captured_at = time.monotonic()
                captured = await asyncio.to_thread(capture.frame, monitor)
                if captured is None or monitor is None:
                    await uia_task
                    return None
                image, pixels, _chosen = captured
                cached = await asyncio.to_thread(
                    point.cached_evidence, hwnd, monitor, pixels
                )
                encode_task = asyncio.create_task(
                    asyncio.to_thread(capture.encode, image, capture.POINT_EDGE)
                )
                if cached is None:
                    pending_ocr = point.start_ocr(pixels)
                    ocr_task = asyncio.create_task(
                        asyncio.to_thread(point.collect_ocr, pending_ocr)
                    )
                    encoded, accessible, ocr_rows = await asyncio.gather(
                        encode_task, uia_task, ocr_task
                    )
                    await asyncio.to_thread(
                        point.remember_evidence,
                        hwnd, monitor, pixels, accessible, ocr_rows,
                    )
                else:
                    accessible, ocr_rows = cached
                    encoded = await encode_task
                    uia_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await uia_task
                window = capture.window_rect(hwnd)
                shot = Shot(*encoded, pixels, monitor, hwnd, window,
                            drawing.Frame.capture(monitor, hwnd, window, encoded[1], encoded[2], pixels, captured=captured_at))
            finally:
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.shield(send(session.ws, type="capture", phase="end", capture_id=session.capture_id))
                session.capture_active = False
            if shot is None:
                return None
            cands = await asyncio.to_thread(
                point.candidates, prompt, shot.pixels, shot.monitor, None, shot.hwnd,
                accessible, ocr_rows,
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


def _same_control(local, aimed) -> bool:
    """True when the measured rectangle is the control the model chose."""
    monitor = aimed.monitor or local.monitor
    if local.bounds is None or not monitor:
        return False
    # nx/ny are normalized within the owning monitor; bounds are desktop pixels
    # as (left, top, width, height). Half a control of slack, so a near-miss on
    # a small button still counts as the same control.
    slack = max(local.bounds[2], local.bounds[3]) / 2
    return locator._inside(
        local.bounds,
        monitor["left"] + aimed.nx * monitor["width"],
        monitor["top"] + aimed.ny * monitor["height"],
        slack,
    )


async def _aim_early(session, shot, target, local_fallback):
    """Verify and show the model's choice while it is still writing its sentence.

    The same checks as after a finished reply: prefer an identical measured
    rectangle, capture a fresh frame, and show nothing if the control moved.
    Returns (aimed, fresh) once the bone is up, or None to leave the target to
    the normal path, which can track or re-resolve a moved control.
    """
    aimed = target
    if local_fallback is not None and _same_control(local_fallback, target):
        aimed = local_fallback
        perf.mark("pointer_local_precision")
    fresh, _, _ = await _unseen_shot(session, capture.POINT_EDGE)
    if fresh is None or locator.changed_at(shot, fresh, aimed):
        return None
    await _aim(session, aimed)
    perf.mark("pointer_early")
    return aimed, fresh


async def _resolve_point(prompt, shot, cfg, cands, history, on_choice=None):
    if cfg["llm"]["mode"] == "agent":
        try:
            result = await locator.locate_and_answer(
                prompt, shot, cfg, cands, history, on_choice=on_choice
            )
        except agents.AgentTimeout:
            perf.mark("locator_timeout")
            log.info("agent locator timed out; withholding the bone")
            return locator.GroundedResult(
                None, "I couldn't lock onto that control quickly enough."
            )
        if result.target is None and not result.answer:
            perf.mark("pointer_withheld")
            return locator.GroundedResult(
                None, "I couldn't lock onto that control safely."
            )
        return result
    if cfg["llm"]["mode"] == "cloud":
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
    await _clear_drawing(session)

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
    said = await asyncio.to_thread(sessions.record, "user_said", text=prompt)
    # What Mellow knows about them for this turn: About you, the memory summary,
    # and the result of any "remember that" or "forget that". Actions never carry it.
    remembered = (
        await asyncio.to_thread(memory.turn_context, cfg, prompt, said)
        if cfg.get("ai_enabled", True)
        else ""
    )
    if remembered:
        # Metadata only: proves memory reached this answer without logging its text.
        log.info("memory: %d chars in this answer", len(remembered))
    reply = ""
    # Preserve partial output.
    partial = {"text": ""}
    # Probe local vision.
    await asyncio.to_thread(llm.probe_vision, cfg["llm"])
    # Check model fit.
    await asyncio.to_thread(llm.check_fit, cfg["llm"])
    sighted = llm.vision_ok(cfg["llm"])
    # Route screen requests.
    wants_pointer = capture.wants_pointing(prompt)
    pointing = sighted and wants_pointer
    asked = sighted and (capture.wants_screen(prompt) or pointing)
    # Selected target.
    aimed: point.Target | None = None
    # Answer generated alongside target selection.
    grounded_answer = ""
    # Actions require cloud or agent mode.
    doing = (
        cfg["llm"]["mode"] in ("cloud", "agent")
        and not wants_pointer
        and capture.wants_action(prompt)
    )
    drawing_route = visual_explanation.request_kind(prompt, automatic=cfg.get("drawing_enabled", False) and sighted,
                                                   history=session.history[:-1])
    draw_requested = drawing_route in ("explicit", "automatic")
    perf.mark("drawing_route_" + drawing_route)
    try:
        did = False
        if draw_requested:
            reply = await _visual_answer(session, prompt, {**cfg, "memory": remembered}, speak, partial)
            # Normal history/logging and final speech cleanup still apply.
            asked = False
        elif doing:
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
        if remembered:
            cfg = {**cfg, "memory": remembered}
        if not asked and not draw_requested:
            reply, asked, _ = await _pass(
                session,
                cfg,
                speak,
                # Vision-off has no marker.
                look="ask" if sighted else "",
                partial=partial,
            )
        if asked and cfg.get("drawing_enabled") and drawing_route != "suppressed":
            # The answering model decided it must see the screen. With drawings
            # on, every screen answer goes through the visual planner: draw, else
            # point, else words. Phrase routing misses this; the model does not.
            perf.mark("drawing_route_look")
            reply = await _visual_answer(session, prompt, cfg, speak, partial)
            asked = False
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

                    # Keep a high-confidence measured result as a safety net,
                    # but let the agent write the normal answer and validate the
                    # same visual request as API mode. This removes canned speech.
                    local_fallback = (
                        point.confident_match(cands)
                        if cfg["llm"]["mode"] == "agent"
                        else None
                    )
                    early: dict = {}

                    def on_choice(target, shot=shot, local_fallback=local_fallback):
                        early["target"] = target
                        early["task"] = asyncio.create_task(
                            _aim_early(session, shot, target, local_fallback)
                        )

                    try:
                        grounded = await _resolve_point(
                            prompt, shot, cfg, cands, session.history,
                            on_choice=on_choice if cfg["llm"]["mode"] == "agent" else None,
                        )
                    except BaseException:
                        # A barge-in must not leave a bone arriving after it.
                        if "task" in early:
                            early["task"].cancel()
                            with suppress(BaseException):
                                await early["task"]
                        raise
                    shown = None
                    if "task" in early:
                        try:
                            shown = await early["task"]
                        except Exception:
                            log.exception("early pointer failed; verifying normally")
                    aimed, grounded_answer = grounded.target, grounded.answer
                    already_shown = shown is not None and (
                        aimed is None or point._same_place(aimed, early["target"])
                    )
                    if already_shown:
                        # Already verified on a fresh frame and on screen.
                        aimed, shot = shown
                        if grounded.target is None:
                            grounded_answer = locator.pointer_reply(aimed)
                    elif local_fallback is not None and aimed is not None:
                        # A unique, high-confidence UIA rectangle is more precise
                        # than a model-copied E index — but only when it is the
                        # same control. Swapping in a lexical match that landed
                        # elsewhere is the phrase-gate veto again, wearing the
                        # word "precision": it points at one control while the
                        # spoken answer describes another.
                        if _same_control(local_fallback, aimed):
                            aimed = local_fallback
                            perf.mark("pointer_local_precision")
                    elif aimed is None and local_fallback is not None:
                        aimed = local_fallback
                        grounded_answer = locator.pointer_reply(aimed)
                        perf.mark("pointer_local_fallback")
                        perf.record_pointer(
                            outcome="local_fallback",
                            candidates=len(cands),
                            source=aimed.source,
                        )
                        log.info("agent failed; using verified local target %r", aimed.label)
                    if aimed and not already_shown:
                        fresh, _, _ = await _unseen_shot(session, capture.POINT_EDGE)
                        if fresh is None:
                            log.info("could not verify the localized target; withholding the bone")
                            perf.mark("pointer_withheld")
                            aimed = None
                            grounded_answer = (
                                "I couldn't verify that control on the fresh screen."
                                if cfg["llm"]["mode"] == "agent"
                                else ""
                            )
                        elif locator.changed_at(shot, fresh, aimed):
                            tracked = await asyncio.to_thread(
                                locator.relocate_target, shot, fresh, aimed
                            )
                            if tracked is not None:
                                aimed, shot = tracked, fresh
                                # The frame moved under step 1. Later steps were
                                # chosen on the old one, so they are no longer
                                # measurements of anything on screen.
                                grounded = locator.GroundedResult(aimed, grounded_answer)
                                log.info("tracked the localized control on the fresh frame")
                            else:
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
                                if cfg["llm"]["mode"] == "agent":
                                    local_fallback = point.confident_match(cands)
                                    grounded = await _resolve_point(
                                        prompt, shot, cfg, cands, session.history
                                    )
                                    aimed, grounded_answer = grounded.target, grounded.answer
                                    if aimed is None and local_fallback is not None:
                                        aimed = local_fallback
                                        grounded_answer = locator.pointer_reply(aimed)
                                        perf.mark("pointer_local_fallback")
                                else:
                                    grounded = await _resolve_point(
                                        prompt, shot, cfg, cands, session.history
                                    )
                                    aimed, grounded_answer = grounded.target, grounded.answer
                                if aimed:
                                    verified, _, _ = await _unseen_shot(
                                        session, capture.POINT_EDGE
                                    )
                                    moved = (
                                        verified is not None
                                        and locator.changed_at(shot, verified, aimed)
                                    )
                                    if moved:
                                        aimed = await asyncio.to_thread(
                                            locator.relocate_target, shot, verified, aimed
                                        )
                                    if verified is None or aimed is None:
                                        log.info(
                                            "localized area moved twice; withholding the bone"
                                        )
                                        perf.mark("pointer_withheld")
                                        grounded_answer = (
                                            "The screen changed before I could verify "
                                            "that control."
                                            if cfg["llm"]["mode"] == "agent"
                                            else ""
                                        )
                                    else:
                                        shot = verified
                        else:
                            shot = fresh
                        if aimed:
                            await _aim(session, aimed)

                beats = (
                    locator.narration(grounded, aimed, grounded_answer)
                    if pointing
                    else ()
                )
                if beats:
                    _pending_guide(session, prompt, grounded, beats, shot, partial)
                    reply = await _narrate(session, beats, speak, partial, shot)
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
                        presentation=getattr(session, "presentation", None) if aimed else None,
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
            await asyncio.shield(_clear_drawing(session))
        raise

    await asyncio.to_thread(
        sessions.record, "assistant_said", **_said(cfg, reply, False)
    )
    session.history.append({"role": "assistant", "content": reply})
    del session.history[: max(0, len(session.history) - HISTORY_TURNS * 2)]

    if speak:
        # Wait for playback.
        await session.speaker.finish()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError

    # The frontend retires the bone.
    await send(ws, type="state", state="idle")
    if pending := getattr(session, "guide_pending", None):
        pending.ready.set()
        session.guide_task = asyncio.create_task(perf.run(_guide_loop(session), perf.Turn("guide")))


async def run_turn(session: Session, prompt: str) -> None:
    """A turn owns its own error handling, because as a separate task it's outside the message loop's"""
    prepared = None
    try:
        pending = getattr(session, "guide_pending", None)
        if pending and re.fullmatch(
            r"\s*(?:next|continue|go on|carry on|done|i(?:'ve| have)? (?:done it|opened it|clicked it)|what next|what now)[.!?\s]*",
            prompt, re.I,
        ):
            pending.ready.set()
            pending.partial = {"text": ""}
            session.guide_task = asyncio.current_task()
            await asyncio.to_thread(sessions.record, "user_said", text=prompt)
            await _guide_loop(session, force=True)
            return
        if pending:
            await _stop_guide(session)
        cfg = config.load()
        if prompt and (cfg.get("ai_enabled") and cfg["llm"]["mode"] in ("cloud", "agent")
                and llm.vision_ok(cfg["llm"]) and capture.wants_pointing(prompt)
                and not visual_explanation.requested(prompt, automatic=cfg.get("drawing_enabled", False), history=session.history)):
            await _hide_point(session)
            prepared = asyncio.create_task(_prepare_pointing(session, prompt))
        if cfg.get("writing_enabled") and prompt and not visual_explanation.requested(prompt, automatic=cfg.get("drawing_enabled", False), history=session.history):
            if prepared is None:
                await _hide_point(session)
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
            await _clear_drawing(session)
        await _stop_guide(session)
        await session.speaker.stop()
        await send(session.ws, type="error", message=errors.message(e))
        await send(session.ws, type="state", state="idle")
    finally:
        await _discard_preparation(prepared)
        memory.learner.turn_ended()


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


async def _writing_start(session: Session, finding=None) -> None:
    session.writer.begin()
    if config.load().get("writing_enabled"):
        # Resolve the destination before any later microphone wait or UI update
        # can allow a transient Windows surface to replace the focused field.
        session.writer.finding = finding or asyncio.create_task(
            asyncio.to_thread(writing.desktop.resolve, writing.FIND_SECONDS)
        )
    else:
        log.info("writing target capture skipped: screen-aware writing is off")
    await writing.status(session, send, "idle")


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

    elif kind == "guide_ack":
        presentation = getattr(session, "presentation", None)
        if presentation is not None and msg.get("presentation_id") == presentation.id:
            presentation.acknowledge(msg.get("outcome"))

    elif kind == "capture_ready":
        if msg.get("capture_id") == session.capture_id and session.capture_id is not None:
            session.capture_safe = msg.get("ok") is True
            session.hidden.set()

    elif kind == "drawing_ack":
        presentation = session.drawing_presentation
        if presentation is not None and msg.get("presentation_id") == presentation.id:
            outcome = msg.get("outcome")
            if outcome == "position":
                try:
                    rects = msg.get("occlusions", [])
                    scale = session.drawing_scale
                    if not isinstance(rects, list) or len(rects) > 3: raise ValueError("invalid pen masks")
                    clean = []
                    for row in rects:
                        if not isinstance(row, list) or len(row) != 4: raise ValueError("invalid pen mask")
                        x,y,w,h = map(drawing.number, row)
                        if not 0 < w <= 256 * scale or not 0 < h <= 64 * scale: raise ValueError("pen mask too large")
                        clean.append([x,y,w,h])
                    session.drawing_pen_occlusions = clean
                except ValueError:
                    pass
                return
            if outcome == "ready":
                try:
                    remaining_ms = drawing.number(msg.get("remaining_ms"))
                    if 0 <= remaining_ms <= 55000:
                        session.drawing_completion_deadline = time.monotonic() + remaining_ms / 1000 + 2
                except ValueError:
                    pass
                try:
                    scale = drawing.number(msg.get("scale", 1))
                    if .5 <= scale <= 8: session.drawing_scale = scale
                except ValueError:
                    pass
                occlusions = msg.get("occlusions", [])
                if isinstance(occlusions, list) and len(occlusions) <= 8:
                    try:
                        checked = [[drawing.number(v) for v in rect] for rect in occlusions if isinstance(rect, list) and len(rect) == 4]
                        if len(checked) == len(occlusions) and all(w > 0 and h > 0 for _, _, w, h in checked):
                            session.drawing_occlusions = checked
                    except ValueError:
                        pass
                presentation.acknowledge("arrived")
                presentation.mark("drawing_ready")
            elif outcome == "complete":
                presentation.mark("drawing_complete")
                if presentation.outcome == "arrived" and not presentation.retired:
                    session.drawing_complete.set()
            elif outcome == "failed":
                reason = msg.get("reason")
                if reason not in {"expired", "source_changed", "pet_hidden", "display_changed",
                                   "replaced", "cleared", "native_failure", "presentation_failed"}:
                    reason = "native_failure"
                session.drawing_failure_reason = reason
                if presentation.outcome is None:
                    if record := getattr(session, "drawing_diagnostic", None):
                        record("not_ready", reason)
                    # Let the playback waiter consume failure before retirement.
                    presentation.acknowledge("failed")
                else:
                    await _invalidate_drawing(session, reason)

    elif kind == "awake":
        # Wake the mic keeper.
        if msg.get("value"):
            session.wake_mic()
            cfg = config.load()
            if cfg.get("ai_enabled"):
                point.warm_ocr()
            if cfg.get("ai_enabled") and cfg["llm"].get("mode") == "agent":
                asyncio.create_task(agents.warm(cfg))
        else:
            session.awake = False
            session.mic_ready = False
            await session.abort()
            await session.cancel_ptt_start()
            await asyncio.to_thread(session.recorder.close, immediate=True)
            await session._send_mic("off")
            await agents.stop()
            point.stop_ocr()

    elif kind == "set_speak":
        cfg = config.load()
        cfg["tts"]["speak"] = bool(msg.get("value"))
        await asyncio.to_thread(config.save, cfg)
        if not cfg["tts"]["speak"]:
            await session.speaker.stop()
        await send(ws, type="speak", value=cfg["tts"]["speak"])

    elif kind == "ptt_start":
        # Start hardware wake-up before cancelling the previous turn so both
        # operations overlap instead of adding their latency.
        session.wake_mic()
        # A live turn wins over background learning, which retries later.
        memory.learner.turn_started()
        cfg = config.load()
        if cfg.get("ai_enabled"):
            point.warm_ocr()
        if cfg.get("ai_enabled") and cfg["llm"].get("mode") == "agent":
            # Idempotent when already ready. On a wake hotkey this overlaps mic,
            # field discovery, cancellation and the user's speech.
            asyncio.create_task(agents.warm(cfg))
        # Snapshot the focused field at the same moment as microphone wake-up.
        # Keep this task detached until abort() has cancelled the prior writer.
        writing_finding = (
            asyncio.create_task(
                asyncio.to_thread(writing.desktop.resolve, writing.FIND_SECONDS)
            )
            if cfg.get("writing_enabled")
            else None
        )
        await session.abort(preserve_guide=True)
        session.turn_monitor = None
        # Pet-only mode cannot listen.
        if standby():
            if writing_finding is not None:
                writing_finding.cancel()
            await send(ws, type="reply_chunk", text=PET_ONLY_LINE)
            await send(ws, type="state", state="idle")
            return
        # Field capture starts immediately after cancellation has settled. It
        # now overlaps microphone readiness instead of waiting behind it.
        await _writing_start(session, writing_finding)
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
        memory.learner.turn_started()
        await session.abort(preserve_guide=True)
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
        # The conversation just finished: it can be learned from once idle.
        memory.learner.nudge()

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
