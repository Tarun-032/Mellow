"""Turn-local performance metadata. Never records user content or credentials."""

import asyncio
from contextlib import contextmanager, aclosing
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import uuid

from mellowd import config

_current = ContextVar("mellow_latency_turn", default=None)
_purpose = ContextVar("mellow_model_purpose", default="answer")
_write_lock = threading.Lock()


class Turn:
    def __init__(self, source, started=None):
        self.id = uuid.uuid4().hex
        self.source = source
        self.started = time.perf_counter() if started is None else started
        self.stages = []
        self.agent_calls = []
        self.pointer = []
        self.visual = []
        self.marks = {}
        self.outcome = "completed"
        self.closed = False
        self.lock = threading.Lock()

    def mark(self, name):
        with self.lock:
            if not self.closed:
                self.marks.setdefault(name, round((time.perf_counter() - self.started) * 1000, 3))

    def stage(self, name, started, outcome, model="", first_content_at=None):
        with self.lock:
            if not self.closed:
                self.stages.append({"stage": name, "start_ms": round((started - self.started) * 1000, 3),
                                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                                    "outcome": outcome, **({"model": model} if model else {}),
                                    **({"first_content_ms": round((first_content_at - started) * 1000, 3)}
                                       if first_content_at is not None else {})})

    def snapshot(self):
        with self.lock:
            self.closed = True
            return {"v": 2, "turn": self.id, "source": self.source, "outcome": self.outcome,
                    "duration_ms": round((time.perf_counter() - self.started) * 1000, 3),
                    "marks_ms": dict(self.marks), "stages": list(self.stages),
                    "agent_calls": list(self.agent_calls),
                    "pointer": list(self.pointer), "visual": list(self.visual)}


def mark(name):
    if turn := _current.get():
        turn.mark(name)


def marker():
    """Bind timing to its owning turn, including receipts on the WS task."""
    turn = _current.get()
    return turn.mark if turn is not None else lambda _: None


def visual_recorder():
    """Bind bounded diagnostic codes to a turn, including native WS receipts."""
    turn = _current.get()
    def record(outcome, reason):
        # Callers supply fixed codes, never provider output or screen text.
        if (not turn or not isinstance(reason, str) or not isinstance(outcome, str)
                or len(reason) > 64 or len(outcome) > 32
                or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in outcome + reason)):
            return
        with turn.lock:
            if not turn.closed and len(turn.visual) < 32:
                turn.visual.append({"outcome": outcome, "reason": reason,
                                    "at_ms": round((time.perf_counter() - turn.started) * 1000, 3)})
    return record


def outcome(value):
    if turn := _current.get():
        with turn.lock:
            if not turn.closed:
                turn.outcome = value


def record_agent(*, provider, purpose, transport, prompt_bytes, image_bytes,
                 schema_bytes, usage, accepted, started, first_event=None,
                 first_text=None, input_sent=None):
    """Attach metadata-only provider timing; user content never enters the log."""
    turn = _current.get()
    if not turn:
        return
    detail = usage.get("output_tokens_details") if isinstance(usage, dict) else {}
    if not isinstance(detail, dict):
        detail = {}
    call = {
        "provider": provider,
        "purpose": purpose,
        "transport": transport,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "prompt_bytes": prompt_bytes,
        "image_bytes": image_bytes,
        "schema_bytes": schema_bytes,
        "accepted": bool(accepted),
        "input_tokens": usage.get("input_tokens", 0) if isinstance(usage, dict) else 0,
        "cached_tokens": (
            usage.get("cache_read_input_tokens", usage.get("cached_input_tokens", 0))
            if isinstance(usage, dict) else 0
        ),
        # Claude reports cache writes apart from input; Codex names them differently.
        "cache_write_tokens": (
            usage.get("cache_creation_input_tokens", usage.get("cache_write_input_tokens", 0))
            if isinstance(usage, dict) else 0
        ),
        "output_tokens": usage.get("output_tokens", 0) if isinstance(usage, dict) else 0,
        "reasoning_tokens": (
            detail.get("thinking_tokens", usage.get("reasoning_output_tokens", 0))
            if isinstance(usage, dict) else 0
        ),
    }
    if first_event is not None:
        call["first_event_ms"] = round((first_event - started) * 1000, 3)
    if first_text is not None:
        call["first_text_ms"] = round((first_text - started) * 1000, 3)
    if input_sent is not None:
        call["local_handoff_ms"] = round((input_sent - started) * 1000, 3)
    with turn.lock:
        if not turn.closed:
            turn.agent_calls.append(call)


def record_model(*, provider, prompt_bytes, usage):
    """An API adapter call, recorded beside the agent calls with transport "api"."""
    turn = _current.get()
    if not turn:
        return
    usage = usage if isinstance(usage, dict) else {}
    call = {
        "provider": provider,
        "purpose": _purpose.get(),
        "transport": "api",
        "prompt_bytes": prompt_bytes,
        "image_bytes": 0,
        "schema_bytes": 0,
        "accepted": True,
        "reported": bool(usage),
        "input_tokens": usage.get("input_tokens", 0),
        "cached_tokens": usage.get("cache_read_input_tokens", usage.get("cached_input_tokens", 0)),
        "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": 0,
    }
    with turn.lock:
        if not turn.closed:
            turn.agent_calls.append(call)


def record_pointer(*, outcome: str, candidates: int = 0, measured: int = 0,
                   source: str = "", step: int = 0) -> None:
    """Attach target-selection metadata without screen text or coordinates."""
    turn = _current.get()
    if not turn:
        return
    row = {
        "outcome": outcome,
        "candidates": int(candidates),
        "measured": int(measured),
    }
    if source:
        row["source"] = source
    # Absent on a one-bone turn, so existing rows keep their shape.
    if step:
        row["step"] = int(step)
    with turn.lock:
        if not turn.closed:
            turn.pointer.append(row)


@contextmanager
def span(name, model=""):
    turn = _current.get()
    started = time.perf_counter()
    result = "completed"
    measurements = {}
    try:
        yield measurements
    except asyncio.CancelledError:
        result = "cancelled"
        raise
    except GeneratorExit:
        result = "closed_early"
        raise
    except BaseException:
        result = "failed"
        raise
    finally:
        if turn:
            turn.stage(name, started, result, model, measurements.get("first_content_at"))


@contextmanager
def purpose(name):
    token = _purpose.set(name)
    try:
        yield
    finally:
        _purpose.reset(token)


def timed(name):
    def decorate(fn):
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def asynchronous(*args, **kwargs):
                with span(name):
                    return await fn(*args, **kwargs)
            return asynchronous
        @wraps(fn)
        def synchronous(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)
        return synchronous
    return decorate


def model_stream(fn):
    """One adapter invocation, including time to first clean content."""
    @wraps(fn)
    async def wrapped(cfg, *args, **kwargs):
        name = _purpose.get()
        with span("model." + name, model=cfg.get("model", "")) as measurements:
            async with aclosing(fn(cfg, *args, **kwargs)) as stream:
                first = True
                async for chunk in stream:
                    if first and chunk:
                        measurements["first_content_at"] = time.perf_counter()
                        mark("model." + name + ".first_content")
                        first = False
                    yield chunk
    return wrapped


def _write(summary):
    # Short-lived handler avoids holding a file open across app/test lifetimes.
    # Serialization happens off-loop, and handler errors cannot fail a turn.
    try:
        with _write_lock:
            folder = config.CONFIG_DIR / "diagnostics"
            folder.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(folder / "latency.jsonl", maxBytes=1_000_000,
                                          backupCount=2, encoding="utf-8")
            try:
                handler.handle(logging.LogRecord("latency", logging.INFO, "", 0,
                                                  json.dumps(summary), (), None))
            finally:
                handler.close()
    except Exception:
        pass


async def run(awaitable, turn):
    token = _current.set(turn)
    try:
        return await awaitable
    except asyncio.CancelledError:
        outcome("cancelled")
        raise
    except BaseException:
        outcome("failed")
        raise
    finally:
        summary = turn.snapshot()
        _current.reset(token)
        # Finish the original turn's immutable summary even during barge-in.
        await asyncio.shield(asyncio.to_thread(_write, summary))
