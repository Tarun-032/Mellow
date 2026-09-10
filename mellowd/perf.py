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
            return {"v": 1, "turn": self.id, "source": self.source, "outcome": self.outcome,
                    "duration_ms": round((time.perf_counter() - self.started) * 1000, 3),
                    "marks_ms": dict(self.marks), "stages": list(self.stages)}


def mark(name):
    if turn := _current.get():
        turn.mark(name)


def outcome(value):
    if turn := _current.get():
        with turn.lock:
            if not turn.closed:
                turn.outcome = value


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
