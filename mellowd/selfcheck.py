"""End-to-end check for the sidecar."""

import asyncio
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from websockets.asyncio.client import connect

URL = "ws://127.0.0.1:8765/ws"
SPEECH_CHECK = "mellow can hear this sentence clearly"


async def recv(ws, timeout=10.0):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


async def drain_until_idle(ws, timeout=180.0):
    """Collect a turn until it ends."""
    reply, first, talked = "", None, False
    t0 = time.perf_counter()
    while True:
        m = await recv(ws, timeout)
        if m["type"] == "error":
            raise AssertionError(f"turn failed: {m['message']}")
        if m["type"] == "reply_chunk":
            first = first or time.perf_counter() - t0
            reply += m["text"]
        elif m["type"] == "state":
            if m["state"] == "talking":
                talked = True
            elif m["state"] == "idle":
                return reply, first, talked


def check_sentences() -> None:
    """Pure logic, so it gets real assertions rather than a vibe check."""
    from mellowd import tts

    buf = tts.SentenceBuffer()
    got = []
    for chunk in ["hello ", "there. how ", "are you? ", "fine"]:
        got += buf.feed(chunk)
    got += buf.flush()
    assert got == ["hello there.", "how are you?", "fine"], got

    # A decimal must not look like a sentence end.
    buf = tts.SentenceBuffer()
    assert buf.feed("it costs 3.5 dollars") == [], "split inside a decimal"
    assert buf.flush() == ["it costs 3.5 dollars"]

    # Long unpunctuated text must still start speaking.
    buf = tts.SentenceBuffer()
    out = buf.feed("word " * 60)
    assert out, "MAX_CHARS force-flush never fired"
    assert all(len(s) <= tts.MAX_CHARS for s in out), [len(s) for s in out]

    assert tts.clean_for_speech("**bold** and `code` 🎉") == "bold and code"
    print("ok  sentence splitter + speech cleaner")


async def check_speech_pipeline() -> None:
    """Fetching a sentence must overlap playing the previous one."""
    import threading

    import sounddevice as sd

    from mellowd import tts

    FETCH, PLAY = 0.10, 0.15
    lines = ["one.", "two.", "three.", "four."]
    played: list[str] = []
    cut = threading.Event()

    def fake_synth(text, cfg=None):
        time.sleep(FETCH)
        return text, len(text)  # (samples, rate), opaque to the Speaker

    def fake_play(samples, rate, blocking=True):
        # sd.stop() is what releases a blocked sd.play — cancelling the task cannot
        if cut.wait(PLAY):
            return
        played.append(samples)

    said: list[str] = []

    async def send(ws, **fields):
        said.append(fields.get("state"))

    real_synth, real_play, real_stop = tts.synth, sd.play, sd.stop
    tts.synth, sd.play, sd.stop = fake_synth, fake_play, cut.set
    try:
        speaker = tts.Speaker(object(), send)
        cut.clear()
        speaker.begin()
        for line in lines:
            await speaker.speak(line)
        t0 = time.perf_counter()
        await speaker.finish()
        elapsed = time.perf_counter() - t0

        assert played == lines, played
        assert said[:1] == ["talking"], said
        # Serialised this would be 4*(FETCH+PLAY) = 1.00s
        serial = len(lines) * (FETCH + PLAY)
        assert elapsed < serial * 0.75, f"still serialised: {elapsed:.2f}s of {serial:.2f}s"

        # Barge-in has to kill both stages.
        cut.clear()
        speaker.begin()
        for line in lines:
            await speaker.speak(line)
        await asyncio.sleep(FETCH + PLAY / 2)
        before = len(played)
        await speaker.stop()
        await asyncio.sleep(FETCH + PLAY)
        assert len(played) == before, f"{len(played) - before} clips played after stop"
    finally:
        tts.synth, sd.play, sd.stop = real_synth, real_play, real_stop

    print(f"ok  speech pipeline overlaps fetch and playback ({elapsed:.2f}s of {serial:.2f}s serial)")


def check_stale_refusals() -> None:
    """A refusal must not outlive the reason for it."""
    from mellowd import llm

    stale = (
        "I cannot see your screen. The model I am running on does not take images.",
        "I can't see your screen. I am mellow.",
        "No, I can't see your screen. I only hear you when you hold the "
        "push-to-talk key, and I answer out loud.",
        "I can't see your screen on my own.",
        "I cannot see your screen, open apps, or click things.",
    )
    for text in stale:
        assert llm._BLIND_CLAIM.search(text), f"missed a real refusal: {text!r}"

    # Answers that merely mention screens or seeing are not refusals.
    keep = (
        "Your screen shows the NVIDIA build page for thinkingmachines slash inkling.",
        "I can see your screen. Visual Studio Code is open to architecture dot md.",
        "You can't see the file because it's hidden — turn on hidden items in the ribbon.",
        "I don't know what that error means without more context.",
        "The capital of France is Paris.",
    )
    for text in keep:
        assert not llm._BLIND_CLAIM.search(text), f"ate a real answer: {text!r}"

    history = [
        {"role": "user", "content": "can you see my screen?"},
        {"role": "assistant", "content": "I can't see your screen. I am mellow."},
        {"role": "user", "content": "what is on it now?"},
    ]
    out = llm._drop_stale_refusals(history)
    # The refusal goes, and the two user turns it was between must not end up adjacent
    assert [m["role"] for m in out] == ["user"], out
    assert out[-1]["content"] == "what is on it now?", out

    kept = llm._drop_stale_refusals(
        [
            {"role": "user", "content": "what is on my screen?"},
            {"role": "assistant", "content": "Your screen shows a settings page."},
        ]
    )
    assert len(kept) == 2, kept
    print(f"ok  stale can't-see turns dropped, {len(keep)} real answers kept")


def check_thinking_filter() -> None:
    """A truncated chain of thought must not reach the user as an answer."""
    from mellowd import llm

    # The bad case: content echoes the thinking from the first token.
    echo = llm._Stream()
    thought = "Okay, the user is asking again about what model I'm running on."
    kept = ""
    for i in range(0, len(thought), 7):
        piece = thought[i : i + 7]
        echo.thoughts += piece
        if not echo.is_thinking(piece):
            kept += piece
    assert kept == "", f"leaked thinking: {kept!r}"

    # The good case: same thinking, then a real answer. Every word survives.
    real = llm._Stream()
    real.thoughts = thought
    answer = "i'm running on nemotron, but i'm still mellow."
    kept = "".join(
        p for p in (answer[i : i + 7] for i in range(0, len(answer), 7))
        if not real.is_thinking(p)
    )
    assert kept == answer, f"ate the answer: {kept!r}"

    # No reasoning channel at all: nothing to compare against, nothing filtered.
    plain = llm._Stream()
    assert not plain.is_thinking("hello")

    # Groq Qwen raw format: thinking arrives inside <think> tags in content
    tagged = llm._Stream()
    parts = [
        "<thi",
        "nk>\nHere's a thinking process:\n1. Analyze.\n",
        "</thi",
        "nk>\n\nI am Mellow.",
    ]
    kept = "".join(tagged.strip_think(p) for p in parts) + tagged.flush_think()
    assert kept == "\n\nI am Mellow.", f"think tags leaked or ate answer: {kept!r}"
    assert tagged.reasoning, "a think block should count as reasoning"

    # Google AI Studio Gemma 4: same idea, different tag (<thought>)
    gemma = llm._Stream()
    raw = (
        "<thought>*   User says: Hello.\n"
        "    *   Answer: I'm doing well.</thought>"
        "I am doing great, just hanging out and ready to help."
    )
    kept = "".join(gemma.strip_think(raw[i : i + 9]) for i in range(0, len(raw), 9))
    kept += gemma.flush_think()
    assert kept == "I am doing great, just hanging out and ready to help.", (
        f"thought tags leaked or ate answer: {kept!r}"
    )
    print("ok  truncated reasoning is filtered, real answers are not")


def check_opener_filter() -> None:
    """Filler openers come off the front; real sentences survive intact."""
    from mellowd import llm

    def stream(answer: str, step: int = 4) -> str:
        """Feed an answer through _Stream the way a provider would, in pieces."""
        seen = llm._Stream()
        out = "".join(
            seen.emit(answer[i : i + step]) for i in range(0, len(answer), step)
        )
        return out + seen.flush()

    strip = {
        "Obviously, the disk is slower.": "The disk is slower.",
        "So, the reason is memory.": "The reason is memory.",
        "Well, it depends on the model.": "It depends on the model.",
        "Basically it holds ten turns.": "It holds ten turns.",
        "Hmm, I don't know.": "I don't know.",
        "Great question! The answer is four.": "The answer is four.",
        "So, well, the answer is four.": "The answer is four.",
    }
    keep = (
        "So far, the count is four.",  # ambiguous word, no comma — an answer
        "Right now it's running on gemma.",
        "Actually running that needs a GPU.",
        "Surely you noticed the lag.",  # must not be eaten as "sure"
        "Somewhere in the config file.",  # must not be eaten as "so"
        "I'm running on gemma three.",
        "Looking at memory use helps.",  # must not be eaten as "look"
    )
    for chunked in (1, 3, 100):  # token-at-a-time, realistic, and one whole gulp
        for bad, good in strip.items():
            got = stream(bad, chunked)
            assert got == good, f"step {chunked}: {bad!r} -> {got!r}, wanted {good!r}"
        for good in keep:
            got = stream(good, chunked)
            assert got == good, f"step {chunked}: mangled {good!r} -> {got!r}"

    # An answer that is nothing but filler is left alone rather than emptied
    assert stream("Well, ") == "Well, ", "an all-filler answer must survive"
    print(f"ok  {len(strip)} filler openers stripped, {len(keep)} lookalikes kept")


def check_tone_contract() -> None:
    """The prompt, anchor and reminder must not contradict the register."""
    from mellowd import config, llm

    # llm.CORE, not config.DEFAULTS["system_prompt"]
    prompt = llm.CORE
    assert config.DEFAULTS["system_prompt"] == "", (
        "the box has a default again, and clearing it can delete the rules"
    )
    answers = [a for _, a in llm.ANCHOR]

    for phrase in ("either way", "lowercase", "one or two sentences"):
        assert phrase not in prompt, f"prompt still says {phrase!r}"
    for answer in answers:
        assert not llm._OPENER.match(answer), f"anchor opens with filler: {answer!r}"
        assert answer[0].isupper(), f"anchor answer is not sentence case: {answer!r}"
        for symbol in ("*", "#", "`", "- "):
            assert symbol not in answer, f"anchor shows markdown {symbol!r}: {answer!r}"

    # The anchor has to demonstrate both lengths, or it only teaches one.
    assert min(len(a) for a in answers) < 80, "no short example in the anchor"
    assert max(len(a) for a in answers) > 200, "no explanation example in the anchor"
    # ...but not a paragraph. The anchor is imitated far more reliably than the rule is followed
    assert max(len(a) for a in answers) < 400, "the anchor teaches a paragraph"
    assert "{model}" in prompt and any("{model}" in a for a in answers)
    assert "start with the answer" in llm.REMINDER.lower()

    # The reminder is the last message in the context
    for name in ("REMINDER", "REMINDER_LOOK", "REMINDER_NOLOOK", "REMINDER_SEEN", "REMINDER_NOSHOT"):
        text = getattr(llm, name)
        assert "never something to quote" in text, f"{name} does not fence itself off"
        assert "numbers as words" not in text, f"{name} still asks for speech spelling"

    # The marker rule lives in the reminders, never in the editable prompt
    assert "[look]" not in prompt, "screen machinery leaked into the editable prompt"

    # And the prompt must not have an opinion about *seeing* either.
    assert "screen" not in prompt, "the prompt has an opinion about seeing the screen"

    # No prohibition-shaped capability line. The one that shipped
    for banned in ("cannot", "never offer", "you can't", "unable to"):
        assert banned not in prompt, f"prohibition-shaped capability line is back: {banned!r}"
    assert "draft the reply" in prompt, "the prompt no longer says what to do instead"
    assert any("here's the note" in a.lower() for a in answers), (
        "no anchor demonstrates writing something out instead of refusing"
    )

    # Nothing about spelling words out phonetically.
    for banned in ("dot pie", "ninety eight degrees", "spell out numbers"):
        assert banned not in prompt, f"the model is being asked to phoneticise again: {banned!r}"
    assert not any("dot pie" in a or " dot com" in a for a in answers), (
        "an anchor still demonstrates speech spelling, which outweighs the rule"
    )

    # No two action anchors the same shape. One example is a template
    acted = [a.split("]", 1)[1].strip() for _, a in llm.ANCHOR_ACT]
    assert len(acted) >= 4, "not enough action examples to show a range"
    openers = [line.split()[0].lower().strip(",.") for line in acted]
    assert len(set(openers)) == len(openers), f"action anchors open alike: {openers}"
    assert max(len(x) for x in acted) > 2 * min(len(x) for x in acted), (
        "every action anchor is the same length, so they teach one shape"
    )
    assert "one short sentence" not in llm._ACT_RULE, (
        "the rule prescribes a shape again, which is what made replies canned"
    )

    # One length for everything now: a paragraph read out loud is tiring.
    assert "three to six" not in prompt, "the old paragraph-length rule is back"
    assert "two or three sentences" in prompt, "no length rule in the prompt"
    assert "exactly [look]" in llm.REMINDER_LOOK and "[look]" not in llm.REMINDER_SEEN

    # Superseding only works on an exact match
    assert prompt not in config.OLD_PROMPTS, "current prompt listed as superseded"
    # Wording that was only ever ours is *cleared*, not rewritten
    old = config.OLD_PROMPTS[-1]
    migrated = config.validate(config.migrate({"system_prompt": old}))
    assert migrated["system_prompt"] == "", "an old default was left in the box"
    # Including the one that was the default until it moved into the code.
    carried = config.validate(config.migrate({"system_prompt": prompt}))
    assert carried["system_prompt"] == "", "a copy of CORE was left in the box"
    mine = "talk like a pirate"
    kept = config.validate(config.migrate({"system_prompt": mine}))
    assert kept["system_prompt"] == mine, "a custom prompt was overwritten"
    # ...and it rides after CORE rather than instead of it.
    both = llm.persona({"system_prompt": mine, "llm": {"model": "m"}})
    assert both.startswith(prompt[:40]) and both.endswith(mine), both[-80:]
    print(f"ok  prompt/anchor/reminder agree, {len(config.OLD_PROMPTS)} old prompts retire")


async def check_temperature() -> None:
    """Temperature reaches both adapters, and Anthropic's lower ceiling holds."""
    from mellowd import config, llm

    cfg = config.validate(config.migrate({}))
    assert cfg["llm"]["temperature"] == 0.3, cfg["llm"]["temperature"]
    assert "temperature" in llm._settings(cfg), "temperature never reaches an adapter"

    for bad in (-0.1, 2.5, "warm", True):
        try:
            config.validate({**cfg, "llm": {**cfg["llm"], "temperature": bad}})
        except ValueError:
            continue
        raise AssertionError(f"temperature {bad!r} was accepted")

    # An OpenAI-legal 1.6 must not 400 the moment someone picks Anthropic.
    hot = {**cfg, "llm": {**cfg["llm"], "temperature": 1.6, "provider": "anthropic"}}
    sent = await _capture_anthropic_payload(llm._settings(hot))
    assert sent["temperature"] == 1.0, f"not clamped for anthropic: {sent}"
    print("ok  temperature is sent, validated, and clamped for anthropic")


async def _capture_anthropic_payload(section: dict) -> dict:
    """Run the Anthropic adapter against a socket that records what we sent."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from mellowd import llm

    seen: dict = {}
    reply = (
        b'data: {"type":"content_block_delta",'
        b'"delta":{"type":"text_delta","text":"ok"}}\n\n'
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            seen.update(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 8793), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        probe = {**section, "api_key": "test", "base_url": "http://127.0.0.1:8793/v1"}
        async for _ in llm._anthropic(probe, []):
            pass
    finally:
        server.shutdown()
    return seen


def check_elevenlabs() -> None:
    """The ElevenLabs adapter, against a socket that records what we sent."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import numpy as np

    from mellowd import config, tts, wav

    seen = {}
    tone = np.sin(np.linspace(0, 400, 12_000)).astype(np.float32)
    body = wav.encode(tone, 24_000)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["headers"] = dict(self.headers)
            length = int(self.headers.get("content-length", 0))
            seen["body"] = json.loads(self.rfile.read(length))
            self.send_response(200)
            self.send_header("content-type", "audio/wav")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 8792), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = dict(config.DEFAULTS)
        cfg["tts"] = {
            **config.DEFAULTS["tts"],
            "mode": "cloud",
            "provider": "elevenlabs",
            "base_url": "http://127.0.0.1:8792/v1",
            "api_key": "test-key",
            "model": "eleven_flash_v2_5",
            "voice": "21m00Tcm4TlvDq8ikWAM",
            # Deliberately outside ElevenLabs' 0.7-1.2 window.
            "speech_speed": 2.0,
        }
        samples, rate = tts.synth("hello", cfg)
    finally:
        server.shutdown()

    assert seen["path"].startswith("/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"), seen["path"]
    assert f"output_format={tts.ELEVEN_FORMAT}" in seen["path"], seen["path"]
    assert seen["headers"].get("xi-api-key") == "test-key", "wrong auth header"
    assert "Authorization" not in seen["headers"], "sent a bearer token to elevenlabs"
    assert seen["body"]["model_id"] == "eleven_flash_v2_5", seen["body"]
    speed = seen["body"]["voice_settings"]["speed"]
    assert speed == tts.ELEVEN_SPEED[1], f"speed not clamped: {speed}"
    assert rate == 24_000 and np.max(np.abs(samples)) > 0.01, "decoded silence"
    print("ok  elevenlabs adapter: voice in path, xi-api-key, speed clamped")


def check_openrouter() -> None:
    """OpenRouter is a preset, but it is the one that cannot serve wav."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import numpy as np

    from mellowd import config, tts

    seen = {}
    tone = (np.sin(np.linspace(0, 400, 12_000)) * 20_000).astype("<i2").tobytes()
    served = {"content_type": "audio/pcm"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.send_header("content-type", served["content_type"])
            self.send_header("content-length", str(len(tone)))
            self.end_headers()
            self.wfile.write(tone)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 8794), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cfg = dict(config.DEFAULTS)
    cfg["tts"] = {
        **config.DEFAULTS["tts"],
        "mode": "cloud",
        "provider": "openrouter",
        "base_url": "http://127.0.0.1:8794/api/v1",
        "api_key": "test-key",
        "model": "fish-audio/s2.1-pro-free:free",
        "voice": "",
    }
    try:
        samples, rate = tts.synth("hello", cfg)
        # A provider that says nothing about the rate gets the documented guess.
        assert rate == tts.CLOUD_PCM_RATE, rate

        # ...and one that does say is believed
        served["content_type"] = "audio/pcm; rate=44100"
        _, rate = tts.synth("hello", cfg)
        assert rate == 44_100, rate
    finally:
        server.shutdown()

    assert seen["path"] == "/api/v1/audio/speech", seen["path"]
    assert seen["auth"] == "Bearer test-key", seen["auth"]
    assert seen["body"]["response_format"] == "pcm", seen["body"]
    assert "voice" not in seen["body"], "sent an empty voice instead of omitting it"
    assert len(samples) == 12_000, len(samples)
    assert np.max(np.abs(samples)) > 0.01, "decoded silence"
    print("ok  openrouter asks for pcm, and raw pcm decodes at the stated rate")


def check_stream_failure() -> None:
    """A microphone that never started must not look open, and must not be final."""
    from mellowd import config, stt

    attempts = []

    class Dead:
        """Advertises fine, refuses to start — the shape of a stale index."""

        def __init__(self, *, device=None, **kw):
            attempts.append(device)

        def start(self):
            raise stt.sd.PortAudioError("refused")

        def stop(self):
            pass

        def close(self):
            pass

    cfg = config.validate(dict(config.DEFAULTS))
    rec = stt.Recorder(cfg)
    real, stt.sd.InputStream = stt.sd.InputStream, Dead
    delay, stt.OPEN_RETRY_DELAY = stt.OPEN_RETRY_DELAY, 0
    try:
        raised = False
        try:
            rec.open()
        except RuntimeError:
            # A sentence for the bubble, not a stack of driver GUIDs
            raised = True
        assert raised, "a microphone that cannot start must not open quietly"

        assert rec._stream is None, "left a stream that was never started"
        assert rec._device is None, "left a name that would take the early return"
        # Every candidate tried once per round, for OPEN_RETRIES rounds
        per_round = len(set(attempts))
        assert per_round >= 1, attempts
        assert len(attempts) == stt.OPEN_RETRIES * per_round, attempts

        # reopen() is opportunistic recovery mid-handler
        rec._stream = Dead()
        rec.reopen()
        assert rec._stream is None, "a failed reopen left a dead stream behind"
    finally:
        stt.sd.InputStream = real
        stt.OPEN_RETRY_DELAY = delay

    # reopen() on a released microphone must stay released
    idle = stt.Recorder(cfg)
    idle.reopen()
    assert idle._stream is None, "reopen() opened a microphone nobody asked for"
    print(f"ok  a failed microphone stays closed after {stt.OPEN_RETRIES}x{per_round} start attempts")


def check_warm_signatures() -> None:
    """warm_models can actually call both loaders."""
    import inspect

    from mellowd import stt, tts

    for name, loader in (("stt.load", stt.load), ("tts.load", tts.load)):
        try:
            inspect.signature(loader).bind(progress=lambda *a: None)
        except TypeError as e:
            raise AssertionError(f"warm_models cannot call {name}: {e}") from None

    # And the call site really does use the keyword. Positional is what broke.
    source = inspect.getsource(__import__("mellowd.main", fromlist=["x"]).warm_models)
    assert "progress=_progress_cb(name)" in source, (
        "warm_models is passing the loaders positionally again"
    )
    print("ok  both speech loaders can be warmed the way warm_models calls them")


def check_warmup() -> None:
    """The keeper: probe until Windows relents, quietly, and respect the nap."""
    from mellowd import main, stt

    class Rec:
        def __init__(self, failures=0, on_open=None):
            self.failures, self.on_open = failures, on_open
            self.opens, self.closed = 0, False

        def open(self, quiet=False):
            self.opens += 1
            if self.opens <= self.failures:
                raise RuntimeError("refused")
            if self.on_open:
                self.on_open()

        def close(self):
            self.closed = True

    def session(rec) -> main.Session:
        s = object.__new__(main.Session)
        s.recorder, s.awake = rec, True
        return s

    refreshes = []
    real_delay, main.WARM_RETRY_SECONDS = main.WARM_RETRY_SECONDS, 0
    real_slow, main.WARM_SLOW_SECONDS = main.WARM_SLOW_SECONDS, 0
    real_refresh, stt.refresh_devices = stt.refresh_devices, lambda: refreshes.append(1) or True
    try:
        # Refused far past the old 6-try cap, then success — no give-up.
        s = session(Rec(failures=40))
        assert s._warm_open()
        assert s.recorder.opens == 41, s.recorder.opens
        assert not s.recorder.closed, "a successful warm-up must hold the mic"
        assert len(refreshes) == 1, "the portaudio refresh must run exactly once"

        # Napping mid-wait stops the probing without an exception.
        rec = Rec(failures=10 ** 9)
        s = session(rec)
        stop = s
        rec.on_open = None
        original_open = rec.open

        def open_and_count(quiet=False):
            if rec.opens == 4:  # the nap arrives between probes
                stop.awake = False
            original_open(quiet)

        rec.open = open_and_count
        assert not s._warm_open()
        assert rec.opens <= 6, f"kept probing after the nap: {rec.opens}"

        # Napped while the open was in flight: the mic must be released
        rec = Rec()
        s = session(rec)
        rec.on_open = lambda: setattr(s, "awake", False)
        assert not s._warm_open()
        assert rec.closed, "an open that lands after the nap must be undone"

        # Already napping: no probe at all.
        s = session(Rec())
        s.awake = False
        assert not s._warm_open()
        assert s.recorder.opens == 0, "warmed up a napping recorder"
    finally:
        main.WARM_RETRY_SECONDS = real_delay
        main.WARM_SLOW_SECONDS = real_slow
        stt.refresh_devices = real_refresh
    print("ok  mic keeper probes without a cap, refreshes once, respects the nap")


def check_tts() -> tuple:
    """Synthesise for real."""
    import numpy as np

    from mellowd import tts

    samples, rate = tts.synth(SPEECH_CHECK)
    # Not "== SAMPLE_RATE": that is Kokoro's number
    assert 8_000 <= rate <= 48_000, f"implausible sample rate {rate}"
    assert len(samples) > 0, "kokoro returned no samples"
    assert np.max(np.abs(samples)) > 0.01, "kokoro returned silence"
    secs = len(samples) / rate
    assert secs > 0.3, f"implausibly short: {secs:.2f}s"
    print(f"ok  {tts.backend()} synthesised {secs:.2f}s of audio at {rate}Hz")
    return samples, rate


def check_resample() -> None:
    """Numeric logic, so it gets a real assertion: a 440Hz tone must still be a 440Hz tone at the new"""
    import numpy as np

    from mellowd import stt

    for src in (44_100, 48_000, 16_000):
        t = np.arange(src) / src
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        out = stt.resample(tone, src, stt.SAMPLE_RATE)

        assert abs(len(out) - stt.SAMPLE_RATE) <= 1, f"{src}: wrong length {len(out)}"
        freqs = np.fft.rfftfreq(len(out), 1 / stt.SAMPLE_RATE)
        peak = freqs[np.argmax(np.abs(np.fft.rfft(out)))]
        assert abs(peak - 440) < 5, f"{src}: pitch shifted to {peak:.0f}Hz"

        # Preserve amplitude as well as pitch.
        loud_in, loud_out = float(np.abs(tone).max()), float(np.abs(out).max())
        assert abs(loud_out - loud_in) < 0.02, (
            f"{src}: level changed {loud_in:.3f} -> {loud_out:.3f}"
        )
    print("ok  resample preserves pitch, length and level (44.1k/48k/16k)")


def _with(cfg: dict, name: str, **fields) -> dict:
    """A copy of cfg with one capability's fields overridden."""
    return {**cfg, name: {**cfg[name], **fields}}


def check_config() -> None:
    from mellowd import config

    base = config.validate(dict(config.DEFAULTS))

    # Every preset of every capability must survive a round trip.
    for name in config.CAPABILITIES:
        for provider, preset in config.PRESETS[name].items():
            checked = config.validate(
                _with(
                    base,
                    name,
                    mode="cloud",
                    provider=provider,
                    base_url=preset["base_url"] or "http://127.0.0.1:9999/v1",
                    api_key="test" if preset["requires_key"] else "",
                )
            )
            assert checked[name]["provider"] == provider, (name, provider)
            assert checked[name]["base_url"].startswith(("http://", "https://"))

    # Mode follows the preset. "Cloud + Ollama" was savable
    snapped = config.validate(_with(base, "llm", mode="cloud", provider="ollama"))
    assert snapped["llm"]["mode"] == "local", snapped["llm"]["mode"]
    # The settings window builds the cloud list from these
    for name in config.CAPABILITIES:
        assert any(not p["local"] for p in config.PRESETS[name].values()), name
    assert any(p["local"] for p in config.LLM_PRESETS.values()), "no on-device llm"

    for name in config.CAPABILITIES:
        try:
            config.validate(_with(base, name, base_url="file:///secret"))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name}: unsafe provider URL was accepted")

    # Every key must be blanked, not just the LLM's.
    loaded = dict(base)
    for name in config.CAPABILITIES:
        loaded = _with(loaded, name, api_key="secret")
    hidden = config.redacted(loaded)
    for name in config.CAPABILITIES:
        assert hidden[name]["api_key"] == "", f"{name} key leaked"
        assert hidden[name]["has_api_key"] is True, name
    print(f"ok  presets, validation and key redaction for {len(config.CAPABILITIES)} capabilities")


def check_migration() -> None:
    """The pre-step-9 flat config must survive, above all its API key."""
    from mellowd import config

    old = {
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": "sk-from-the-old-schema",
        "model": "llama-3.3-70b",
        "max_tokens": 300,  # the superseded default, should be lifted
        "stt_model": "medium.en",
        "input_device": 3,
        "input_channel": 1,
        "voice": "af_bella",
        "speech_speed": 1.25,
        "speak": False,
        "system_prompt": "be brief",
    }
    new = config.validate(config.migrate(dict(old)))
    assert new["llm"]["api_key"] == old["api_key"], "API KEY LOST ON MIGRATION"
    assert new["llm"]["provider"] == "groq" and new["llm"]["mode"] == "cloud"
    # Against DEFAULTS, not a literal
    assert (
        new["llm"]["max_tokens"] == config.DEFAULTS["llm"]["max_tokens"]
    ), "superseded default not lifted all the way to today's"
    assert new["stt"]["local_model"] == "medium.en"
    # A microphone is saved by name now.
    assert new["stt"]["input_device"] is None, new["stt"]["input_device"]
    assert "input_channel" not in new["stt"], "the channel picker is gone"

    assert new["stt"]["mode"] == "local", "old configs had no cloud speech"
    assert new["tts"]["local_voice"] == "af_bella"
    assert new["tts"]["speech_speed"] == 1.25 and new["tts"]["speak"] is False
    assert new["system_prompt"] == "be brief"
    assert "provider" not in new, "flat keys left behind alongside nested ones"

    # Idempotent: an already-migrated config must come back unchanged.
    assert config.validate(config.migrate(dict(new))) == new, "migrate is not idempotent"

    # A local LLM stays local.
    local = config.validate(config.migrate({**old, "provider": "ollama", "api_key": ""}))
    assert local["llm"]["mode"] == "local", local["llm"]["mode"]
    print("ok  flat config migrates, keeps its key, and is idempotent")


def check_wav() -> None:
    """Cloud speech rides on this in both directions, so it gets real numbers."""
    import numpy as np

    from mellowd import wav

    t = np.arange(16_000, dtype=np.float32) / 16_000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    back, rate = wav.decode(wav.encode(tone, 16_000))
    assert rate == 16_000, rate
    assert back.shape == tone.shape, (back.shape, tone.shape)
    # 16-bit quantisation is the only loss allowed
    assert float(np.max(np.abs(back - tone))) < 1e-3

    # Full scale must not wrap around to the opposite sign.
    ends, _ = wav.decode(wav.encode(np.array([-1.0, 1.0], np.float32), 16_000))
    assert ends[0] < -0.99 and ends[1] > 0.99, ends
    print("ok  wav round trip is sample-accurate and doesn't clip-wrap")


def check_audio_selection() -> None:
    import numpy as np

    from mellowd import stt

    t = np.arange(stt.SAMPLE_RATE, dtype=np.float32) / stt.SAMPLE_RATE
    quiet = 0.01 * np.sin(2 * np.pi * 220 * t)
    speech = 0.08 * np.sin(2 * np.pi * 440 * t)
    selected, channel = stt.choose_channel(np.column_stack([quiet, speech]))
    assert channel == 1
    assert np.max(np.abs(selected)) > 0.07
    gained, gain = stt.apply_quiet_gain(selected)
    assert 1 < gain <= stt.MAX_GAIN
    assert np.max(np.abs(gained)) <= 1

    # The real failure
    hiss = 0.09 * np.sin(2 * np.pi * 220 * t)
    burst = np.zeros_like(t)
    voice = slice(0, len(t) // 10)
    burst[voice] = 0.25 * np.sin(2 * np.pi * 440 * t[voice])
    stacked = np.column_stack([hiss, burst])

    whole_take = np.sqrt(np.mean(np.square(stacked, dtype=np.float64), axis=0))
    assert whole_take[0] > whole_take[1], "the regression case stopped regressing"

    _, channel = stt.choose_channel(stacked)
    assert channel == 1, "picked the noisy capsule over the one with speech in it"
    print("ok  microphone channel selection and bounded quiet gain")


def check_devices() -> None:
    """The settings window must offer one line per real microphone."""
    from mellowd import stt

    devices = stt.input_devices()
    names = [d["name"] for d in devices]
    assert names, "no input devices at all"
    assert len(names) == len(set(names)), f"same microphone listed twice: {names}"
    assert all("index" not in d for d in devices), "still exposing positional indices"
    assert sum(d["default"] for d in devices) <= 1, "more than one default"
    # Whatever "let Windows decide" resolves to has to be something real.
    assert stt._resolve(None) is None or isinstance(stt._resolve(None), int)
    assert stt._resolve("a microphone nobody has") == stt._resolve(None)
    print(f"ok  {len(names)} microphone(s) offered, saved by name: {names[0]}")


def check_errors() -> None:
    """A provider saying no has to reach the user as something actionable."""
    import httpx

    from mellowd import errors

    body = '{"error":{"message":"Rate limit reached for model openai/gpt-oss-120b"}}'
    limited = str(errors.provider_error(429, body, "groq"))
    assert "rate limiting" in limited, limited
    # The raw body belongs in the log, not the speech bubble.
    assert "Rate limit reached" not in limited, limited

    assert "api key" in str(errors.provider_error(401, "", "groq"))
    assert "out of credit" in str(errors.provider_error(402, "", "openrouter"))
    missing = str(errors.provider_error(404, "", "groq", "openai/gpt-oss-120b"))
    assert "openai/gpt-oss-120b" in missing, missing

    # Ollama not running is the most common failure of all, and "ConnectError" told nobody to go
    request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    unreachable = errors.message(httpx.ConnectError("refused", request=request))
    assert "can't reach 127.0.0.1" in unreachable, unreachable

    # Anything already worded passes through untouched
    assert errors.message(RuntimeError(limited)) == limited
    print("ok  provider failures read as sentences, raw bodies stay in the log")


def check_reminders() -> None:
    """The one piece of clock arithmetic in the feature, so it gets real asserts."""
    from datetime import datetime, timedelta

    from mellowd import remind

    at = lambda h, m: datetime(2026, 3, 14, h, m)

    once = {"id": "a", "time": "09:00", "text": "standup", "daily": False}
    daily = {"id": "b", "time": "09:00", "text": "stretch", "daily": True}

    # Nothing is due before its minute arrives.
    fired, keep = remind.due([once], at(8, 59))
    assert fired == [], fired
    assert len(keep) == 1, keep

    # A one-off fires once, and firing is what retires it.
    fired, keep = remind.due([once], at(9, 0))
    assert [f["text"] for f in fired] == ["standup"], fired
    assert keep == [], keep

    # A daily fires, is kept
    fired, keep = remind.due([daily], at(9, 0))
    assert len(fired) == 1, fired
    assert keep[0]["last_fired"] == "2026-03-14", keep
    again, keep = remind.due(keep, at(9, 0))
    assert again == [], again
    # ...but tomorrow it fires again.
    tomorrow, _ = remind.due(keep, at(9, 0) + timedelta(days=1))
    assert len(tomorrow) == 1, tomorrow

    # Closed over the moment: inside the grace window it still fires
    late, _ = remind.due([once], at(9, 0) + remind.GRACE - timedelta(seconds=1))
    assert len(late) == 1, late
    missed, keep = remind.due([once], at(9, 0) + remind.GRACE + timedelta(minutes=1))
    assert missed == [], missed
    assert len(keep) == 1, "a missed one-off should wait, not vanish"

    # A hand-edited file must not stop the healthy entries from firing.
    junk = ["nonsense", {"time": "25:00", "text": "bad hour"}, {"time": "9", "text": "no colon"},
            {"time": "09:00", "text": "   "}, once]
    fired, keep = remind.due(junk, at(9, 0))
    assert [f["text"] for f in fired] == ["standup"], fired
    assert keep == [], keep

    # Round-trips through storage are normalised the same way
    stored = remind.normalize([{"time": "7:5", "text": "  padded  "}])
    assert stored[0]["time"] == "07:05", stored
    assert stored[0]["text"] == "padded", stored
    assert stored[0]["id"], stored
    try:
        remind.normalize({"not": "a list"})
    except ValueError:
        pass
    else:
        raise AssertionError("a non-list should be refused, not coerced")

    print("ok  reminders fire once, dailies re-arm, stale ones are written off")


def check_stt_speech(samples, rate: int) -> None:
    """Transcribe actual generated speech; silence only proves kernels load."""
    from mellowd import config, stt

    audio = stt.resample(samples, rate, stt.SAMPLE_RATE)
    text = stt.transcribe(audio)
    words = set(text.lower().replace(".", "").split())
    expected = {"mellow", "hear", "sentence", "clearly"}
    engine = config.load()["stt"]["local_model"]
    assert len(words & expected) >= 3, f"{engine} misheard speech check: {text!r}"
    print(f"ok  {engine} on {stt.backend()} recognised speech: {text!r}")


def check_preroll() -> None:
    """The take must start *before* the press, or the first word is lost again."""
    import numpy as np

    from mellowd import config, stt

    rate, block = 48_000, 512
    # A fixed cfg, so the check doesn't depend on which microphone is saved.
    cfg = config.validate(dict(config.DEFAULTS))
    rec = stt.Recorder(cfg)
    rec._stream, rec._device = object(), cfg["stt"].get("input_device")
    rec._rate, rec._preroll = rate, int(rate * stt.PREROLL_SECONDS)

    for _ in range(rate // block):  # a full second of speech before the press
        rec._capture(np.full((block, 1), 0.5, np.float32), block, None, None)

    held = rec._ring_samples
    assert held >= rec._preroll, f"ring starved: {held} < {rec._preroll}"
    assert held < rec._preroll + 2 * block, f"ring grew unbounded: {held}"

    rec.start()  # press and release in the same instant
    captured = np.concatenate(rec._frames)
    rec._armed = False
    assert len(captured) >= rec._preroll, f"pre-roll lost: {len(captured)}"
    assert (captured == 0.5).all(), "pre-roll is not the audio that preceded it"
    print(f"ok  {len(captured) / rate:.2f}s of pre-roll survives an instant press")


@contextmanager
def _scratch_log():
    """Point the session log at a throwaway directory and put it back after."""
    import tempfile

    from mellowd import sessions

    saved = (sessions.SESSIONS_DIR, sessions.INDEX_PATH, sessions.MEDIA_DIR)
    saved_open, saved_current = dict(sessions._open), sessions._current
    with tempfile.TemporaryDirectory() as tmp:
        sessions.SESSIONS_DIR = Path(tmp) / "sessions"
        sessions.INDEX_PATH = sessions.SESSIONS_DIR / "index.jsonl"
        sessions.MEDIA_DIR = sessions.SESSIONS_DIR / "media"
        sessions._open.clear()
        sessions._current = ""
        try:
            yield Path(tmp)
        finally:
            sessions.SESSIONS_DIR, sessions.INDEX_PATH, sessions.MEDIA_DIR = saved
            sessions._open.clear()
            sessions._open.update(saved_open)
            sessions._current = saved_current


@contextmanager
def _scratch_config(cfg: dict):
    """Point config at a throwaway file with the given contents, then restore."""
    import tempfile

    from mellowd import config

    saved = config.CONFIG_PATH
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        config.CONFIG_PATH = path
        try:
            yield path
        finally:
            config.CONFIG_PATH = saved


async def check_pet_only() -> None:
    """Step 8's just-the-pet mode: a config with no brain saves, warm-up loads neither engine, and the"""
    from mellowd import config, main

    pet = {
        **config.DEFAULTS,
        "ai_enabled": False,
        "llm": {**config.DEFAULTS["llm"], "model": ""},
    }
    saved = config.validate(pet)
    assert saved["ai_enabled"] is False and saved["llm"]["model"] == "", saved["llm"]
    # With the brain on, the same empty model is still a mistake.
    try:
        config.validate({**config.DEFAULTS, "llm": {**config.DEFAULTS["llm"], "model": ""}})
        raise AssertionError("an empty model passed with ai_enabled on")
    except ValueError:
        pass
    print("ok  pet-only config validates with no model named")

    with _scratch_config(pet):
        assert main.standby() is True
    with _scratch_config({**config.DEFAULTS, "ai_enabled": True}):
        assert main.standby() is False
    # No file at all is the first-run case: standby until the wizard finishes.
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        real = config.CONFIG_PATH
        config.CONFIG_PATH = Path(tmp) / "absent.json"
        try:
            assert main.standby() is True
        finally:
            config.CONFIG_PATH = real
    print("ok  standby: absent config or ai off, never an on-brain config")

    # Warm-up: stub both loaders, prove neither is called.
    with _scratch_config(pet):
        called = []
        real_stt, real_tts = main.stt.load, main.tts.load
        main.stt.load = lambda cfg=None, progress=None: called.append("stt")
        main.tts.load = lambda progress=None: called.append("tts")
        try:
            await asyncio.wait_for(main.warm_models(), 10)
        finally:
            main.stt.load, main.tts.load = real_stt, real_tts
        assert called == [], called
    print("ok  pet-only warm-up loads neither engine")

    # The hotkey in pet-only: one bubble line, back to idle, recorder closed.
    class FakeRecorder:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            return None

        def close(self):
            pass

    class FakeWS:
        def __init__(self):
            self.sent = []

        async def send_text(self, raw):
            self.sent.append(json.loads(raw))

    class FakeSpeaker:
        def begin(self):
            pass

        async def speak(self, sentence):
            pass

        async def stop(self):
            pass

        async def finish(self):
            pass

    with _scratch_config(pet):
        s = object.__new__(main.Session)
        s.ws = FakeWS()
        s.history = []
        s.destination = None
        s.speaker = FakeSpeaker()
        s.turn = None
        s.recorder = FakeRecorder()
        await main.handle(s, {"type": "ptt_start"})
        kinds = [(m["type"], m.get("state")) for m in s.ws.sent]
        assert ("reply_chunk", None) in [(k, v) for k, v in kinds], kinds
        assert ("state", "idle") in kinds, kinds
        assert s.recorder.started is False, "the recorder opened in pet-only"
        # And the typed path answers the same way.
        s.ws.sent.clear()
        await main.handle(s, {"type": "text", "text": "hello"})
        kinds = [m["type"] for m in s.ws.sent]
        assert "reply_chunk" in kinds and ("state", "idle") in [
            (m["type"], m.get("state")) for m in s.ws.sent
        ], s.ws.sent
        assert s.turn is None, "pet-only started a turn"
    print("ok  pet-only hotkey: one line, idle, microphone never opened")

    # The progress feed: bytes only ever go up across files
    main._download_progress["tts"].update(
        state="idle", name="", done=0, total=0, error="", base=0
    )
    cb = main._progress_cb("tts")
    cb("kokoro-v1.0.onnx", 100, 100)
    first = dict(main._download_progress["tts"])
    cb("voices-v1.0.bin", 50, 200)
    second = dict(main._download_progress["tts"])
    assert first["done"] == 100 and second["done"] == 150, (first, second)
    assert second["total"] == 300 and second["state"] == "running", second

    # A local voice preview must not claim readiness after only one of Kokoro's two files
    import tempfile as _model_tempfile
    real_models_dir = main.tts.models.MODELS_DIR
    with _model_tempfile.TemporaryDirectory() as tmp:
        main.tts.models.MODELS_DIR = Path(tmp)
        try:
            assert not main.tts.local_available()
            (Path(tmp) / "kokoro-v1.0.onnx.part").write_bytes(b"partial")
            assert not main.tts.local_available()
            (Path(tmp) / "kokoro-v1.0.onnx").write_bytes(b"model")
            assert not main.tts.local_available()
            (Path(tmp) / "voices-v1.0.bin").write_bytes(b"voices")
            assert main.tts.local_available()
            assert (await main.available_models()) == {"tts": True}
        finally:
            main.tts.models.MODELS_DIR = real_models_dir
    print("ok  local voice availability: both complete Kokoro files required")

    real_stt_load = main.stt.load
    main.stt.load = lambda cfg=None, progress=None: (_ for _ in ()).throw(
        RuntimeError("the disk filled up")
    )
    try:
        main._download_progress["stt"].update(
            state="running", name="", done=0, total=0, error="", base=0
        )
        await main._run_download("stt", config.DEFAULTS)
    finally:
        main.stt.load = real_stt_load
    failed = main._download_progress["stt"]
    assert failed["state"] == "failed", failed
    assert "disk filled up" in failed["error"] and "\n" not in failed["error"], failed

    # First-run setup has no config file yet, so the explicit download route must not consult standby().
    selected = config.validate(config.DEFAULTS)
    calls = []
    real_standby, real_stt_load, real_tts_load = (
        main.standby,
        main.stt.load,
        main.tts.load,
    )
    main.standby = lambda: (_ for _ in ()).throw(
        AssertionError("the first-run download route consulted standby")
    )
    main.stt.load = lambda cfg=None, progress=None: calls.append(("stt", cfg, progress))
    main.tts.load = lambda progress=None, cfg=None: calls.append(("tts", cfg, progress))
    try:
        for which in ("stt", "tts"):
            await main.start_model_download({"which": which, "settings": selected})
            await main._download_tasks[which]
    finally:
        main.standby, main.stt.load, main.tts.load = (
            real_standby,
            real_stt_load,
            real_tts_load,
        )
    assert [call[0] for call in calls] == ["stt", "tts"], calls
    assert all(call[1]["stt"]["mode"] == "local" for call in calls), calls
    assert all(call[2] is not None for call in calls), calls
    main._download_progress["stt"].update(state="idle", error="")
    main._download_progress["tts"].update(state="idle", name="", done=0, total=0, base=0)
    print("ok  first-run downloads: candidate reaches STT and TTS with real progress")
    print("ok  download progress: monotonic bytes, failure as one sentence")


def check_sessions() -> None:
    """The event log round-trips, survives a torn tail, honours the toggle, segments after silence, keeps"""
    import threading
    from datetime import datetime, timedelta, timezone

    from mellowd import config, sessions

    try:
        config.validate({**config.DEFAULTS, "remember_conversations": "yes"})
    except ValueError:
        pass
    else:
        raise AssertionError("remember_conversations accepted a non-bool")

    real_load = config.load

    def remembering(flag: bool):
        return lambda: {**real_load(), "remember_conversations": flag}

    with _scratch_log():
        try:
            config.load = remembering(True)

            # Round trip: both events land, sequenced
            sessions.record("user_said", text="hello world this is mellow")
            sessions.record(
                "assistant_said", text="hi", model="test-model",
                provider="ollama", base_url="http://x", aborted=False,
            )
            entries = sessions.list_sessions()
            assert len(entries) == 1, entries
            assert entries[0]["title"] == "hello world this is mellow", entries
            assert entries[0]["turns"] == 1 and entries[0]["events"] == 2, entries
            assert entries[0]["kind"] == "conversation" and not entries[0]["parent"]
            events = sessions.read(entries[0]["id"])
            assert [e["type"] for e in events] == [
                "session_start", "user_said", "assistant_said"
            ], events
            assert [e["seq"] for e in events] == [0, 1, 2], events
            assert events[2]["model"] == "test-model" and not events[2]["aborted"]

            # A crash mid-write leaves a partial final line
            with sessions._path(entries[0]["id"]).open("a", encoding="utf-8") as f:
                f.write('{"v":1,"seq":9,"ts":"2026')
            again = sessions.read(entries[0]["id"])
            assert len(again) == 3, f"a torn tail corrupted the session: {again}"

            # ...and the next event must not fuse itself onto the tear.
            sessions.record("user_said", text="written after the crash")
            after = sessions.read(entries[0]["id"])
            assert len(after) == 4, f"the append fused onto the torn line: {after}"
            assert after[-1]["text"] == "written after the crash", after[-1]

            # Half an hour of silence starts a new session
            aged = sessions._current
            sessions._open[aged].last_ts -= timedelta(minutes=31)
            sessions.record("user_said", text="a fresh conversation begins here")
            entries = sessions.list_sessions()
            assert len(entries) == 2, entries
            first, second = sorted(entries, key=lambda e: e["started_at"])
            assert second["title"] == "a fresh conversation begins here"
            assert first["ended_at"], "the old session was never closed"
            # The boundary goes on disk either way
            closed = sessions.read(aged)[-1]
            assert closed["type"] == "session_ended", closed
            assert closed["reason"] == "silence", closed

            # Titles cut on a word boundary, not mid-word.
            sessions._open[sessions._current].last_ts -= timedelta(minutes=31)
            sessions.record(
                "user_said", text="what is the meaning of " + "extraordinary " * 10
            )
            top = sessions.list_sessions()[0]
            assert top["title"].startswith("what is the meaning of"), top
            assert len(top["title"]) <= 49, top["title"]

            # An agent's session (step 16) runs beside the conversation
            talk = sessions._current
            agent = sessions.open_session(kind="agent", parent=talk)
            assert sessions._current == talk, "an agent session stole the default"
            sessions._open[agent].last_ts -= timedelta(minutes=90)
            sessions.record("tool_call", session=agent, tool="click", args={"x": 1})
            assert sessions._current == talk
            agent_events = sessions.read(agent)
            assert [e["type"] for e in agent_events] == [
                "session_start", "tool_call"
            ], agent_events
            assert agent_events[0]["parent"] == talk
            assert agent_events[0]["kind"] == "agent"
            assert len(sessions.read(talk)) == 2, "the agent wrote into the conversation"

            # Two threads, two sessions: no interleaving, no lost counts.
            other = sessions.open_session(kind="agent", parent=talk)

            def hammer(target: str, tag: str) -> None:
                for _ in range(40):
                    sessions.record("tool_call", session=target, tag=tag)

            threads = [
                threading.Thread(target=hammer, args=(agent, "a")),
                threading.Thread(target=hammer, args=(other, "b")),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for target, tag, already in ((agent, "a", 1), (other, "b", 0)):
                logged = sessions.read(target)[1:]  # drop the session_start header
                assert len(logged) == 40 + already, (target, len(logged))
                assert {e.get("tag") for e in logged[already:]} == {tag}, target
                assert [e["seq"] for e in logged] == list(
                    range(1, 41 + already)
                ), target

            # A step-15 tool argument json can't serialise must cost one event
            sessions.record("tool_call", session=agent, args={"when": datetime.now()})
            assert sessions.read(agent)[-1]["type"] == "tool_call"

            # The index is a cache: lose it and every session comes back.
            expected = {e["id"] for e in sessions.list_sessions()}
            sessions.INDEX_PATH.unlink()
            rebuilt = sessions.list_sessions()
            assert {e["id"] for e in rebuilt} == expected, rebuilt
            assert sessions.INDEX_PATH.exists(), "the rebuild was never written back"
            recovered = next(e for e in rebuilt if e["id"] == agent)
            assert recovered["kind"] == "agent", recovered
            assert recovered["parent"] == talk, recovered

            # Toggle off: nothing is written at all.
            config.load = remembering(False)
            before = sessions.list_sessions()
            sessions.record("user_said", text="nobody should ever read this")
            assert sessions.list_sessions() == before, "the toggle did not stop writes"
            config.load = remembering(True)

            # Retention, clock one: screenshots go at a week.
            fresh, stale = sessions.media_path(), sessions.media_path()
            fresh.write_bytes(b"x")
            stale.write_bytes(b"x")
            old = time.time() - sessions.IMAGE_KEEP.total_seconds() - 60
            os.utime(stale, (old, old))
            sessions.sweep()
            assert fresh.exists(), "a fresh screenshot was swept"
            assert not stale.exists(), "an eight-day-old screenshot survived"
            assert len(sessions.list_sessions()) == len(before), "sweep ate live sessions"

            # Retention, clock two: text goes at a year.
            sessions.close()
            doomed = sessions.list_sessions()[0]["id"]
            stamp = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
            sessions._write_index(
                [
                    {**e, "started_at": stamp, "ended_at": stamp}
                    if e["id"] == doomed
                    else e
                    for e in sessions.list_sessions()
                ]
            )
            sessions.sweep()
            assert doomed not in {
                e["id"] for e in sessions.list_sessions()
            }, "a year-old session survived the sweep"

            # Clear takes the whole tree, screenshots included.
            survivor = sessions.list_sessions()[0]["id"]
            kept = sessions.media_path()
            kept.write_bytes(b"secret")
            assert sessions.clear() >= 2
            assert sessions.list_sessions() == [], "clear left the index populated"
            assert not kept.exists(), "clear left a screenshot on disk"
            assert not sessions.MEDIA_DIR.exists(), "clear left the media directory"
            assert sessions.read(survivor) is None, "clear left a session file behind"

            # And it recovers: the next event opens a fresh session.
            sessions.record("user_said", text="after the clear")
            assert len(sessions.list_sessions()) == 1
        finally:
            config.load = real_load

    print("ok  session log segments, isolates agents, rebuilds, sweeps and clears")


async def check_turn_logging() -> None:
    """A completed turn, a barged-in turn and a failed turn all land in the log, and a reconnect picks the"""
    from contextlib import suppress

    from mellowd import config, llm, main, sessions

    class FakeWS:
        async def send_text(self, text):
            pass

    class FakeSpeaker:
        # Mirrors the real Speaker: begin() is sync, the rest are awaited.
        def begin(self):
            pass

        async def speak(self, sentence):
            pass

        async def stop(self):
            pass

        async def finish(self):
            pass

    real_chat = llm.chat
    real_load = config.load
    # The check must not depend on the user's own toggle
    config.load = lambda: {**real_load(), "remember_conversations": True}

    def blank_session():
        s = object.__new__(main.Session)
        s.ws = FakeWS()
        s.history = []
        s.destination = None
        s.speaker = FakeSpeaker()
        return s

    async def chat(chunks, hang_at_end=False, boom=False):
        for chunk in chunks:
            yield chunk
        if boom:
            raise RuntimeError("the provider hung up")
        if hang_at_end:
            await asyncio.sleep(3600)  # cancelled long before this fires

    # main._pass picks its brain from the saved config
    from mellowd import agents

    real_agent_chat = agents.chat

    def brain(make):
        llm.chat = agents.chat = make

    with _scratch_log():
        try:
            # A turn that finishes: aborted comes out false.
            brain(lambda history, cfg, image=None: chat(["a short ", "answer."]))
            await main.answer(blank_session(), "say something")

            # A turn cut off mid-stream: whatever was said lands anyway.
            brain(
                lambda history, cfg, image=None: chat(
                    ["one ", "two ", "three "], hang_at_end=True
                )
            )
            task = asyncio.create_task(main.answer(blank_session(), "count"))
            await asyncio.sleep(0.2)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

            # A turn the provider kills
            brain(lambda history, cfg, image=None: chat(["half an "], boom=True))
            noisy = logging.getLogger("mellowd")
            noisy.disabled = True
            try:
                await main.run_turn(blank_session(), "explain something")
            finally:
                noisy.disabled = False

            logged = sessions.read(sessions.list_sessions()[0]["id"])
            said = [e for e in logged if e["type"] == "assistant_said"]
            assert len(said) == 2, said
            assert not said[0]["aborted"], said[0]
            assert said[0]["text"] == "a short answer.", said[0]
            assert said[1]["aborted"] and said[1]["text"] == "one two three ", said[1]
            failed = [e for e in logged if e["type"] == "turn_failed"]
            assert len(failed) == 1 and failed[0]["reason"], failed

            # A reconnect picks the conversation back up
            history, destination = sessions.resume()
            assert [m["role"] for m in history] == [
                "user", "assistant", "user", "assistant"
            ], history
            assert history[0]["content"] == "say something", history
            assert history[-1]["content"] == "one two three ", history
            cfg = real_load()["llm"]
            # The model is in the triple on purpose
            assert destination == (
                cfg["provider"], cfg["base_url"], cfg["model"]
            ), destination

            # The next turn must stay in the same session file.
            before = len(sessions.list_sessions())
            brain(lambda history, cfg, image=None: chat(["still ", "here."]))
            await main.answer(blank_session(), "are you there")
            assert len(sessions.list_sessions()) == before, "resume split the session"
            newest = sessions.read(sessions.list_sessions()[0]["id"])
            assert newest[-1]["text"] == "still here.", newest[-1]
            seqs = [e["seq"] for e in newest]
            assert seqs == sorted(set(seqs)), f"resume reused a seq: {seqs}"

            # "New conversation" has to survive a reload.
            ended = sessions._current
            sessions.close()
            assert sessions.read(ended)[-1]["type"] == "session_ended"
            assert sessions.read(ended)[-1]["reason"] == "user"
            assert sessions.resume() == ([], None), "a closed session came back"

            await main.answer(blank_session(), "starting over")
            assert sessions._current != ended, "the next turn reopened a closed session"
            assert len(sessions.list_sessions()) == before + 1, "no new session started"
        finally:
            llm.chat = real_chat
            agents.chat = real_agent_chat
            config.load = real_load

    print("ok  completed, barged-in and failed turns are logged, and resume picks up")


def check_vision() -> None:
    """The vision flag resolves, and the right screen rule reaches the request."""
    from mellowd import config, llm

    # The editable prompt carries no screen machinery at all.
    assert "[look]" not in config.DEFAULTS["system_prompt"]
    for clause in (llm.REMINDER_LOOK, llm.REMINDER_NOLOOK, llm.REMINDER_SEEN):
        assert "[look]" not in clause or clause is llm.REMINDER_LOOK
        assert "Reminder: you are mellow" in clause

    # Auto resolution over a spread of real model ids.
    cases = {
        "gpt-oss-120b": False,
        # False is the correct *guess*
        "gemma3:4b": False,
        "nemotron-3-super": False,
        "gemini-3.6-flash": True,
        "google/gemini-2.0-flash-001": True,
        "gpt-4o-mini": True,
        "openai/gpt-oss-120b": False,  # hint must not fire on "gpt-" alone
        "qwen2.5-vl:7b": True,
        "claude-sonnet-4": True,
        "glm-5.2": False,
        "llava:13b": True,
    }
    for name, want in cases.items():
        got = config.resolves_vision({"vision": "auto", "model": name})
        assert got == want, f"{name}: resolved {got}, wanted {want}"

    # Overrides beat the name every time.
    assert config.resolves_vision({"vision": "on", "model": "gpt-oss-120b"})
    assert not config.resolves_vision({"vision": "off", "model": "gpt-4o"})

    # Validation accepts only the three modes.
    cfg = config.validate(dict(config.DEFAULTS))
    assert cfg["llm"]["vision"] == "auto"
    try:
        config.validate({**config.DEFAULTS, "llm": {**cfg["llm"], "vision": "maybe"}})
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown vision mode was accepted")

    # The reminder each configuration produces.
    look = llm._settings({**cfg, "llm": {**cfg["llm"], "vision": "on"}})
    assert look["vision_ok"] is True
    nolook = llm._settings({**cfg, "llm": {**cfg["llm"], "vision": "off"}})
    assert nolook["vision_ok"] is False

    # The marker is taught by example, not only described
    def shown(**over) -> bool:
        turns = llm._anchored({"model": "m", "anchor": True, **over}, [])
        return any(t["content"] == llm.LOOK for t in turns)

    assert shown(vision_ok=True), "the look example never reaches a vision model"
    # Both phrasings. "can you see my screen?" is the one that failed in use
    asked = [q for q, _ in llm.ANCHOR_LOOK]
    assert any("can you see" in q for q in asked), asked
    assert len(asked) >= 2, "only one way of asking is demonstrated"
    assert not shown(vision_ok=False), "a blind model was shown how to ask for eyes"
    assert not shown(vision_ok=True, screen="seen"), "phase 2 was invited to ask again"
    # And it stays out of the plain anchor
    assert all(llm.LOOK not in a for _, a in llm.ANCHOR)
    print("ok  vision resolves per model, overrides win, rules stay out of the prompt")


def check_wants_screen() -> None:
    """The gate that decides whether a turn needs eyes."""
    from mellowd import capture

    look = (
        "can you see my screen?",
        "what is on my screen right now",
        "whats on the screen",
        "can you read this email",
        "explain this part on my screen",
        "what does this error say",
        "what am i looking at",
        "can you tell me what i am seeing",
        "read the top of that window for me",
        "describe this image for me",
        "look at this and tell me whats wrong",
        "what does this dialog mean",
        "summarise this page",
        "whats the error in this code",
        "take a look at my display",
        "read that message out to me",
        "what is this popup asking me",
        "can you see what im reading",
        "whats in front of me right now",
        "show me whats wrong with this form",
        # A pointer with no noun after it. These missed, so no screenshot was taken
        "can you explain this",
        "can you explain this?",
        "what is this",
        "whats this",
        "explain this to me",
        "describe this",
    )
    # The expensive half. A false positive costs one ignored screenshot
    answer = (
        "what is the capital of france",
        "explain why my laptop gets slow with lots of tabs",
        "what model are you running on",
        "set a reminder for 9am",
        "who are you",
        "whats the difference between memory and disk",
        "can you read this book you recommended",
        "tell me a joke",
        "how do i center a div",
        "what time is it in tokyo",
        "look into whether python has a switch statement",
        "see you later",
        "start a pomodoro",
        "what did you just say",
        "explain how memory paging works",
        "lets see what you can do",
        # The other side of the bare-pointer rule
        "explain this concept again",
        "can you explain this idea in simpler terms",
        "what is this thing you mentioned",
    )
    missed = [q for q in look if not capture.wants_screen(q)]
    fired = [q for q in answer if capture.wants_screen(q)]
    assert not missed, f"would not have looked: {missed}"
    assert not fired, f"would have looked for no reason: {fired}"

    # Case and punctuation are not signal — speech-to-text supplies neither reliably
    assert capture.wants_screen("WHAT IS ON MY SCREEN")
    assert capture.wants_screen("what is on my screen")

    print(f"ok  screen-intent gate: {len(look)} look, {len(answer)} don't, 0 wrong")


def check_wants_pointing() -> None:
    """Which questions earn a bone, and which only want words."""
    from mellowd import capture

    point = (
        "what should i click for opening files",
        "what do i click to export this",
        "where is the export button",
        "which menu has the settings",
        "how do i export this video",
        "what do i press to save this file",
        "where do i find the settings for this",
        "how do i change the font on this page",
        "show me the button to start a render",
        # Everything below is a sentence a real person actually said to Mellow and did not get a bone
        "where can i see my profile",
        "where do i see my profile",
        "can you help me find the export button",
        "what should i click to get new models",
        "how do i get to my account settings",
        "which one opens a new chat",
        "i want to find the download link",
        "how can i see my usage",
        "take me to my billing page",
        "how do i switch to the models tab",
    )
    words = (
        "how do i make pasta",
        "what time is it in tokyo",
        "how do i center a div",
        "read this email to me",
        "what is on my screen",
        "who wrote this book",
        "explain how memory paging works",
        "where do i find a good pasta recipe",
        "tell me where the story goes next",
        "summarise this page",
    )
    missed = [q for q in point if not capture.wants_pointing(q)]
    fired = [q for q in words if capture.wants_pointing(q)]
    assert not missed, f"would not have pointed: {missed}"
    assert not fired, f"would have pointed for no reason: {fired}"

    # A pointing question must also pull a screenshot
    for q in point:
        assert capture.wants_screen(q) or capture.wants_pointing(q), q

    print(f"ok  pointing gate: {len(point)} point, {len(words)} don't, 0 wrong")


def check_vision_probe() -> None:
    """Ollama's own answer beats the name table, and only where it applies."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    from mellowd import config, llm

    asked = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(
                self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}"
            )
            asked.append((self.path, body.get("model")))
            # What a real Ollama returns for these two, measured on this laptop.
            seeing = body.get("model", "").startswith("gemma3:4b")
            payload = json.dumps(
                {"capabilities": ["completion", "vision"] if seeing else ["completion"]}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 8796), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    saved = dict(llm._VISION_CACHE)
    llm._VISION_CACHE.clear()
    try:
        # The /v1 the chat adapter talks to is not where /api/show lives.
        assert llm._ollama_root("http://127.0.0.1:8796/v1") == "http://127.0.0.1:8796"
        assert llm._ollama_root("http://127.0.0.1:8796") == "http://127.0.0.1:8796"

        def section(model, vision="auto", provider="ollama"):
            return {
                "provider": provider,
                "vision": vision,
                "model": model,
                "base_url": "http://127.0.0.1:8796/v1",
            }

        # The whole point: the table says no, Ollama says yes, Ollama wins.
        seeing = section("gemma3:4b")
        assert not config.resolves_vision(seeing), "the name table changed under us"
        assert not llm.vision_ok(seeing), "answered before anyone asked ollama"
        llm.probe_vision(seeing)
        assert llm.vision_ok(seeing), "ollama said vision and we ignored it"
        assert asked == [("/api/show", "gemma3:4b")], asked

        # A text-only local model is believed too, and the answer is cached
        blind = section("gemma3:1b")
        llm.probe_vision(blind)
        llm.probe_vision(blind)
        assert not llm.vision_ok(blind)
        assert len(asked) == 2, asked

        # An explicit setting is the user's answer
        for mode, want in (("on", True), ("off", False)):
            forced = section("gemma3:1b" if want else "gemma3:4b", vision=mode)
            before = len(asked)
            llm.probe_vision(forced)
            assert llm.vision_ok(forced) is want, (mode, want)
            assert len(asked) == before, "probed a model the user had already decided"

        # Cloud providers have no /api/show, so they keep the name table.
        cloud = section("gpt-4o", provider="openai")
        before = len(asked)
        llm.probe_vision(cloud)
        assert len(asked) == before, "probed a provider that has no such endpoint"
        assert llm.vision_ok(cloud), "cloud lost its name-table guess"

        # Ollama down: no answer, no cache entry, and the guess stands
        llm._VISION_CACHE.clear()
        dead = {**section("gemma3:4b"), "base_url": "http://127.0.0.1:8797/v1"}
        llm.probe_vision(dead)
        assert not llm._VISION_CACHE, "cached a failure"
        assert llm.vision_ok(dead) == config.resolves_vision(dead)
    finally:
        server.shutdown()
        llm._VISION_CACHE.clear()
        llm._VISION_CACHE.update(saved)

    print("ok  ollama's own capabilities beat the name table, overrides beat both")


async def _run_pass(chunks, look="ask", fired=None):
    """Drive main._pass with a canned stream."""
    from mellowd import llm, main

    class FakeWS:
        def __init__(self):
            self.sent = []

        async def send_text(self, text):
            self.sent.append(json.loads(text))

    async def gen():
        for c in chunks:
            yield c

    ws = FakeWS()
    s = object.__new__(main.Session)
    s.ws = ws
    s.history = []
    s.destination = None
    s.speaker = type("S", (), {"begin": lambda self: None})()
    async def hook(pick):
        # What had already reached the bubble when the marker landed.
        fired.append((pick, len([m for m in ws.sent if m.get("type") == "reply_chunk"])))

    real_chat = llm.chat
    llm.chat = lambda history, cfg, image=None: gen()
    try:
        reply, asked, point = await main._pass(
            s, {}, speak=False, look=look, on_point=hook if fired is not None else None
        )
    finally:
        llm.chat = real_chat
    texts = "".join(m["text"] for m in ws.sent if m.get("type") == "reply_chunk")
    return reply, asked, texts, point


async def check_marker_hold() -> None:
    """Phase 1's hold: [look] is caught whole, lookalikes flush untouched."""
    from mellowd import llm, main, tts

    run = _run_pass

    # The exact marker, chopped every way, is caught with nothing leaking out.
    for step in (1, 2, 3, 6, 100):
        reply, asked, texts, _ = await run(
            [llm.LOOK[i : i + step] for i in range(0, len(llm.LOOK), step)]
        )
        assert asked, f"step {step}: marker missed"
        assert reply == "" and texts == "", f"step {step}: leaked {reply!r} {texts!r}"

    # Marker plus trailing words in one gulp — still a marker, still silent.
    reply, asked, texts, _ = await run([f"{llm.LOOK}\n\nLet me see."])
    assert asked and texts == "", (reply, asked, texts)

    # A lookalike that starts the same but isn't the marker flushes whole.
    keep = "[looking] around the room."
    reply, asked, texts, _ = await run([keep[i : i + 3] for i in range(0, len(keep), 3)])
    assert not asked and texts == keep, (asked, texts)

    # Whitespace before the marker is tolerated.
    reply, asked, _, _ = await run(["  ", "[look]"])
    assert asked

    # A stream cut off inside a would-be marker degrades to text, not silence.
    reply, asked, texts, _ = await run(["[lo"])
    assert not asked and texts == "[lo", (asked, texts)

    # The regression this window exists for: one word of preamble used to defeat the scan entirely
    preamble = "Sure, let me look. [look]"
    for step in (1, 4, 100):
        reply, asked, texts, _ = await run(
            [preamble[i : i + step] for i in range(0, len(preamble), step)]
        )
        assert asked, f"step {step}: preamble hid the marker"
        assert texts == "", f"step {step}: leaked {texts!r} before asking"

    # Past the window it is an ordinary answer, not a hang.
    long_answer = "word " * 40
    reply, asked, texts, _ = await run([long_answer[i : i + 5] for i in range(0, 200, 5)])
    assert not asked and texts == long_answer, (asked, len(texts))

    # Phase 2 already holds the screenshot: a stray marker is dropped
    reply, asked, texts, _ = await run(["[look] I see a code editor."], look="strip")
    assert not asked, "phase 2 asked for a second screenshot"
    assert "[look]" not in texts and "code editor" in texts, texts

    # Vision off: the model was never told the marker exists
    reply, asked, texts, _ = await run(["[a] bracketed answer."], look="")
    assert not asked and texts == "[a] bracketed answer.", (asked, texts)

    # Past the window it is no longer a request — the answer is already being spoken
    tail = "This is a perfectly ordinary answer that simply runs on for a good "
    tail += "while before anything else happens at all. "
    assert len(tail) > main.LOOK_SCAN, len(tail)
    reply, asked, texts, _ = await run([tail, "Also [look] is not for you to see."])
    assert not asked, "a marker past the window was mistaken for a request"
    assert "[look]" not in texts, texts
    assert texts.startswith(tail) and "not for you to see" in texts, texts

    # The backstop for the spoken half. clean_for_speech runs on whole sentences
    assert tts.clean_for_speech("[look] here it is") == "here it is"
    assert tts.clean_for_speech("it [looks like] rain") == "it [looks like] rain"
    assert tts.clean_for_speech("take a look at this") == "take a look at this"

    print("ok  [look] survives preambles and chunking, phase 2 and vision-off stay clean")


async def check_point_marker() -> None:
    """[POINT:...] never reaches the bubble or the voice, at any chunking."""
    from mellowd import main, tts

    answer = "Open the File menu in the top left. "
    marker = "[POINT:12]"

    # Chopped every way, including straight through the middle of the marker.
    for step in (1, 2, 3, 6, 100):
        whole = marker + " " + answer
        _, _, texts, point = await _run_pass(
            [whole[i : i + step] for i in range(0, len(whole), step)], look=""
        )
        assert point == 12, f"step {step}: got {point!r}"
        assert texts.strip() == answer.strip(), f"step {step}: leaked {texts!r}"
        _, _, texts, point = await _run_pass([answer + marker], look="")
        assert point == 12 and texts.strip() == answer.strip(), (point, texts)

    # Words instead of a number. Not asked for and accepted anyway
    _, _, _, point = await _run_pass(["[POINT:API Keys] There."], look="")
    assert point == "API Keys", point

    # The hook fires the instant the marker lands
    fired = []
    await _run_pass(["[POINT:", "3] Cl", "ick that."], look="pick", fired=fired)
    assert fired == [(3, 0)], f"the answer started before the bone did: {fired}"

    # "Pointing wouldn't help" must be distinguishable from "the model forgot"
    _, _, texts, point = await _run_pass([f"{answer}[POINT:none]"], look="")
    # A veto, and it has to be tellable apart from "the model said nothing"
    assert point is main.NONE and texts.strip() == answer.strip(), (point, texts)

    # Brackets that close inside the stream are ordinary words and must not be held back
    ordinary = "Check step [1] and the [second] one."
    _, _, texts, point = await _run_pass(
        [ordinary[i : i + 4] for i in range(0, len(ordinary), 4)], look=""
    )
    assert point is None and texts == ordinary, (point, texts)

    # An unclosed bracket that never becomes a marker is released at the end of the stream
    _, _, texts, point = await _run_pass(["All done [", "but not really"], look="")
    assert point is None and texts == "All done [but not really", texts

    # ...and one that runs past the hold is released without waiting.
    runaway = "[" + "x" * (main.POINT_HOLD + 20)
    _, _, texts, _ = await _run_pass([runaway], look="")
    assert texts == runaway, texts

    # Both markers on one turn: phase 2 strips [look] and still finds the point.
    _, _, texts, point = await _run_pass(
        ["[look] Click Export. [POINT:9]"], look="strip"
    )
    assert point == 9, point
    assert "[look]" not in texts and "POINT" not in texts, texts

    # The backstop for the spoken half, for a marker that somehow survives.
    assert tts.clean_for_speech("Click it. [POINT:1]") == "Click it."
    assert tts.clean_for_speech("Nothing here. [POINT:none]") == "Nothing here."

    print("ok  [POINT:...] parses split across chunks and never reaches bubble or voice")


async def check_act() -> None:
    """Doing things: what gets offered, what the marker means, what runs."""
    from mellowd import act, capture, main, sessions

    rows = [
        act.Thing(label="Spotify", kind="app", target="Spotify.exe"),
        act.Thing(label="File Explorer", kind="app", target="Microsoft.Windows.Explorer"),
        act.Thing(label="Downloads", kind="place", target="shell:Downloads"),
        act.Thing(label="Google Drive", kind="site", target="https://drive.google.com"),
        act.Thing(label="YouTube", kind="site", target="https://www.youtube.com"),
        act.Thing(label="spotify-notes.txt", kind="file", target="C:\\x\\spotify-notes.txt"),
    ]

    def listing(query):
        saved = act.apps, act.places, act.sites, act.files
        try:
            act.apps = lambda: [r for r in rows if r.kind == "app"]
            act.places = lambda: [r for r in rows if r.kind == "place"]
            act.sites = lambda: [r for r in rows if r.kind == "site"]
            act.files = lambda: [r for r in rows if r.kind == "file"]
            return act.catalog(query)
        finally:
            act.apps, act.places, act.sites, act.files = saved

    # The app beats their own note about it. A bare word is the least likely way anyone refers to a file
    top = listing("open spotify")[0]
    assert (top.label, top.kind) == ("Spotify", "app"), top

    # A verb row is scored on the sentence, never on its own label
    assert listing("play back in black")[0].kind == "youtube"
    assert listing("play back in black on spotify")[0].kind == "spotify"
    assert listing("turn spotify down to 50")[0].kind == "volume"
    assert listing("make it quieter")[0].kind == "volume"

    # Asking politely has to work.
    for polite in (
        "hey mellow can you open spotify for me",
        "hey mellow could you please open my downloads",
        "mellow can you open spotify please",
    ):
        top = listing(polite)[0]
        assert top.score >= act.THRESHOLD, (polite, top.label, top.score)

    # "google" alone is not a search verb.
    assert listing("open google drive")[0].kind == "site"
    assert listing("hey can you open google drive please")[0].kind == "site"
    assert listing("search for cheap flights")[0].kind == "google"

    # Exact commands bypass the fragile model marker.
    downloads = act.direct("could you please open my downloads folder", listing("open downloads"))
    assert downloads and downloads[0].kind == "place" and downloads[1] == "", downloads
    youtube = act.direct("open youtube", listing("open youtube"))
    assert youtube and youtube[0].kind == "site", youtube
    play = act.direct(
        "open youtube and play something", listing("open youtube and play something")
    )
    assert play and play[0].kind == "youtube" and play[1] == "music", play
    natural = act.direct(
        "Do you know about a song Back in Black? Can you pay the play that on YouTube?",
        listing("play Back in Black on youtube"),
    )
    assert natural and natural[0].kind == "youtube" and natural[1] == "Back in Black", natural
    assert act.media_argument("play that on youtube", "that") is None
    assert act.direct("open the file menu", listing("open the file menu")) is None

    # An acknowledgement is not filler on an action turn, and it survives by construction
    from mellowd import llm

    stream = llm._Stream()
    acted = stream.emit("[DO:1|x] Sure, putting that on now.") + stream.flush()
    assert "Sure," in acted, acted
    plain = llm._Stream()
    assert (plain.emit("Sure! The answer is 42.") + plain.flush()) == "The answer is 42."

    # Nothing here is worth doing, and the turn has to be free to be about something else entirely.
    quiet = listing("what does this error mean")
    assert not quiet or quiet[0].score < act.THRESHOLD, quiet

    # The gate is loose on purpose and the catalog is the second half of it.
    assert capture.wants_action("open file explorer")
    assert capture.wants_action("turn spotify down")
    assert not capture.wants_action("what is on my screen")
    assert not capture.wants_action("summarise this page")

    # A question about *where* is never a command to *do*
    for asking in (
        "show me where to click to run this",
        "show me where i should click for the extension part",
        "where do i click to open the extensions panel",
        "point me to the extensions icon",
        "which one opens a new chat",
        "where can i see my profile",
    ):
        assert not capture.wants_action(asking), f"acting stole {asking!r}"
        assert capture.wants_pointing(asking), f"and pointing did not catch it: {asking!r}"

    # ...without costing any of the sentences that must still open something.
    for doing in (
        "open chrome",
        "hey mellow can you open chrome for me",
        "open my downloads",
        "show me my downloads",
        "play back in black",
        "turn chrome down to 30",
    ):
        assert capture.wants_action(doing), f"stopped opening for {doing!r}"

    # The marker, at every chunking, with and without an argument.
    whole = "[DO:2|back in black] Putting that on."
    for step in (1, 3, 7, 100):
        text, tail, deed = "", "", None
        for i in range(0, len(whole), step):
            text, tail, found = main._split_point(tail + whole[i : i + step], main._DO_TOKEN)
            if found:
                deed = found
        assert deed == (2, "back in black"), f"step {step}: {deed!r}"
    assert main._split_point("[DO:4] Done.", main._DO_TOKEN)[2] == (4, "")
    assert main._split_point("[DO:none] No.", main._DO_TOKEN)[2] is main.NONE

    # ...and the pointing marker still parses on its own token
    assert main._split_point("[POINT:3] There.")[2] == 3

    # Resolving a deed onto a row. Everything that is not a row is nothing
    assert main._chosen((1, ""), rows)[0] is rows[0]
    assert main._chosen((5, "x"), rows) == (rows[4], "x")
    assert main._chosen((0, ""), rows)[0] is None
    assert main._chosen((99, ""), rows)[0] is None
    assert main._chosen(main.NONE, rows)[0] is None
    assert main._chosen(None, rows)[0] is None
    assert main._chosen(("Downloads", ""), rows)[0] is rows[2]
    assert main._chosen(("the printer settings", ""), rows)[0] is None

    # A declined turn is spotted before a word of it is emitted
    assert main._declined("[DO:none] let me look")
    assert not main._declined("[DO:2] opening")

    # Integration: an exact Downloads request executes and answers without entering _pass
    sent, ran = [], []

    class FakeWS:
        async def send_text(self, text):
            sent.append(json.loads(text))

    exact = act.Thing(
        label="Downloads", kind="place", target="shell:Downloads", score=1.0
    )
    saved = act.catalog, act.run, main._pass, sessions.record
    try:
        act.catalog = lambda query: [exact]
        act.run = lambda thing, argument="": ran.append((thing, argument)) or "opened Downloads"

        async def no_agent(*args, **kwargs):
            raise AssertionError("an exact Downloads command called the agent")

        main._pass = no_agent
        sessions.record = lambda *args, **kwargs: None
        partial = {"text": ""}
        reply, did = await main._act(
            type("Session", (), {"ws": FakeWS()})(),
            {"llm": {}},
            False,
            partial,
            "open my downloads folder",
        )
    finally:
        act.catalog, act.run, main._pass, sessions.record = saved
    assert did and reply == "Opened Downloads.", (did, reply)
    assert ran == [(exact, "")], ran
    assert partial["text"] == "Opened Downloads.", partial
    assert sent == [{"type": "reply_chunk", "text": "Opened Downloads."}], sent

    # The argument split for a volume request
    assert act._volume_args("spotify 50") == ("spotify", 0.5)
    assert act._volume_args("spotify to 30 percent") == ("spotify", 0.3)
    assert act._volume_args("chrome 200")[1] == 1.0
    assert act._volume_args("nothing here") == (None, 0.0)

    # Saying when, out loud. Every one of these is a shape a speech recogniser actually produces
    from datetime import datetime

    from mellowd import remind

    now = datetime(2026, 8, 27, 14, 5)
    for said, clock, text, daily in (
        ("in 10 minutes take the pizza out", "14:15", "take the pizza out", False),
        ("in ten minutes walk the dog", "14:15", "walk the dog", False),
        ("in twenty five minutes stretch", "14:30", "stretch", False),
        ("in thirty five minutes tea", "14:40", "tea", False),
        ("in an hour call mum", "15:05", "call mum", False),
        ("in half an hour stand up", "14:35", "stand up", False),
        ("at 9pm dinner with sharif", "21:00", "dinner with sharif", False),
        ("at 21:30 stand up", "21:30", "stand up", False),
        ("at 3 pick up the parcel", "15:00", "pick up the parcel", False),
        ("every day at 7pm water the plants", "19:00", "water the plants", True),
    ):
        got = remind.at(said, now)
        assert got == (clock, text, daily), f"{said!r} -> {got!r}"

    # When is the one thing that cannot be guessed
    assert remind.at("take the bins out", now) is None
    assert remind.at("at 25:00 nonsense", now) is None
    assert remind.at("", now) is None

    # The timers route to their own rows, and stopping is not starting.
    assert listing("remind me in ten minutes to stretch")[0].kind == "remind"
    assert listing("set a reminder for 9pm")[0].kind == "remind"
    assert listing("set a pomodoro for 25 minutes")[0].kind == "pomodoro"
    assert listing("start a focus round")[0].kind == "pomodoro"
    assert listing("stop the pomodoro")[0].kind == "pomodoro_stop"
    assert listing("cancel the focus timer")[0].kind == "pomodoro_stop"

    # A count of minutes, however it was said.
    assert [act.minutes(x) for x in ("25 minutes", "twenty five", "thirty five",
                                     "forty five", "half an hour")] == [25, 25, 35, 45, 30]
    assert act.minutes("") is None

    # The pomodoro is the frontend's
    assert set(act.ON_SCREEN) == {"pomodoro", "pomodoro_stop"}

    print("ok  acting: exact commands run directly; fuzzy commands stay catalog-bounded")


async def check_one_turn_one_bone() -> None:
    """A pointing turn answers once and stops, whatever the screen does after."""
    import types

    import numpy as np

    from mellowd import agents, capture, config, llm, locator, main, point, sessions

    sent = []

    class FakeWS:
        async def send_text(self, text):
            msg = json.loads(text)
            sent.append(msg)
            if msg.get("type") == "capture" and msg.get("phase") == "begin":
                session.hidden.set()

    calls = []

    def fake_chat(history, cfg, image=None, **kw):
        calls.append(1)
        text = "[POINT:1] Open the File menu in the top left."

        async def gen():
            for i in range(0, len(text), 7):
                yield text[i : i + 7]

        return gen()

    cfg = config.load()
    cfg["llm"] = {**cfg["llm"], "mode": "agent", "provider": "codex"}
    cfg["tts"] = {**cfg["tts"], "speak": True}

    # A screen that never stops moving
    frames = iter([np.full((36, 64), i * 7 % 255, np.int16) for i in range(200)])

    session = object.__new__(main.Session)
    session.ws = FakeWS()
    session.history = []
    session.destination = None
    session.hidden = asyncio.Event()
    session.hidden.set()
    session.speaker = types.SimpleNamespace(
        begin=lambda: None,
        speak=lambda t: asyncio.sleep(0),
        finish=lambda: asyncio.sleep(0),
    )

    saved = (
        llm.chat, agents.chat, llm.probe_vision, llm.vision_ok, config.load,
        capture.thumbnail, capture.grab, capture.media_bytes, capture.foreground,
        sessions.record, main.HIDE_TIMEOUT, point.candidates,
        locator.locate, locator.locate_and_answer, locator.changed_at,
        capture.active_monitor,
    )
    try:
        llm.chat = agents.chat = fake_chat
        llm.probe_vision = lambda c: None
        llm.vision_ok = lambda c: True
        config.load = lambda: cfg
        capture.thumbnail = lambda: next(frames, None)
        capture.grab = lambda *a: (bytes.fromhex("ffd8") + b"fake", 2048, 1152, None)
        capture.media_bytes = lambda b: None
        capture.foreground = lambda: ("premiere.exe", "Adobe Premiere Pro")
        sessions.record = lambda *a, **k: None
        target_monitor = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
        target = point.Target(
            0.5,
            0.05,
            "File menu",
            "uia",
            1.0,
            "menu item",
            monitor=target_monitor,
        )
        point.candidates = lambda *args, **kwargs: [target]
        locator.locate = lambda *args, **kwargs: asyncio.sleep(0, result=target)
        locator.locate_and_answer = lambda *args, **kwargs: asyncio.sleep(
            0,
            result=locator.GroundedResult(
                target, "Open the File menu in the top left."
            ),
        )
        locator.changed_at = lambda *args: False
        capture.active_monitor = lambda: {"left": 0, "top": 0, "width": 2048, "height": 1152}
        main.HIDE_TIMEOUT = 1.0
        await asyncio.wait_for(
            main.answer(session, "what do i click to export this"), 20
        )
    finally:
        (
            llm.chat, agents.chat, llm.probe_vision, llm.vision_ok, config.load,
            capture.thumbnail, capture.grab, capture.media_bytes, capture.foreground,
            sessions.record, main.HIDE_TIMEOUT, point.candidates,
            locator.locate, locator.locate_and_answer, locator.changed_at,
            capture.active_monitor,
        ) = saved

    assert calls == [], "agent pointing made a redundant generic chat call"

    points = [m for m in sent if m["type"] == "point"]
    # One clear on the way in, then the bone. And it is still up at the end
    assert [p["nx"] for p in points] == [None, 0.5], points
    assert points[-1]["label"] == "File menu", points[-1]
    assert points[-1]["monitor"] == target_monitor, points[-1]

    bubble = "".join(m["text"] for m in sent if m["type"] == "reply_chunk")
    assert "POINT" not in bubble and "[" not in bubble, bubble
    assert bubble.count("Open the File menu") == 1, bubble
    print("ok  one question, one answer, one bone, and a moving screen changes nothing")


async def check_turn_monitor_lock() -> None:
    """Capture acknowledgements cannot redirect an already submitted turn."""
    import types

    from mellowd import main, point

    locked = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
    pet_monitor = {"left": 0, "top": 0, "width": 2560, "height": 1440}
    sent = []

    class FakeWS:
        async def send_text(self, text):
            msg = json.loads(text)
            sent.append(msg)
            if msg == {"type": "capture", "phase": "begin"}:
                session.hidden.set()

    session = types.SimpleNamespace(
        ws=FakeWS(), hidden=asyncio.Event(), turn_monitor=locked
    )
    await main.handle(
        session, {"type": "capture_ready", "monitor": pet_monitor}
    )
    assert session.turn_monitor == locked, session.turn_monitor

    measured = []
    original_shot = main._shot
    try:
        def fake_shot(max_edge, monitor):
            measured.append(monitor)
            return None, "", ""

        main._shot = fake_shot
        await main._unseen_shot(session)
    finally:
        main._shot = original_shot
    assert measured == [locked], measured

    target = point.Target(
        0.25, 0.75, "target", "uia", 1.0, monitor=locked
    )
    await main._aim(session, target)
    aimed = [message for message in sent if message.get("type") == "point"][-1]
    assert aimed["monitor"] == locked, aimed
    print("ok  one cursor monitor stays locked through capture and pointing")


def check_point_score() -> None:
    """The list handed to the model: what stays on it, and in what order."""
    from mellowd import point

    # OCR does not put spaces where you expect them - this exact sidebar comes back as "APIKeys"
    assert point.squash("API Keys") == point.squash("APIKeys") == "apikeys"

    mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}

    def listing(query, rows, page=None):
        saved = point.monitor, point.uia_candidates, point.ocr_candidates
        try:
            point.monitor = lambda: mon
            point.uia_candidates = lambda hwnd=0: (
                [r for r in rows if r[6] == "uia"], page
            )
            point.ocr_candidates = lambda pixels: [r for r in rows if r[6] == "ocr"]
            return point.candidates(query, None)
        finally:
            point.monitor, point.uia_candidates, point.ocr_candidates = saved

    def row(name, x, y, w, h, kind="", source="ocr"):
        return (name, x, y, w, h, kind, source)

    # Ordering: the rows the question is about come first
    got = listing("point me towards the API key", [
        row("Models", 7, 440, 102, 29),
        row("APIKeys", 40, 565, 90, 25),
        row("Usage Logs", 20, 674, 120, 25),
    ])
    assert [c.label for c in got][0] == "APIKeys", [c.label for c in got]
    assert len(got) == 3, "a row was dropped that could have been the answer"

    # VS Code appends shortcuts and status badges to the accessible name.
    vscode = listing("where is source control", [
        row(
            "Source Control (Ctrl+Shift+G) - 9 pending changes",
            20, 150, 48, 48, "tab", "uia",
        ),
    ])
    assert len(vscode) == 1 and vscode[0].label == "Source Control", vscode

    # The bug that put the bone in the tab strip twice.
    browser = listing("what should i click to get new models", [
        row("Models | NVIDIA NIM", 660, 12, 110, 26, "tab", "uia"),
        row("New Tab", 1480, 12, 30, 26, "button", "uia"),
        row("Models", 300, 170, 60, 20),
        row("Model Card", 1080, 245, 100, 20),
    ])
    assert browser[0].label == "Models" and browser[0].ny * 1080 > 100, browser[0]

    # And the model is told which is which, in words
    shown = point.describe(browser)
    assert '"Models | NVIDIA NIM" tab 37,2' in shown, shown
    # No document rect here

    # Chrome calls its address bar "Address and search bar"
    page_rect = (0, 145, 1917, 933)
    amazon = listing("where should i click to update my address", [
        row("Address and search bar", 190, 66, 1500, 28, "text box", "uia"),
        row("New Tab", 1490, 12, 30, 26, "button", "uia"),
        row("Update location", 210, 186, 120, 20),
        row("Account & Lists", 1520, 186, 140, 20),
    ], page_rect)
    assert [c.chrome for c in amazon] == [False, False, True, True], amazon
    assert amazon[0].label == "Update location", amazon[0]
    told = point.describe(amazon)
    assert told.index("Update location") < told.index("BROWSER"), told
    assert told.index("BROWSER") < told.index("Address and search bar"), told

    # And the browser can never take more than its share of the list
    crowd = listing("where do i click to save", [
        row("Tab %d" % i, i * 20, 12, 18, 26, "tab", "uia")
        for i in range(point.MAX_CHROME + 20)
    ] + [row("Save", 400, 400, 60, 20)], page_rect)
    assert sum(c.chrome for c in crowd) == point.MAX_CHROME, crowd
    assert crowd[0].label == "Save", crowd[0]

    # The split must never be able to starve the list
    elsewhere = listing("where is the source control", [
        row("Source Control", 30, 178, 46, 46, "tab", "uia"),
        row("Explorer", 30, 70, 46, 46, "tab", "uia"),
        row("Extensions", 30, 290, 46, 46, "tab", "uia"),
    ], (-1920, 0, 1920, 1080))
    assert len(elsewhere) == 3, f"the off-screen document ate the list: {elsewhere}"
    assert not any(c.chrome for c in elsewhere), elsewhere
    assert elsewhere[0].label == "Source Control", elsewhere[0]

    # ...and even with a document that *is* on this screen
    nothing_inside = listing("where is the toolbar", [
        row("Toolbar", 30, 900, 60, 20),
    ], (0, 0, 1920, 200))
    assert len(nothing_inside) == 1 and not nothing_inside[0].chrome, nothing_inside

    # A native app has no document in its tree, and then none of it is furniture
    native = listing("where is the export button", [
        row("Export Media", 400, 40, 100, 20, "menu item", "uia"),
    ])
    assert native and not native[0].chrome, native

    # A pane the size of the window is the window.
    assert [c.label for c in listing("where are the settings", [
        row("Settings", 0, 0, 1900, 1000),
        row("Settings", 800, 400, 90, 24, "button", "uia"),
    ])] == ["Settings"]

    # Prose is not a click target, and neither is a stray letter.
    kept = [c.label for c in listing("how do i export", [
        row("Export", 100, 100, 70, 20),
        row("Follows Selected Date Range and then some more", 700, 300, 400, 20),
        row("x", 5, 5, 8, 8),
    ])]
    assert kept == ["Export"], kept

    # The same thing read twice, once by each source. One row
    twice = listing("open the export panel", [
        row("Export", 101, 101, 68, 18, False, "ocr"),
        row("Export", 100, 100, 70, 20, "button", "uia"),
    ])
    assert len(twice) == 1 and twice[0].source == "uia", twice

    # A second monitor is somewhere the overlay cannot draw
    assert listing("where is the export button", [
        row("Export", -900, 500, 80, 24, "button", "uia")
    ]) == []

    # The cap is a real wall
    flood = [row("Item %d" % i, i, i, 20, 10) for i in range(point.MAX_ITEMS + 50)]
    assert len(listing("anything", flood)) == point.MAX_ITEMS

    print("ok  the on-screen list: nothing unpointable offered, likeliest first")


def check_point_pick() -> None:
    """Turning what the model said into a row, or into nothing."""
    from mellowd import main, point

    rows = [
        point.Target(0.04, 0.22, "New", "uia", 1.0),
        point.Target(0.40, 0.02, "New chat - Claude", "ocr", 0.0),
        point.Target(0.05, 0.26, "Projects", "ocr", 0.0),
    ]

    # A number is one-based, because the list the model reads is.
    assert main._picked(1, rows) is rows[0]
    assert main._picked(3, rows) is rows[2]

    # Everything that is not a row on that list is no bone.
    assert main._picked(0, rows) is None
    assert main._picked(99, rows) is None
    assert main._picked(main.NONE, rows) is None
    assert main._picked(None, rows) is None
    assert main._picked(2, []) is None

    # Words instead of a number, matched against the rows it was shown and nothing else
    assert main._picked("New", rows) is rows[0]
    assert main._picked("Projects", rows) is rows[2]
    assert main._picked("the print dialog", rows) is None

    print("ok  a pick is a row on the list or it is no bone at all")


async def check_locator() -> None:
    """Coarse/fine parsing and coordinate mapping without a live model."""
    import types
    from dataclasses import replace

    import numpy as np

    from mellowd import locator, point

    mon = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
    pixels = np.full((1080, 1920, 3), 35, dtype=np.uint8)
    shot = types.SimpleNamespace(pixels=pixels, monitor=mon)
    source = point.Target(
        44 / 1920,
        210 / 1080,
        "Source Control",
        "uia",
        1.0,
        "tab",
        False,
        (-1900, 186, 48, 48),
        mon,
    )
    nearby = point.Target(
        80 / 1920,
        200 / 1080,
        "Source panel",
        "ocr",
        0.0,
        "",
        False,
        (-1870, 185, 60, 32),
        mon,
    )
    replies = iter(["[REGION:E1]", "[TARGET:E2]"])
    real = locator._call
    try:
        async def answer(*args):
            return next(replies)

        locator._call = answer
        found = await locator.locate(
            "where is source control", shot, {"llm": {}}, [source, nearby]
        )
    finally:
        locator._call = real
    assert found and found.label == "Source Control" and found.bounds == source.bounds, found

    replies = iter(["REGION:C96", "TARGET:G144"])
    try:
        async def answer_grid(*args):
            return next(replies)

        locator._call = answer_grid
        visual = await locator.locate("the unlabeled icon", shot, {"llm": {}}, [])
    finally:
        locator._call = real
    assert visual and visual.source == "visual-grid", visual
    assert 0.8 < visual.nx < 1 and 0.8 < visual.ny < 1, visual
    assert visual.monitor["left"] == -1920, visual.monitor

    # These are the literal replies captured from failed Codex/Claude turns.
    assert locator._bare_choice("E1", "coarse") == "E1"
    assert locator._bare_choice("E7", "fine") == "E7"
    assert locator._bare_choice("13", "coarse") == "C13"
    assert locator._bare_choice("13", "fine") == "G13"
    assert locator._grounded_fields(
        '{"selection":"E1","answer":"Click Source Control."}',
        "coarse",
        {"NONE", "E1", "E2"},
        [source, nearby],
    ) == ("E1", "Click Source Control.")

    # Codex 0.150 was observed returning a useful natural-language answer instead of the schema.
    extensions = point.Target(
        28 / 1920,
        285 / 1080,
        "Extensions",
        "uia",
        1.0,
        "button",
        False,
        (-1918, 262, 46, 46),
        mon,
    )
    prose = "Click the Extensions icon on the far-left Activity Bar, around (20, 265)."
    assert locator._grounded_fields(
        prose, "coarse", {"NONE", "E1"}, [extensions]
    )[0] == "E1"
    duplicate = replace(extensions, source="ocr", score=0.0)
    assert locator._grounded_fields(
        prose, "coarse", {"NONE", "E1", "E2"}, [extensions, duplicate]
    )[0] == "E1", "an OCR duplicate hid the exact UIA control"

    # Agent pointing makes one structured call when the overview already has the exact measured control
    calls = []
    real_grounded = locator.agents.complete_grounded
    try:
        async def grounded(*args):
            calls.append(args)
            return '{"selection":"E1","answer":"Click Source Control on the left."}'

        locator.agents.complete_grounded = grounded
        result = await locator.locate_and_answer(
            "point at source control",
            shot,
            {"llm": {"mode": "agent", "provider": "codex", "model": ""}},
            [source, nearby],
            [{"role": "user", "content": "point at source control"}],
        )
    finally:
        locator.agents.complete_grounded = real_grounded
    assert len(calls) == 1, f"exact hitbox needed {len(calls)} agent calls"
    assert result.target and result.target.bounds == source.bounds, result
    assert result.answer == "Click Source Control on the left.", result.answer
    print("ok  locator accepts CLI variants, snaps to hitboxes, and combines the answer")


def check_point_list() -> None:
    """The two screens that failed, and whether the answer is on the list."""
    import pathlib
    import time

    import numpy as np
    from PIL import Image

    from mellowd import point

    here = pathlib.Path(__file__).parent / "fixtures"
    frames = {
        "claude-new-chat": ("where do i start a new chat", "New", 77, 235),
        "tokenrouter-profile": ("point at my profile", "perspectiveai2", 1803, 191),
        "nvidia-models": ("what should i click to get new models", "Models", 326, 180),
        "amazon-update-location": (
            "where should i click to update my address", "Updatelocation", 270, 196,
        ),
    }
    if not all((here / (name + ".png")).exists() for name in frames):
        print(".. pointing fixtures not present, skipping the real-screen list")
        return

    saved = point.uia_candidates
    try:
        # A png has no foreground window
        point.uia_candidates = lambda hwnd=0: ([], None)
        for name, (query, label, wx, wy) in frames.items():
            frame = np.asarray(Image.open(here / (name + ".png")).convert("RGB"))
            started = time.perf_counter()
            rows = point.candidates(query, frame)
            took = time.perf_counter() - started
            assert rows, name + ": nothing readable on a screen full of text"
            assert len(rows) <= point.MAX_ITEMS, "%s: %d rows" % (name, len(rows))
            exact = [
                (i, c) for i, c in enumerate(rows, 1)
                if point.squash(c.label) == point.squash(label)
            ]
            hit = exact or [
                (i, c) for i, c in enumerate(rows, 1)
                if point.squash(label) in point.squash(c.label)
            ]
            assert hit, "%s: %r is on the screen but not on the list" % (name, label)
            i, best = hit[0]
            x, y = best.nx * frame.shape[1], best.ny * frame.shape[0]
            assert abs(x - wx) < 12 and abs(y - wy) < 12, (
                "%s: row %d %r is at (%.0f,%.0f), wanted (%d,%d)"
                % (name, i, best.label, x, y, wx, wy)
            )
            print(
                "ok  %s: %d rows in %.2fs, %r at (%.0f,%.0f) is row %d"
                % (name, len(rows), took, best.label, x, y, i)
            )
    finally:
        point.uia_candidates = saved


def check_dpi() -> None:
    """The sidecar and UI Automation must measure in the same pixels."""
    import ctypes

    from mellowd import main

    main.set_dpi_aware()
    level = ctypes.c_int()
    ctypes.windll.shcore.GetProcessDpiAwareness(None, ctypes.byref(level))
    assert level.value == 2, f"dpi awareness is {level.value}, wanted 2 (per monitor)"

    from mellowd import point

    monitor = point.monitor()
    assert monitor and monitor["width"] > 0, monitor
    print(f"ok  per-monitor dpi aware; primary is {monitor['width']}x{monitor['height']}")


def check_ocr() -> None:
    """Windows' OCR engine, over a picture this check draws itself."""
    import time

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from mellowd import point

    # A real face at a real size.
    font = None
    for face in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(face, 22)
            break
        except OSError:
            continue
    if font is None:
        print("..  no truetype face to render with; OCR tier unverified here")
        return

    canvas = Image.new("RGB", (900, 400), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    placed = {"API Keys": (60, 120), "Usage Logs": (60, 200), "Export": (600, 120)}
    for text, (x, y) in placed.items():
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    started = time.monotonic()
    found = point.ocr_candidates(np.asarray(canvas))
    took = time.monotonic() - started
    if not found:
        print("..  no OCR language pack on this machine; the text tier is off")
        return
    # Fast enough to run *before* the answer rather than behind it. rapidocr was tried here and took
    assert took < 3.0, f"OCR took {took:.1f}s; the bone waits on this"

    # Every label drawn has to reach the list
    saved = point.monitor, point.uia_candidates, point.ocr_candidates
    try:
        point.monitor = lambda: {"left": 0, "top": 0, "width": 900, "height": 400}
        point.uia_candidates = lambda hwnd=0: ([], None)
        point.ocr_candidates = lambda pixels: found
        rows = point.candidates("what is on screen", None)
    finally:
        point.monitor, point.uia_candidates, point.ocr_candidates = saved

    for want, (x, y) in placed.items():
        hit = [c for c in rows if point.squash(want) in point.squash(c.label)]
        assert hit, f"{want!r} was drawn on screen and is not on the list"
        landed = (hit[0].nx * 900, hit[0].ny * 400)
        assert abs(landed[0] - x) < 90 and abs(landed[1] - y) < 30, (
            f"{want!r} landed at {landed}, wanted near {(x, y)}"
        )

    # Nothing on this canvas is a scrollbar
    assert not [c for c in rows if "scroll" in c.label.lower()], rows

    print(f"ok  windows OCR read the screen in {took:.2f}s and every label landed")


def check_uia() -> None:
    """One real walk of whatever window happens to have focus."""
    import time

    from mellowd import point

    started = time.monotonic()
    found, page = point.uia_candidates()
    took = time.monotonic() - started
    assert took < point.UIA_BUDGET + 2.0, f"the walk ran {took:.1f}s past its budget"
    if not found:
        print("..  no accessible window in the foreground; uia tier unverified here")
        return
    monitor = point.monitor()
    on_screen = [
        c for c in found
        if c[1] < monitor["left"] + monitor["width"] and c[2] < monitor["top"] + monitor["height"]
    ]
    assert on_screen, "every control was off the primary monitor"
    print(f"ok  uia read {len(found)} named controls in {took:.2f}s")


async def check_point_first() -> None:
    """The bone leaves before the answer does, and only ever onto a real row."""
    import types

    from mellowd import agents, capture, config, llm, locator, main, point, sessions

    rows = [
        point.Target(0.03, 0.53, "API Keys", "ocr", 0.0),
        point.Target(0.40, 0.02, "TokenRouter - Chrome", "uia", 1.0),
    ]

    async def turn(reply_text, offered=rows):
        sent = []

        class FakeWS:
            async def send_text(self, text):
                msg = json.loads(text)
                sent.append(msg)
                if msg.get("type") == "capture" and msg.get("phase") == "begin":
                    session.hidden.set()

        def fake_chat(messages, cfg, image=None, **kw):
            async def gen():
                for i in range(0, len(reply_text), 8):
                    yield reply_text[i : i + 8]

            return gen()

        cfg = config.load()
        cfg["llm"] = {**cfg["llm"], "mode": "cloud", "provider": "custom"}
        cfg["tts"] = {**cfg["tts"], "speak": False}

        session = object.__new__(main.Session)
        session.ws = FakeWS()
        session.history = []
        session.destination = None
        session.hidden = asyncio.Event()
        session.speaker = types.SimpleNamespace(
            begin=lambda: None,
            speak=lambda t: asyncio.sleep(0),
            finish=lambda: asyncio.sleep(0),
        )
        saved = (
            llm.chat, agents.chat, llm.probe_vision, llm.vision_ok, config.load,
            capture.grab, capture.media_bytes, capture.foreground,
            capture.thumbnail, sessions.record, point.candidates,
            main.HIDE_TIMEOUT, locator.locate, locator.changed_at,
            capture.active_monitor,
        )
        try:
            llm.chat = agents.chat = fake_chat
            llm.probe_vision = lambda c: None
            llm.vision_ok = lambda c: True
            config.load = lambda: cfg
            capture.grab = lambda *a: (bytes.fromhex("ffd8") + b"x", 2048, 1152, None)
            capture.media_bytes = lambda b: None
            capture.foreground = lambda: ("chrome.exe", "TokenRouter")
            capture.thumbnail = lambda: None
            sessions.record = lambda *a, **k: None
            point.candidates = lambda *args, **kwargs: list(offered)
            resolved = offered[0] if offered and reply_text.startswith("[POINT:1]") else None
            locator.locate = lambda *args, **kwargs: asyncio.sleep(0, result=resolved)
            locator.changed_at = lambda *args: False
            capture.active_monitor = lambda: {"left": 0, "top": 0, "width": 2048, "height": 1152}
            main.HIDE_TIMEOUT = 1.0
            await asyncio.wait_for(
                main.answer(session, "point me towards the API key"), 20
            )
        finally:
            (
                llm.chat, agents.chat, llm.probe_vision, llm.vision_ok, config.load,
                capture.grab, capture.media_bytes, capture.foreground,
                capture.thumbnail, sessions.record, point.candidates,
                main.HIDE_TIMEOUT, locator.locate, locator.changed_at,
                capture.active_monitor,
            ) = saved
        return sent

    sent = await turn("[POINT:1] That is where your keys live. Copy one from there.")
    # Every turn opens by clearing whatever the last one pointed at, so the bone that matters
    assert sent[0] == {"type": "point", "nx": None}, sent[0]
    kinds = [m["type"] for m in sent[1:]]
    assert "point" in kinds and "reply_chunk" in kinds, kinds
    assert kinds.index("point") < kinds.index("reply_chunk"), (
        "the answer started before the bone did: " + str(kinds)
    )
    aimed = sent[1:][kinds.index("point")]
    # The row's own fractions go out untouched.
    assert aimed["nx"] == 0.03 and aimed["label"] == "API Keys", aimed
    bubble = "".join(m["text"] for m in sent if m["type"] == "reply_chunk")
    assert "POINT" not in bubble and bubble.lstrip().startswith("That is"), bubble
    # And it is left there.
    assert sent[-1]["type"] != "point" or sent[-1]["nx"] is not None, sent[-1]

    # Nothing on the list fits
    sent = await turn("[POINT:none] Docs saves by itself, there is no save button.")
    assert not [m for m in sent if m["type"] == "point" and m["nx"] is not None], sent
    assert any(m["type"] == "reply_chunk" for m in sent), "no answer either"

    # A number nobody offered is not a bone.
    sent = await turn("[POINT:47] Over there somewhere.")
    assert not [m for m in sent if m["type"] == "point" and m["nx"] is not None], sent

    # No marker at all still answers
    sent = await turn("The keys are in the sidebar.")
    assert not [m for m in sent if m["type"] == "point" and m["nx"] is not None], sent
    assert any(m["type"] == "reply_chunk" for m in sent), "no answer either"

    # A screen with nothing readable on it: no list
    sent = await turn("[POINT:1] Right here.", offered=[])
    assert not [m for m in sent if m["type"] == "point" and m["nx"] is not None], sent

    print("ok  bone flies on the pick, before the answer, and only onto a real row")


def check_screen_change() -> None:
    """The watcher fires on a menu opening and shrugs at a pixel dog."""
    import numpy as np

    from mellowd import capture

    base = np.full((36, capture.THUMB_W), 128, dtype=np.int16)

    # A menu opening: a large block goes dark.
    menu = base.copy()
    menu[4:20, 2:14] = 20
    assert capture.changed(base, menu), "a menu opening was missed"

    # Mellow idling in the corner: a few pixels, well under the threshold.
    dog = base.copy()
    dog[32:36, 58:64] = 20
    assert not capture.changed(base, dog), "the pet's own animation would re-point"

    # Noise below CHANGE_LEVEL is not a change however widely it is spread.
    dither = base.copy()
    dither[:, :] = 128 + capture.CHANGE_LEVEL - 1
    assert not capture.changed(base, dither), "a brightness nudge counted as a change"

    # Mismatched or missing frames are "nothing happened", never a crash.
    assert not capture.changed(None, menu)
    assert not capture.changed(base, np.zeros((4, 4), dtype=np.int16))

    print(f"ok  screen change fires over {capture.CHANGE_FRAC:.0%}, ignores the pet")


async def check_screen_request() -> None:
    """The wire carries what the plan promises, per mode."""
    import base64
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from mellowd import config, llm, point

    seen = {}
    frame = b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\ndata: [DONE]\n\n'

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 8795), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    async def once(cfg, messages, image=None):
        async def drain():
            async for _ in llm.chat(messages, cfg, image=image):
                pass

        await asyncio.wait_for(drain(), 10)

    base = dict(config.DEFAULTS)
    base["llm"] = {
        **config.DEFAULTS["llm"],
        "provider": "custom",
        "mode": "cloud",
        "base_url": "http://127.0.0.1:8795/v1",
        "api_key": "k",
        "model": "test-vision",
        "reasoning_effort": "",
    }
    messages = [{"role": "user", "content": "read my screen"}]
    # Not a real JPEG and doesn't need to be: the adapter only base64s it.
    fake_jpeg = b"\xff\xd8\xff\xe0fakejpegbytes"

    try:
        # Phase 2: image attached -> parts + REMINDER_SEEN.
        await once(base, messages, image=fake_jpeg)
        body = seen["body"]
        user = [m for m in body["messages"] if m["role"] == "user"][-1]
        parts = user["content"]
        # Image first, question last, in that order. Both adapters agree on it now
        assert isinstance(parts, list) and parts[0]["type"] == "image_url", parts
        url = parts[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,"), url[:40]
        assert base64.b64decode(url.split(",", 1)[1]) == fake_jpeg
        assert parts[1]["type"] == "text" and parts[1]["text"] == "read my screen", parts
        last_system = [
            m for m in body["messages"] if m["role"] == "system"
        ][-1]["content"]
        assert "attached to their latest message" in last_system, last_system
        assert "[look]" not in last_system, "phase 2 must not re-arm the marker"
        # The one demonstration of a screen answer, and only on this pass.
        answers = [m["content"] for m in body["messages"] if m["role"] == "assistant"]
        assert any(llm.ANCHOR_SEEN[0][1] == a for a in answers), "no seen anchor on phase 2"

        # Phase 1, vision-capable: plain strings + REMINDER_LOOK.
        seeing = {**base, "llm": {**base["llm"], "vision": "on"}}
        await once(seeing, messages)
        body = seen["body"]
        user = [m for m in body["messages"] if m["role"] == "user"][-1]
        assert isinstance(user["content"], str), "image parts leaked into phase 1"
        last_system = [m for m in body["messages"] if m["role"] == "system"][-1]["content"]
        assert "exactly [look]" in last_system and "attached" not in last_system

        # Vision off: REMINDER_NOLOOK, and no marker instruction anywhere.
        blind = {**base, "llm": {**base["llm"], "vision": "off"}}
        await once(blind, messages)
        last_system = [m for m in seen["body"]["messages"] if m["role"] == "system"][-1][
            "content"
        ]
        assert "takes no images" in last_system and "[look]" not in last_system

        # Pointing: the same phase-2 image
        listing = point.describe([
            point.Target(0.03, 0.53, "API Keys", "ocr", 0.0),
            point.Target(0.40, 0.02, "TokenRouter - Chrome", "uia", 1.0),
        ])
        aiming = {
            **base,
            "llm": {**base["llm"], "screen": "seen", "point": True, "items": listing},
        }
        await once(aiming, messages, image=fake_jpeg)
        last_system = [m for m in seen["body"]["messages"] if m["role"] == "system"][-1][
            "content"
        ]
        assert "[POINT:n]" in last_system and "[POINT:none]" in last_system, last_system
        assert '"API Keys"' in last_system, "the list never reached the prompt"
        answers = [m["content"] for m in seen["body"]["messages"] if m["role"] == "assistant"]
        assert any(llm.ANCHOR_POINT[0][1] == a for a in answers), "no pointing anchor"
        # Both outcomes demonstrated
        assert any(llm.ANCHOR_POINT[1][1] == a for a in answers), "no [POINT:none] anchor"
        assert not any(llm.ANCHOR_SEEN[0][1] == a for a in answers), (
            "a markerless screen answer was shown alongside the pointing rule"
        )

        # A later step of the same walkthrough. Its rule is not REMINDER_SEEN's
        guiding = {**aiming, "llm": {**aiming["llm"], "screen": "guide"}}
        await once(guiding, messages, image=fake_jpeg)
        last_system = [m for m in seen["body"]["messages"] if m["role"] == "system"][-1][
            "content"
        ]
        assert "one step at a time" in last_system, last_system
        assert "[POINT:n]" in last_system, "the guide step lost the pick rule"
    finally:
        server.shutdown()

    print("ok  image parts ride only on phase 2, and each mode sends its rule")


def check_capture() -> None:
    """A real screenshot: decodable JPEG, sensible size, native res untouched."""
    import io

    from mellowd import capture

    grabbed = capture.grab()
    assert grabbed, "capture returned nothing"
    data, width, height, pixels = grabbed
    # The unshrunk frame
    assert pixels is not None and pixels.shape[2] == 3, getattr(pixels, "shape", None)
    assert max(pixels.shape[:2]) >= max(width, height), "the pixels were shrunk too"
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    image = io.BytesIO(data)
    from PIL import Image

    with Image.open(image) as im:
        # The size travels with the bytes now
        assert im.size == (width, height), f"{im.size} reported as {(width, height)}"
        long_edge = max(im.size)
        assert long_edge <= capture.MAX_EDGE, f"oversized: {im.size}"
        # 1080p-and-under monitors must go through untouched
        assert long_edge == min(long_edge, capture.MAX_EDGE) or long_edge == (
            capture.MAX_EDGE
        ), im.size
    assert len(data) < 1_500_000, f"{len(data)} bytes is not a shrunken JPEG"

    # A pointing turn gets a smaller frame
    small = capture.grab(capture.POINT_EDGE)
    assert small, "capture returned nothing for a pointing turn"
    tight, narrow, short, raw = small
    assert max(narrow, short) <= capture.POINT_EDGE, (narrow, short)
    assert capture.POINT_EDGE < capture.MAX_EDGE, "the two frames are one frame"
    # Still the full-resolution pixels: the OCR reads those
    assert max(raw.shape[:2]) >= max(pixels.shape[:2]), "the pointing pixels were shrunk"
    if max(width, height) > capture.POINT_EDGE:
        assert len(tight) < len(data), (len(tight), len(data))

    app, title = capture.foreground()
    print(
        f"ok  captured {long_edge}px JPEG ({len(data) // 1024}KB), "
        f"{max(narrow, short)}px ({len(tight) // 1024}KB) to point with"
        f" | foreground: {app or '?'!r} {title[:40]!r}"
    )


async def check_agents() -> None:
    """The agent brain, offline: registry, prompts, argv shapes, parsers."""
    from mellowd import agents, config, llm, main, point

    # The settings window renders straight out of this table
    assert set(config.AGENT_PRESETS) == {"claude", "codex"}, config.AGENT_PRESETS
    for agent_id, preset in config.AGENT_PRESETS.items():
        assert preset["label"] and preset["install"], agent_id
        assert preset["binaries"], agent_id
        assert "models" in preset and "models_cmd" in preset, agent_id
        assert agent_id in agents._PARSERS, f"no parser for {agent_id}"
    print("ok  agent registry complete (two CLIs, both with parsers)")

    section = {
        **config.DEFAULTS["llm"],
        "mode": "agent",
        "provider": "claude",
        "model": "",
        # Composed the way agents.chat composes it, placeholder intact.
        "system_prompt": llm.persona({"system_prompt": "", "llm": {"model": ""}}, "{model}"),
    }

    # One question, no history. The persona goes to --system-prompt so it replaces Claude Code's own
    system, user = agents.build_prompt(
        [{"role": "user", "content": "what is a waffle?"}], section
    )
    assert system.startswith("you are mellow"), system[:60]
    assert "Claude Code" in system, "{model} must fall back to the agent label"
    # _REMINDER_TONE opens "Reminder: you are mellow"
    assert system not in user, "the persona must not be sent twice"
    assert "They just said: what is a waffle?" in user, user[-200:]
    assert "[look]" in user, "a vision-capable agent must be offered the marker"
    print("ok  agent prompt splits persona from question and screen rule")

    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what is a waffle?"},
    ]
    _, user = agents.build_prompt(history, section)
    assert "Conversation so far:" in user
    assert "They said: hi" in user and "You answered: hello" in user
    assert user.index("You answered: hello") < user.index("They just said:")
    print("ok  agent prompt replays history inline")

    # Prompt context is now an agent-only latency budget.
    long_history = []
    for i in range(10):
        long_history.extend(
            (
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            )
        )
    standalone = [*long_history, {"role": "user", "content": "open downloads"}]
    fast = agents.select_history(standalone, "fast")
    assert len(fast) == 3 and fast[0]["content"] == "question 9", fast
    followup = agents.select_history(
        [*long_history, {"role": "user", "content": "do that again"}], "fast"
    )
    assert len(followup) == 7 and followup[0]["content"] == "question 7", followup
    balanced = agents.select_history(standalone, "balanced")
    assert len(balanced) == 7 and balanced[0]["content"] == "question 7", balanced
    deep = agents.select_history(standalone, "deep")
    assert len(deep) == 21 and deep[0]["content"] == "question 0", deep
    malformed = agents.select_history(
        [
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "complete"},
            {"role": "assistant", "content": "pair"},
            {"role": "user", "content": "current"},
        ],
        "fast",
    )
    assert [m["content"] for m in malformed] == ["complete", "pair", "current"]
    print("ok  agent history keeps complete relevant exchanges per speed preset")

    blind = {**section, "vision": "off"}
    assert "takes no images" in agents.build_prompt(history, blind)[1]
    _, seen = agents.build_prompt(history, section, seen=True)
    assert "attached to their latest message" in seen
    assert "[look]" not in seen, "a screenshot turn must not invite another"

    # The gap that let step 14 ship with agent mode silently unable to point.
    listing = point.describe([
        point.Target(0.03, 0.53, "API Keys", "ocr", 0.0),
        point.Target(0.40, 0.02, "TokenRouter - Chrome", "uia", 1.0),
    ])
    aiming = {**section, "screen": "seen", "point": True, "items": listing}
    _, ask = agents.build_prompt(history, aiming, seen=True)
    assert "[POINT:n]" in ask and "[POINT:none]" in ask, ask[-300:]
    assert "API Keys" in ask, "the agent never saw the list it must pick from"
    assert llm.ANCHOR_POINT[0][1] in ask, "no worked example reached the agent"
    assert llm.ANCHOR_POINT[1][1] in ask, "no [POINT:none] example reached the agent"

    # Nothing readable on screen means no list
    _, blindish = agents.build_prompt(history, {**aiming, "items": ""}, seen=True)
    assert "[POINT:" not in blindish, "asked for a pick with nothing to pick from"

    _, walk = agents.build_prompt(
        history, {**aiming, "screen": "guide"}, seen=True
    )
    assert "one step at a time" in walk, "the walkthrough rule never reached the agent"
    assert "[POINT:n]" in walk, "a walkthrough step could not point"

    _, broke = agents.build_prompt(history, {**section, "screen": "failed"})
    assert "capture failed" in broke, "a failed capture blamed the model instead"
    print("ok  agent prompts carry every screen rule llm has, pointing included")

    # agents.chat itself, with only the subprocess stubbed.
    talked = {}

    def fake_turn(provider, section, messages, image):
        talked["section"] = section
        return object()

    async def fake_stream(provider, turn):
        yield "hello"

    live = config.load()
    live["llm"] = {**live["llm"], "mode": "agent", "provider": "claude", "model": ""}
    real_turn, real_stream = agents._turn, agents._stream
    try:
        agents._turn, agents._stream = fake_turn, fake_stream
        said = "".join(
            [c async for c in agents.chat([{"role": "user", "content": "hi"}], live)]
        )
    finally:
        agents._turn, agents._stream = real_turn, real_stream
    assert said == "hello", said
    persona = talked["section"]["system_prompt"]
    assert persona.startswith("you are mellow"), persona[:60]
    # The placeholder has to survive to build_prompt
    assert "{model}" in persona, "the model placeholder was substituted too early"

    stub = ["stub-cli"]
    argv = agents.build_argv(stub, "claude", "PERSONA", "QUESTION")
    assert argv[-1] == "QUESTION", argv
    assert argv[argv.index("--system-prompt") + 1] == "PERSONA"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv and "--include-partial-messages" in argv
    # No tools at all. The screenshot rides inline now
    assert argv[argv.index("--tools") + 1] == "", argv
    for flag in ("--safe-mode", "--strict-mcp-config", "--setting-sources", "--no-session-persistence"):
        assert flag in argv, f"{flag} missing — claude would load the user's own config"
    pinned = agents.build_argv(stub, "claude", "P", "Q", model="sonnet")
    assert pinned[pinned.index("--model") + 1] == "sonnet" and pinned[-1] == "Q", pinned
    fast_argv = agents.build_argv(stub, "claude", "P", "Q", effort="low")
    assert fast_argv[fast_argv.index("--effort") + 1] == "low", fast_argv

    argv = agents.build_argv(stub, "claude", "P", "Q", has_image=True)
    assert argv[-2:] == ["--input-format", "stream-json"], argv
    assert "Q" not in argv, "with an image the question travels over stdin, not argv"

    argv = agents.build_argv(
        stub,
        "codex",
        "PERSONA",
        "QUESTION",
        image_path="s.jpg",
        schema_path="schema.json",
        prompt_stdin=True,
    )
    assert argv[1:3] == ["exec", "--json"], argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in argv
    assert "--ignore-user-config" in argv and "--ignore-rules" in argv
    assert argv[argv.index("-i") + 1] == "s.jpg"
    assert argv[argv.index("--output-schema") + 1] == "schema.json"
    # -i is variadic: without the separator the prompt becomes a second image path and Codex silently
    assert argv[-2] == "--", argv
    # The persona is deliberately absent
    assert argv[-1] == "-", argv
    codex_fast = agents.build_argv(
        stub, "codex", "P", "Q", prompt_stdin=True, effort="low"
    )
    assert 'model_reasoning_effort="low"' in codex_fast, codex_fast
    assert agents._effort_rejected("unknown configuration key model_reasoning_effort")
    assert agents._effort_rejected("unexpected argument '--effort'")
    assert not agents._effort_rejected("usage limit reached")
    print("ok  argv builders: no tools, trimmed harness, codex prompt separated")

    codex_section = {**section, "provider": "codex"}
    real_workspace = agents.WORKSPACE
    agents.WORKSPACE = Path.cwd()
    turn = agents._turn("codex", codex_section, [{"role": "user", "content": "hi"}])
    try:
        brief = (turn.cwd / "AGENTS.md").read_text(encoding="utf-8")
        assert brief.startswith("you are mellow"), brief[:80]
        assert turn.payload and b"They just said: hi" in turn.payload
        assert turn.cwd != agents.WORKSPACE, "Codex turns share mutable files"
        assert turn.fallback_argv is not None, "Fast needs a no-effort fallback"
        assert 'model_reasoning_effort="low"' in turn.argv, turn.argv
        assert 'model_reasoning_effort="low"' not in turn.fallback_argv
    finally:
        turn.cleanup()
        agents.WORKSPACE = real_workspace
    print("ok  codex persona lands in AGENTS.md, its own instruction channel")

    payload = agents._payload("QUESTION", b"\xff\xd8jpegbytes")
    assert payload is not None and payload.endswith(b"\n")
    body = json.loads(payload)
    parts = body["message"]["content"]
    # Image first, question last — end of context is the privileged position
    assert parts[0]["type"] == "image" and parts[0]["source"]["media_type"] == "image/jpeg"
    assert parts[-1] == {"type": "text", "text": "QUESTION"}
    assert agents._payload("Q", None) is None
    print("ok  claude stdin message carries the screenshot inline, image first")

    # --- real captured stream lines ------------------------------------------
    fam: dict = {}
    chunks: list[str] = []
    for line in (
        '{"type":"system","subtype":"init","session_id":"s1"}',
        '{"type":"stream_event","event":{"type":"content_block_delta",'
        '"delta":{"type":"text_delta","text":"Hel"}}}',
        '{"type":"stream_event","event":{"type":"content_block_delta",'
        '"delta":{"type":"text_delta","text":"lo"}}}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}',
        '{"type":"result","subtype":"success","result":"Hello","session_id":"s1"}',
    ):
        chunks += agents._parse_family(line, fam)
    assert chunks == ["Hel", "lo"], chunks  # the settled message hides behind deltas
    assert fam.get("session_id") == "s1", fam

    structured: dict = {}
    agents._parse_family(
        '{"type":"result","subtype":"success","structured_output":'
        '{"selection":"E1","answer":"Click Extensions."}}',
        structured,
    )
    assert json.loads(structured["result_text"])["selection"] == "E1", structured

    err_state: dict = {}
    agents._parse_family(
        '{"type":"result","subtype":"error_during_execution","result":"not logged in"}',
        err_state,
    )
    assert err_state.get("error") == "not logged in", err_state

    # Codex, captured from `codex exec --json` on a live account.
    cod: dict = {}
    chunks = []
    for line in (
        '{"type":"thread.started","thread_id":"01a0"}',
        '{"type":"item.completed","item":{"id":"item_0","type":"error",'
        '"message":"Model metadata for `x` not found. Defaulting to fallback metadata"}}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"item_1","type":"agent_message",'
        '"text":"Your screen shows a terminal."}}',
    ):
        chunks += agents._parse_codex(line, cod)
    assert chunks == ["Your screen shows a terminal."], chunks
    # A nested error item is a warning the run recovers from; treating it as fatal would kill turns
    assert "error" not in cod, cod

    # And the real failure shape — on stdout, not stderr
    cod_err: dict = {}
    assert agents._parse_codex(
        '{"type":"error","message":"{\\"type\\":\\"error\\",\\"status\\":400,'
        '\\"error\\":{\\"message\\":\\"The \'gpt-5.6-sol\' model is not supported '
        'when using Codex with a ChatGPT account.\\"}}"}',
        cod_err,
    ) == []
    assert "not supported when using" in cod_err["error"], cod_err
    turn_failed: dict = {}
    agents._parse_codex(
        '{"type":"turn.failed","error":{"message":"stream disconnected"}}', turn_failed
    )
    assert turn_failed["error"] == "stream disconnected", turn_failed

    # The older wire shapes still work, because these CLIs change them.
    old: dict = {}
    assert agents._parse_codex(
        '{"msg":{"type":"agent_message_delta","delta":"hi"}}', old
    ) == ["hi"]
    assert agents._parse_codex(
        '{"msg":{"type":"agent_message","message":"hi there"}}', old
    ) == []  # deltas already carried this turn
    print("ok  stream parsers match real captured claude and codex output")

    found = agents._parse_models(
        "codex",
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-5.5", "display_name": "GPT-5.5", "visibility": "list"},
                    {
                        "slug": "gpt-5.4-mini",
                        "display_name": "GPT-5.4 Mini",
                        "visibility": "list",
                    },
                    {"slug": "auto-review", "display_name": "Hidden", "visibility": "hide"},
                ]
            }
        ),
    )
    assert found == {"gpt-5.5": "GPT-5.5"}, found
    real_models = agents.models
    try:
        agents.models = lambda agent_id, refresh=False: {"gpt-5.5": "GPT-5.5"}
        agents.require_exact_model("codex", "gpt-5.5")
        try:
            agents.require_exact_model("codex", "gpt-5.4-mini")
            raise AssertionError("a routed legacy model was accepted")
        except ValueError as e:
            assert "will not let the agent silently substitute" in str(e), e
    finally:
        agents.models = real_models
    # Claude has no list command, so the preset aliases are the list.
    assert agents.models("claude") == config.AGENT_PRESETS["claude"]["models"]
    print("ok  model lists omit routed entries and exact selections are enforced")

    err = agents._failure("claude", "Please run /login to authenticate")
    assert "signed in" in str(err), err
    err = agents._failure("codex", "usage limit reached for your plan")
    assert "limit" in str(err), err
    # The live failure on a real account: the configured model is not on the plan.
    err = agents._failure(
        "codex", "The 'gpt-5.6-sol' model is not supported when using Codex with a ChatGPT account."
    )
    assert "Model list" in str(err), err
    err = agents._failure("claude", '[claude-code:unrecognized_model] {"model":"Sonnet 4.8"}')
    assert "Model list" in str(err), err
    print("ok  failure wording names the fix")

    candidate = {
        **config.DEFAULTS,
        "llm": {
            **config.DEFAULTS["llm"],
            "mode": "agent",
            "provider": "claude",
            "model": "",
            "base_url": "",
            "api_key": "",
        },
    }
    saved = config.validate(candidate)["llm"]
    assert saved["base_url"] == "" and saved["api_key"] == "", saved
    assert saved["agent_speed"] == "fast", saved
    old_llm = dict(config.DEFAULTS["llm"])
    old_llm.pop("agent_speed")
    assert config.validate({**config.DEFAULTS, "llm": old_llm})["llm"]["agent_speed"] == "fast"
    try:
        config.validate({**candidate, "llm": {**saved, "agent_speed": "turbo"}})
        raise AssertionError("an unknown agent speed was accepted")
    except ValueError:
        pass
    fast_cfg = config.validate(candidate)
    deep_cfg = config.validate(
        {**candidate, "llm": {**candidate["llm"], "agent_speed": "deep"}}
    )
    assert main._engine_signature(fast_cfg) == main._engine_signature(deep_cfg)
    try:
        config.validate({**candidate, "llm": {**saved, "provider": "nope"}})
        raise AssertionError("an unknown agent was accepted")
    except ValueError:
        pass
    try:
        config.validate({**config.DEFAULTS, "stt": {**config.DEFAULTS["stt"], "mode": "agent"}})
        raise AssertionError("stt accepted agent mode")
    except ValueError:
        pass
    # A config parked on one of the four retired CLIs must not take the whole sidecar down on load
    retired = config.migrate(
        {"llm": {"mode": "agent", "provider": "antigravity", "model": "gemini-3.7-flash-high"}}
    )["llm"]
    assert retired == {"mode": "agent", "provider": "claude", "model": ""}, retired
    print("ok  config: agent mode validates, retired agents migrate, keys preserved")

    # Connect verifies the actual image+schema route
    real_complete_vision = agents.complete_vision
    try:
        async def fake_capability(prompt, cfg, image, schema=None):
            assert image.startswith(b"\xff\xd8"), "capability probe did not send a JPEG"
            assert schema and schema["properties"]["selection"]["enum"] == ["E1"]
            return '{"selection":"E1"}'

        agents.complete_vision = fake_capability
        capable, detail = await agents.check_capabilities("codex", "")
    finally:
        agents.complete_vision = real_complete_vision
    assert capable and "images" in detail, detail
    print("ok  Connect capability probe verifies image plus structured output")

    # Keep the token-consuming live probe opt-in.
    for agent_id in config.AGENT_PRESETS:
        if agents.find(agent_id) is None:
            print(f"..  {agent_id} not installed; native auth status unavailable")
            continue
        signed, detail = agents.auth_status(agent_id)
        print(f"ok  {agent_id} native auth status: {'signed in' if signed else detail}")
    if os.environ.get("MELLOW_LIVE_AGENT_CHECK") != "1":
        print("..  live agent model turn skipped (set MELLOW_LIVE_AGENT_CHECK=1 to spend one turn)")
        return

    live_agent = str(os.environ.get("MELLOW_LIVE_AGENT", "claude")).lower()
    if live_agent not in config.AGENT_PRESETS:
        print(f"..  unknown live agent {live_agent!r}")
        return
    if agents.find(live_agent) is None:
        print(f"..  {live_agent} not installed; live agent turn unverified here")
        return
    entry = [a for a in agents.catalog() if a["id"] == live_agent][0]
    assert entry["installed"], entry
    try:
        capable, answer = await asyncio.wait_for(
            agents.check_capabilities(live_agent, ""), 75
        )
        if not capable:
            raise RuntimeError(answer)
        print(f"ok  live {live_agent} image+schema turn: {answer[:60]}")
    except Exception as e:
        print(f"..  {live_agent} cli present but live probe failed ({e})")


async def check_agent_connect() -> None:
    """Connect saves only after native auth and vision/model verification."""
    from mellowd import agents, main

    seen = []
    saved = (
        agents.find,
        agents.require_exact_model,
        agents.auth_status,
        agents.check_capabilities,
    )
    try:
        agents.find = lambda agent_id: ["stub-cli"]
        agents.require_exact_model = lambda agent_id, model: None
        agents.auth_status = lambda agent_id: (True, "oauth")

        async def capability(agent_id, model, agent_speed):
            seen.append((agent_id, model, agent_speed))
            return True, "vision works"

        agents.check_capabilities = capability
        result = await main.agent_login({"agent": "codex", "model": "gpt-test"})
    finally:
        (
            agents.find,
            agents.require_exact_model,
            agents.auth_status,
            agents.check_capabilities,
        ) = saved
    assert seen == [("codex", "gpt-test", "fast")], seen
    assert result == {
        "ok": True,
        "installed": True,
        "signed_in": True,
        "model_ok": True,
        "vision_ok": True,
        "detail": "vision works",
    }, result
    print("ok  Connect verifies the selected agent model before settings save")


async def main() -> None:
    check_config()
    check_migration()
    check_wav()
    check_resample()
    check_audio_selection()
    check_devices()
    check_stream_failure()
    check_warm_signatures()
    check_warmup()
    check_errors()
    check_reminders()
    check_sessions()
    await check_turn_logging()
    check_preroll()
    check_sentences()
    await check_speech_pipeline()
    check_stale_refusals()
    check_thinking_filter()
    check_opener_filter()
    check_tone_contract()
    check_vision()
    check_wants_screen()
    check_wants_pointing()
    check_vision_probe()
    await check_marker_hold()
    await check_point_marker()
    check_point_score()
    check_point_pick()
    await check_locator()
    check_point_list()
    await check_point_first()
    check_dpi()
    check_ocr()
    check_uia()
    check_screen_change()
    await check_one_turn_one_bone()
    await check_turn_monitor_lock()
    await check_act()
    await check_screen_request()
    check_capture()
    await check_agents()
    await check_agent_connect()
    await check_pet_only()
    await check_temperature()
    check_elevenlabs()
    check_openrouter()
    samples, rate = check_tts()
    check_stt_speech(samples, rate)

    async with connect(URL) as ws:
        hello = await recv(ws)
        assert hello == {"type": "state", "state": "idle"}, f"bad greeting: {hello}"
        # The shell needs the voice flag before the user right-clicks
        voice = await recv(ws)
        assert voice["type"] == "speak", f"expected speak flag, got {voice}"
        assert isinstance(voice["value"], bool), f"speak must be a bool: {voice}"
        print(f"ok  handshake -> idle, speak={voice['value']}")

        await ws.send(json.dumps({"type": "ping", "text": "selfcheck"}))
        pong = await recv(ws)
        assert pong["type"] == "pong", f"expected pong, got {pong}"
        assert pong["echo"] == "selfcheck", f"echo mangled: {pong}"
        print("ok  ping/pong round trip")

        await ws.send(json.dumps({"type": "nonsense"}))
        err = await recv(ws)
        assert err["type"] == "error", f"unknown message should error, got {err}"
        print("ok  unknown message rejected")

        # Mute round trip, then straight back — this writes config
        await ws.send(json.dumps({"type": "set_speak", "value": not voice["value"]}))
        flipped = await recv(ws)
        assert flipped == {"type": "speak", "value": not voice["value"]}, flipped
        await ws.send(json.dumps({"type": "set_speak", "value": voice["value"]}))
        assert (await recv(ws))["value"] == voice["value"]
        print("ok  mute toggles and restores")

        # Waking starts the quiet keeper and reports its real readiness.
        await ws.send(json.dumps({"type": "awake", "value": True}))
        await ws.send(json.dumps({"type": "ping", "text": "awake"}))
        mic_ready = False
        while True:
            alive = await recv(ws, timeout=30.0)
            if alive["type"] == "microphone":
                mic_ready = alive["state"] == "ready"
                continue
            assert alive["type"] == "pong", f"awake broke the socket: {alive}"
            break
        while not mic_ready:
            mic = await recv(ws, timeout=30.0)
            assert mic["type"] == "microphone", f"expected mic readiness, got {mic}"
            mic_ready = mic["state"] == "ready"
        print("ok  mic warms without blocking the socket, then reports ready")

        # Must exceed stt.MIN_SECONDS or transcribe() short-circuits and the model never loads
        await ws.send(json.dumps({"type": "ptt_start"}))
        assert (await recv(ws))["state"] == "listening"
        await asyncio.sleep(1.5)
        await ws.send(json.dumps({"type": "ptt_end"}))

        assert (await recv(ws))["state"] == "thinking"
        tr = await recv(ws, timeout=180.0)  # first run downloads the model
        assert tr["type"] == "transcript", f"expected transcript, got {tr}"
        await drain_until_idle(ws)
        if tr["text"] == "…didn't catch that":
            print("..  push-to-talk mic path reached; live recognition unverified (silence)")
        else:
            print(f"ok  push-to-talk -> mic -> whisper (heard: {tr['text']!r})")

        # Typed path exercises the LLM without depending on what the mic heard.
        from mellowd import config

        cfg = config.load()
        llm = cfg["llm"]
        print(f"..  asking {llm['provider']}/{llm['model']} (first call loads the model)")
        await ws.send(json.dumps({"type": "text", "text": "say hello in five words"}))
        assert (await recv(ws))["state"] == "thinking"
        reply, first, talked = await drain_until_idle(ws)

        assert reply.strip(), "llm returned an empty reply"
        print(f"ok  llm streamed {len(reply)} chars, first token {first:.2f}s")
        print(f"    reply: {reply.strip()[:120]}")
        if cfg["tts"]["speak"]:
            assert talked, "turn finished without ever entering the talking state"
            print("ok  spoke the reply out loud")

        # Identity, through the real socket and the real model.
        await ws.send(json.dumps({"type": "text", "text": "what is your name?"}))
        assert (await recv(ws))["state"] == "thinking"
        who, _, _ = await drain_until_idle(ws)
        assert "mellow" in who.lower(), f"model would not answer to mellow: {who!r}"
        print(f"ok  answers to its own name: {who.strip()[:80]}")

        # The turns above must have reached disk, not just the screen.
        import httpx

        base = "http://127.0.0.1:8765"  # same sidecar URL points at
        listing = httpx.get(f"{base}/history", timeout=10).json()
        assert listing["sessions"], "no sessions in the index after two live turns"
        newest = listing["sessions"][0]
        logged = httpx.get(f"{base}/history/{newest['id']}", timeout=10).json()["events"]
        kinds = [e["type"] for e in logged]
        assert "user_said" in kinds and "assistant_said" in kinds, kinds
        said = [e for e in logged if e["type"] == "assistant_said"][-1]
        assert said["model"] == cfg["llm"]["model"], said
        print(f"ok  live turns logged ({len(kinds)} events, model {said['model']})")

        # Barge-in: ask for something long, cut it off mid-sentence.
        if cfg["tts"]["speak"]:
            await ws.send(
                json.dumps({"type": "text", "text": "count slowly from one to twenty"})
            )
            assert (await recv(ws))["state"] == "thinking"
            while (await recv(ws, 120.0)).get("state") != "talking":
                pass  # wait until audio is actually playing

            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "cancel"}))
            while True:
                m = await recv(ws, 30.0)
                if m["type"] == "state" and m["state"] == "idle":
                    break
            took = time.perf_counter() - t0
            assert took < 5.0, f"barge-in took {took:.1f}s — audio isn't being cut off"
            print(f"ok  barge-in stopped speech in {took:.2f}s")

    print("\nall checks passed")


if __name__ == "__main__":
    asyncio.run(main())
