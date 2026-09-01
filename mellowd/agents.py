"""Coding-agent CLIs as Mellow's answer model."""

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

# llm at module level
from mellowd import config, llm

log = logging.getLogger("mellowd.agents")

# Keep CLIs outside project trees so they cannot read repository memory files.
WORKSPACE = config.CONFIG_DIR / "agents"

# Keeps a console window from flashing behind the pet on every turn.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# How long "what models does this account have" may take before the settings window gives up
MODELS_TIMEOUT = 20.0

# A small, deterministic context router is faster and more predictable than spending another model
_HISTORY_LIMITS = {
    "fast": (1, 3),
    "balanced": (3, 6),
    "deep": (10, 10),
}
_FOLLOW_UP = re.compile(
    r"(?:\b(?:it|that|this|those|them|there|same|again|continue|previous|earlier)\b"
    r"|\b(?:tell me more|what about|how about|and then)\b)",
    re.IGNORECASE,
)

# A new CLI can reject an effort flag even though the rest of its protocol is compatible.
_EFFORT_UNSUPPORTED: set[tuple[str, str]] = set()


@dataclass
class Invocation:
    """One isolated CLI process and everything needed to run it safely."""

    argv: list[str]
    fallback_argv: list[str] | None
    payload: bytes | None
    cwd: Path
    effort_key: tuple[str, str] | None = None
    image_bytes: int = 0
    image_transport: str = "none"
    temporary: Path | None = None

    def cleanup(self) -> None:
        if self.temporary is not None:
            shutil.rmtree(self.temporary, ignore_errors=True)
            self.temporary = None


def find(agent_id: str) -> list[str] | None:
    """The argv prefix that runs this agent, or None if it isn't installed."""
    preset = config.AGENT_PRESETS[agent_id]
    for name in preset["binaries"]:
        for ext in ("", ".exe", ".cmd", ".bat"):
            hit = shutil.which(name + ext)
            if hit:
                return [hit]
    return None


# Model lists are per account and cost a subprocess to fetch
_MODELS: dict[str, dict[str, str]] = {}

# Codex currently accepts this legacy slug but serves GPT-5.6 Luna instead.
_CODEX_ROUTED_MODELS = {"gpt-5.4-mini"}


def _parse_models(agent_id: str, out: str) -> dict[str, str]:
    """One CLI's model listing, as {value the --model flag takes: what to show}."""
    if agent_id == "codex":
        # {"models":[{"slug":…,"display_name":…,"visibility":"list"|"hide"}]}.
        data = json.loads(out)
        return {
            m["slug"]: m.get("display_name") or m["slug"]
            for m in data.get("models", [])
            # `upgrade` means Codex may accept this legacy slug while actually serving its replacement.
            if (
                m.get("slug")
                and m.get("visibility") != "hide"
                and not m.get("upgrade")
                and m.get("slug") not in _CODEX_ROUTED_MODELS
            )
        }
    return {}


def models(agent_id: str, refresh: bool = False) -> dict[str, str]:
    """What this account can actually run, asked of the CLI itself."""
    preset = config.AGENT_PRESETS[agent_id]
    fallback = dict(preset["models"])
    if not preset["models_cmd"]:
        return fallback
    if refresh:
        _MODELS.pop(agent_id, None)
    elif agent_id in _MODELS:
        return _MODELS[agent_id]

    prefix = find(agent_id)
    if prefix is None:
        return fallback
    try:
        done = subprocess.run(
            [*prefix, *preset["models_cmd"]],
            cwd=str(WORKSPACE) if WORKSPACE.exists() else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=MODELS_TIMEOUT,
            creationflags=CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL,
        )
        found = _parse_models(agent_id, done.stdout)
    except Exception as e:
        log.warning("%s model list failed: %s", preset["label"], e)
        return fallback
    if not found:
        return fallback
    _MODELS[agent_id] = found
    return found


def require_exact_model(agent_id: str, model: str, refresh: bool = False) -> None:
    """Reject an explicit model unless the CLI promises to run it as-is."""
    selected = str(model or "").strip()
    if not selected:
        return
    if selected in models(agent_id, refresh):
        return
    label = config.AGENT_PRESETS[agent_id]["label"]
    raise ValueError(
        f"{selected} is not available as an exact {label} model. "
        "Choose another model from the refreshed Model list; Mellow will not "
        "let the agent silently substitute one."
    )


def catalog(refresh: bool = False) -> list[dict]:
    """Everything the settings window shows, detection included."""
    out = []
    for agent_id, preset in config.AGENT_PRESETS.items():
        prefix = find(agent_id)
        signed_in, auth_detail = auth_status(agent_id) if prefix else (False, "not installed")
        out.append(
            {
                "id": agent_id,
                "label": preset["label"],
                "installed": prefix is not None,
                "path": prefix[0] if prefix else "",
                "install": preset["install"],
                "vision": preset["vision"],
                # Only installed agents get asked
                "models": models(agent_id, refresh) if prefix else dict(preset["models"]),
                "signed_in": signed_in,
                "auth_detail": auth_detail,
            }
        )
    return out


AUTH_STATUS_ARGS = {
    "codex": ["login", "status"],
    "claude": ["auth", "status"],
}
LOGIN_ARGS = {
    "codex": ["login"],
    "claude": ["auth", "login"],
}


def auth_status(agent_id: str) -> tuple[bool, str]:
    """Use each CLI's native, non-token-burning authentication status."""
    prefix = find(agent_id)
    if prefix is None:
        return False, "not installed"
    try:
        done = subprocess.run(
            [*prefix, *AUTH_STATUS_ARGS[agent_id]],
            cwd=str(WORKSPACE) if WORKSPACE.exists() else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=MODELS_TIMEOUT,
            creationflags=CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        return False, str(e)[:160]
    detail = (done.stdout or done.stderr).strip()
    if done.returncode != 0:
        return False, detail[:160] or "not signed in"
    if agent_id == "claude":
        try:
            data = json.loads(done.stdout)
            return bool(data.get("loggedIn")), str(data.get("authMethod") or "signed in")
        except (TypeError, ValueError):
            pass
    return True, detail.splitlines()[0][:160] if detail else "signed in"


def login(agent_id: str) -> None:
    """Open the CLI's own sign-in flow in a visible console window."""
    prefix = find(agent_id)
    if prefix is None:
        raise RuntimeError(f"{config.AGENT_PRESETS[agent_id]['label']} is not installed")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen([*prefix, *LOGIN_ARGS[agent_id]], cwd=WORKSPACE, creationflags=flags)


def _reminder(section: dict, seen: bool) -> str:
    """Which screen rule rides on this prompt."""
    from mellowd import llm

    return llm._reminder_for(
        {
            **section,
            # llm reads a mode string where this side has always had a bool.
            "screen": section.get("screen") or ("seen" if seen else ""),
            # _reminder_for defaults this true for probes that bypass config.
            "vision_ok": config.resolves_vision(section),
        }
    )


def _examples(section: dict) -> str:
    """The one-shot exchanges, flattened into the agent's single user message."""
    from mellowd import llm

    screen = section.get("screen") or ""
    if screen not in ("seen", "guide"):
        return ""
    pairs = llm.ANCHOR_POINT if section.get("items") else llm.ANCHOR_SEEN
    lines = []
    for question, answer in pairs:
        lines.append(f"They said: {question}")
        lines.append(f"You answered: {answer}")
    return "For example:\n" + "\n".join(lines)


def select_history(messages: list[dict], speed: str = "fast") -> list[dict]:
    """Return the current question plus only the useful trailing exchanges."""
    if not messages:
        return []
    current = messages[-1]
    question = str(current.get("content", ""))
    ordinary, dependent = _HISTORY_LIMITS.get(speed, _HISTORY_LIMITS["fast"])
    limit = dependent if _FOLLOW_UP.search(question) else ordinary

    exchanges: list[tuple[dict, dict]] = []
    pending_user: dict | None = None
    for message in messages[:-1]:
        role = message.get("role")
        if role == "user":
            pending_user = message
        elif role == "assistant" and pending_user is not None:
            exchanges.append((pending_user, message))
            pending_user = None

    selected: list[dict] = []
    for user, assistant in exchanges[-limit:]:
        selected.extend((user, assistant))
    selected.append(current)
    return selected


def build_prompt(
    messages: list[dict], section: dict, seen: bool = False
) -> tuple[str, str]:
    """One turn as (system, user)."""
    preset = config.AGENT_PRESETS[section["provider"]]
    display = section.get("model") or preset["label"]
    system = section.get("system_prompt", "").replace("{model}", display)

    speed = str(section.get("agent_speed") or "fast")
    messages = select_history(messages, speed)
    question = str(messages[-1].get("content", "")) if messages else ""
    parts = []
    prior = messages[:-1]
    log.debug("agent context preset=%s prior_messages=%d", speed, len(prior))
    if prior:
        lines = []
        for m in prior:
            who = "They said" if m.get("role") == "user" else "You answered"
            lines.append(f"{who}: {m.get('content', '')}")
        parts.append("Conversation so far:\n" + "\n".join(lines))
    parts.append(f"They just said: {question}")
    aware = {**section, "screen": section.get("screen") or ("seen" if seen else "")}
    examples = _examples(aware)
    if examples:
        parts.append(examples)
    parts.append(_reminder(section, seen))
    return system, "\n\n".join(parts)


# Everything Claude Code does on the way to a first token that Mellow has no use for. Measured
CLAUDE_TRIM = (
    "--safe-mode",
    "--tools",
    "",
    "--strict-mcp-config",
    "--mcp-config",
    '{"mcpServers":{}}',
    "--setting-sources",
    "",
    "--disable-slash-commands",
    "--no-session-persistence",
)


def build_argv(
    prefix: list[str],
    agent_id: str,
    system: str,
    user: str,
    model: str = "",
    has_image: bool = False,
    image_path: str | None = None,
    schema: dict | None = None,
    schema_path: str | None = None,
    prompt_stdin: bool = False,
    effort: str = "",
) -> list[str]:
    """The headless command for one turn, per agent."""
    extra = ["--model", model] if model else []

    if agent_id == "claude":
        # stream-json output without --verbose is rejected
        argv = [
            *prefix,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--system-prompt",
            system,
            *CLAUDE_TRIM,
            *(["--effort", effort] if effort else []),
            *extra,
        ]
        if has_image:
            # The question travels inside the stdin message instead
            argv = [*argv, "--input-format", "stream-json"]
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
        if has_image:
            return argv
        return [*argv, user]

    if agent_id == "codex":
        # `system` is deliberately unused here
        argv = [
            *prefix,
            "exec",
            "--json",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            *(
                ["--config", f'model_reasoning_effort="{effort}"']
                if effort
                else []
            ),
        ]
        if image_path:
            argv += ["-i", image_path]
        if schema_path:
            argv += ["--output-schema", schema_path]
        argv += extra
        # The `--` is load-bearing: -i is variadic
        return [*argv, "--", "-" if prompt_stdin else user]

    raise RuntimeError(f"no argv builder for {agent_id}")


def _payload(user: str, image: bytes | None) -> bytes | None:
    """Claude's stdin message when a screenshot is attached, else None."""
    if image is None:
        return None
    message = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(image).decode("ascii"),
                    },
                },
                {"type": "text", "text": user},
            ],
        },
    }
    return (json.dumps(message) + "\n").encode("utf-8")


def _prepare(
    agent_id: str,
    section: dict,
    system: str,
    user: str,
    image: bytes | None = None,
    schema: dict | None = None,
) -> Invocation:
    """Build one isolated invocation from an already separated prompt."""
    prefix = find(agent_id)
    if prefix is None:
        raise RuntimeError(
            f"{config.AGENT_PRESETS[agent_id]['label']} is not installed "
            "- Settings lists the install command"
        )
    # Run this before a model process exists.
    require_exact_model(agent_id, section.get("model", ""))
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    temporary = None
    cwd = WORKSPACE
    image_path = None
    schema_path = None
    inline = image is not None and agent_id == "claude"
    if agent_id == "codex":
        cwd = WORKSPACE / f"turn-{uuid.uuid4().hex}"
        cwd.mkdir()
        temporary = cwd
        (cwd / "AGENTS.md").write_text(system, encoding="utf-8")
        if image is not None:
            image_path = str(cwd / "screen.jpg")
            (cwd / "screen.jpg").write_bytes(image)
        if schema is not None:
            schema_path = str(cwd / "output.schema.json")
            (cwd / "output.schema.json").write_text(
                json.dumps(schema, separators=(",", ":")), encoding="utf-8"
            )
    speed = str(section.get("agent_speed") or "fast")
    effort = config.AGENT_SPEED_EFFORT.get(speed, "low")
    effort_key = (agent_id, str(section.get("model", "")))
    if effort_key in _EFFORT_UNSUPPORTED:
        effort = ""
    argv = build_argv(
        prefix,
        agent_id,
        system,
        user,
        section.get("model", ""),
        has_image=inline,
        image_path=image_path,
        schema=schema,
        schema_path=schema_path,
        prompt_stdin=agent_id == "codex",
        effort=effort,
    )
    fallback_argv = None
    if effort:
        fallback_argv = build_argv(
            prefix,
            agent_id,
            system,
            user,
            section.get("model", ""),
            has_image=inline,
            image_path=image_path,
            schema=schema,
            schema_path=schema_path,
            prompt_stdin=agent_id == "codex",
        )
    if inline:
        payload = _payload(user, image)
        transport = "inline"
    elif agent_id == "codex":
        payload = (user + "\n").encode("utf-8")
        transport = "-i" if image is not None else "none"
    else:
        payload = None
        transport = "none"
    return Invocation(
        argv=argv,
        fallback_argv=fallback_argv,
        payload=payload,
        cwd=cwd,
        effort_key=effort_key if effort else None,
        image_bytes=len(image or b""),
        image_transport=transport,
        temporary=temporary,
    )


def _turn(
    agent_id: str,
    section: dict,
    messages: list[dict],
    image: bytes | None = None,
    schema: dict | None = None,
) -> Invocation:
    """Detection, prompt, argv and stdin for one turn — the single entry point."""
    system, user = build_prompt(messages, section, seen=image is not None)
    return _prepare(agent_id, section, system, user, image, schema)


def _probe_section(
    agent_id: str, model: str = "", agent_speed: str = "fast"
) -> dict:
    """The settings-window probe: one real turn, no persona, no screen rule."""
    return {
        "mode": "agent",
        "provider": agent_id,
        "model": model,
        "agent_speed": agent_speed,
        "vision": "off",
        "system_prompt": "",
    }


async def check_signed_in(
    agent_id: str, model: str = "", agent_speed: str = "fast"
) -> tuple[bool, str]:
    """Ask the CLI itself, with one tiny real turn."""
    messages = [{"role": "user", "content": "Reply with only the word connected."}]
    try:
        turn = _turn(
            agent_id, _probe_section(agent_id, model, agent_speed), messages
        )
        chunks: list[str] = []
        async for chunk in _stream(agent_id, turn):
            chunks.append(chunk)
            if len("".join(chunks)) >= 80:
                break
    except Exception as e:
        return False, str(e)
    answer = "".join(chunks).strip()
    if not answer:
        return False, "empty reply"
    return True, answer[:60]


async def check_capabilities(
    agent_id: str, model: str = "", agent_speed: str = "fast"
) -> tuple[bool, str]:
    """Verify the selected model's image and structured-output path."""
    import io

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (160, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((44, 22, 116, 74), fill=(220, 45, 90), outline="black", width=3)
    draw.text((68, 40), "E1", fill="white")
    encoded = io.BytesIO()
    image.save(encoded, "JPEG", quality=90)
    schema = {
        "type": "object",
        "properties": {"selection": {"type": "string", "enum": ["E1"]}},
        "required": ["selection"],
        "additionalProperties": False,
    }
    cfg = {"llm": _probe_section(agent_id, model, agent_speed)}
    try:
        raw = await complete_vision(
            "Select the magenta rectangle labelled E1. Return the schema object.",
            cfg,
            encoded.getvalue(),
            schema,
        )
        data = json.loads(raw.strip().strip("`").removeprefix("json").strip())
    except Exception as e:
        return False, str(e)[:240]
    if not isinstance(data, dict) or data.get("selection") != "E1":
        return False, f"vision check returned {raw[:160] or 'nothing'}"
    return True, "signed in; selected model can see images and return grounded output"


def _parse_family(line: str, state: dict) -> list[str]:
    """Claude-family NDJSON."""
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    if not isinstance(obj, dict):
        return []

    kind = obj.get("type")
    if kind == "stream_event":
        event = obj.get("event") or {}
        if event.get("type") == "content_block_delta":
            delta = event.get("delta") or {}
            text = delta.get("text")
            if delta.get("type") == "text_delta" and text:
                state["mode"] = "delta"
                state["emitted"] = True
                return [text]
        return []

    if kind == "assistant":
        content = (obj.get("message") or {}).get("content") or []
        joined = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if joined:
            state.setdefault("finals", []).append(joined)
        return []

    if kind == "result":
        if obj.get("session_id"):
            state["session_id"] = obj["session_id"]
        subtype = str(obj.get("subtype", ""))
        if subtype.startswith("error"):
            # Claude reports auth/quota failures as error results with exit code 0
            state["error"] = str(obj.get("result") or subtype)
        elif obj.get("structured_output") is not None:
            # Claude's --json-schema result can be carried separately from the ordinary text result.
            state["result_text"] = json.dumps(
                obj["structured_output"], separators=(",", ":")
            )
        elif obj.get("result"):
            state["result_text"] = str(obj["result"])
        if isinstance(obj.get("usage"), dict):
            state["usage"] = obj["usage"]
    return []


def _parse_codex(line: str, state: dict) -> list[str]:
    """Codex exec --json: newline-delimited progress events."""
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    if not isinstance(obj, dict):
        return []

    kind = str(obj.get("type") or "")
    if kind == "error":
        state["error"] = str(obj.get("message") or "codex reported an error")
        return []
    if kind == "turn.failed":
        err = obj.get("error") if isinstance(obj.get("error"), dict) else {}
        state["error"] = str(err.get("message") or "the turn failed")
        return []
    if kind == "turn.completed":
        if isinstance(obj.get("usage"), dict):
            state["usage"] = obj["usage"]
        return []

    msg = obj.get("msg") if isinstance(obj.get("msg"), dict) else obj
    item = msg.get("item") if isinstance(msg.get("item"), dict) else {}
    kind = str(msg.get("type") or kind)
    text = ""

    if kind == "agent_message_delta":
        text = str(msg.get("delta") or "")
        if text:
            state["mode"] = "delta"
            state["emitted"] = True
            return [text]
        return []

    if kind == "agent_message":
        text = str(msg.get("message") or "")
    elif kind in ("item.completed", "item.updated"):
        # The current shape: the finished answer is an item of type agent_message carrying `text`.
        if item.get("type") == "agent_message":
            text = str(item.get("text") or "")
    if not text:
        if msg.get("last_agent_message"):
            state["result_text"] = str(msg["last_agent_message"])
        return []
    if state.get("mode") == "delta":
        return []  # deltas already carried this turn's words
    state["mode"] = "whole"
    state["emitted"] = True
    return [text]


_PARSERS = {
    "claude": _parse_family,
    "codex": _parse_codex,
}

_LOGIN_HINT = "isn't signed in. Press Connect in settings and sign in"


def _failure(agent_id: str, stderr: str) -> RuntimeError:
    """A provider refusal, in words the person can act on."""
    preset = config.AGENT_PRESETS[agent_id]
    label = preset["label"]
    log.warning("%s failed: %s", label, stderr[:300].strip())
    low = stderr.lower()
    if "not supported when using" in low or "with a chatgpt account" in low:
        # The plan doesn't include the configured model.
        return RuntimeError(
            f"{label} can't run that model on your plan. Pick another one from "
            "the Model list in settings."
        )
    if "unrecognized_model" in low or "unrecognized model" in low or "unknown model" in low:
        return RuntimeError(
            f"{label} doesn't know that model name. Pick one from the Model "
            "list in settings, or choose your plan's default."
        )
    if any(s in low for s in ("not logged in", "log in", "/login", "unauthorized", "api key")):
        return RuntimeError(f"{label} {_LOGIN_HINT}.")
    if "enoent" in low or "not recognized" in low or "is not found" in low:
        return RuntimeError(
            f"{label} vanished mid-run. Check that it is still installed."
        )
    if "rate limit" in low or "usage limit" in low or "limit reached" in low:
        return RuntimeError(
            f"{label}'s plan limit is used up right now. Wait a bit and ask again."
        )
    if "credit" in low or "billing" in low:
        return RuntimeError(f"{label} says the account is out of credit.")
    detail = stderr.strip().splitlines()[-1][:200] if stderr.strip() else "no details"
    return RuntimeError(f"{label} failed: {detail}")


def _effort_rejected(detail: str) -> bool:
    """Whether a clean no-output failure came from the optional effort knob."""
    low = detail.lower()
    return any(
        marker in low
        for marker in (
            "--effort",
            "model_reasoning_effort",
            "reasoning effort",
            "reasoning_effort",
        )
    ) and any(
        marker in low
        for marker in (
            "unknown",
            "unsupported",
            "unrecognized",
            "unexpected",
            "invalid",
            "not supported",
        )
    )


async def _stream(agent_id: str, turn: Invocation) -> AsyncIterator[str]:
    """Run one headless turn, yielding reply text as it arrives."""
    label = config.AGENT_PRESETS[agent_id]["label"]
    state: dict = {}
    parser = _PARSERS[agent_id]
    started = time.perf_counter()

    proc = await asyncio.create_subprocess_exec(
        *turn.argv,
        cwd=turn.cwd,
        # Closed unless we have something to say
        stdin=asyncio.subprocess.PIPE if turn.payload else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # The default 64KB readline limit is real here
        limit=1 << 24,
        creationflags=CREATE_NO_WINDOW,
    )

    # Stdout and stderr both drain concurrently
    err_tail: list[bytes] = []

    async def drain_err() -> None:
        assert proc.stderr is not None
        async for line in proc.stderr:
            err_tail.append(line)
            del err_tail[:-12]

    async def feed() -> None:
        assert proc.stdin is not None and turn.payload is not None
        # A base64 screenshot is far larger than a pipe buffer
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
            proc.stdin.write(turn.payload)
            await proc.stdin.drain()
            proc.stdin.close()

    err_task = asyncio.create_task(drain_err())
    feed_task = asyncio.create_task(feed()) if turn.payload else None
    log.info(
        "agent turn via %s (image=%d bytes via %s, stdin=%d bytes)",
        agent_id,
        turn.image_bytes,
        turn.image_transport,
        len(turn.payload or b""),
    )

    try:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            for chunk in parser(raw.decode("utf-8", errors="replace"), state):
                yield chunk

        await proc.wait()

        stderr = b"".join(err_tail).decode("utf-8", errors="replace")
        failure = str(state.get("error") or (stderr if proc.returncode != 0 else ""))
        if failure:
            # Effort is an optimization, never a requirement.
            if (
                not state.get("emitted")
                and turn.fallback_argv is not None
                and turn.effort_key is not None
                and _effort_rejected(failure)
            ):
                _EFFORT_UNSUPPORTED.add(turn.effort_key)
                log.info(
                    "%s does not support the selected effort; retrying with its default",
                    label,
                )
                fallback = Invocation(
                    argv=turn.fallback_argv,
                    fallback_argv=None,
                    payload=turn.payload,
                    cwd=turn.cwd,
                    effort_key=None,
                    image_bytes=turn.image_bytes,
                    image_transport=turn.image_transport,
                    # The outer invocation owns this shared temporary folder.
                    temporary=None,
                )
                async for chunk in _stream(agent_id, fallback):
                    yield chunk
                return
            raise _failure(agent_id, failure)
        if not state.get("emitted"):
            # Nothing streamed: fall back to a whole reply the parser held back
            leftovers = state.get("finals") or []
            text = leftovers[-1] if leftovers else state.get("result_text", "")
            if not text:
                raise RuntimeError(f"{label} returned no speech.")
            yield text
    finally:
        if proc.returncode is None:
            proc.kill()
        for task in (err_task, feed_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        elapsed = time.perf_counter() - started
        usage = state.get("usage") or {}
        log.info(
            "agent turn via %s finished in %.2fs%s",
            agent_id,
            elapsed,
            f" usage={usage}" if usage else "",
        )
        turn.cleanup()


async def chat(
    messages: list[dict],
    cfg: dict | None = None,
    image: bytes | None = None,
) -> AsyncIterator[str]:
    """Same shape as llm.chat, so main._pass cannot tell the difference."""
    cfg = cfg or config.load()
    # The prompt lives beside the capabilities, not inside llm
    section = {**cfg["llm"], "system_prompt": llm.persona(cfg, "{model}")}
    turn = _turn(section["provider"], section, messages, image)
    async for chunk in _stream(section["provider"], turn):
        yield chunk


async def complete_vision(
    prompt: str, cfg: dict, image: bytes, schema: dict | None = None
) -> str:
    """Strict-output image call through the selected subscription CLI."""
    section = cfg["llm"]
    agent_id = section["provider"]
    system = (
        "You are a precise GUI locator. Follow the requested output grammar "
        "exactly and output no explanation."
    )
    turn = _prepare(agent_id, section, system, prompt, image, schema)
    chunks = []
    async for chunk in _stream(agent_id, turn):
        chunks.append(chunk)
        if len("".join(chunks)) > 240:
            break
    return "".join(chunks).strip()


async def complete_grounded(
    locator_prompt: str,
    cfg: dict,
    image: bytes,
    messages: list[dict],
    schema: dict,
) -> str:
    """One agent call that chooses the target and writes Mellow's answer."""
    section = {
        **cfg["llm"],
        "screen": "seen",
        "target": "",
        "items": "",
        "system_prompt": llm.persona(cfg, "{model}"),
    }
    system, user = build_prompt(messages, section, seen=True)
    user += (
        "\n\nYou must also ground this answer to the annotated screenshot. "
        "Return one JSON object matching the supplied schema. The selection "
        "field chooses the exact annotated target; the answer field is the "
        "short plain-language answer Mellow should say. Do not put JSON or "
        "selection identifiers inside the answer.\n\n" + locator_prompt
    )
    turn = _prepare(
        section["provider"], section, system, user, image=image, schema=schema
    )
    chunks: list[str] = []
    async for chunk in _stream(section["provider"], turn):
        chunks.append(chunk)
        if len("".join(chunks)) > 4096:
            break
    return "".join(chunks).strip()


async def test(cfg: dict) -> str:
    """Consume a tiny real turn — the same probe the HTTP adapters answer."""
    agent_id = cfg["llm"]["provider"]
    messages = [{"role": "user", "content": "Reply with only the word connected."}]
    turn = _turn(
        agent_id,
        _probe_section(
            agent_id,
            cfg["llm"].get("model", ""),
            cfg["llm"].get("agent_speed", "fast"),
        ),
        messages,
    )
    chunks: list[str] = []
    async for chunk in _stream(agent_id, turn):
        chunks.append(chunk)
        if len("".join(chunks)) >= 80:
            break
    answer = "".join(chunks).strip()
    if not answer:
        raise RuntimeError(
            f"{config.AGENT_PRESETS[agent_id]['label']} returned no speech."
        )
    return answer[:80]
