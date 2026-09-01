r"""Config lives in %APPDATA%\Mellow\config.json."""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "Mellow"
CONFIG_PATH = CONFIG_DIR / "config.json"

CAPABILITIES = ("llm", "stt", "tts")
MODES = ("local", "cloud")
# Only the answer model can be a coding agent
LLM_MODES = MODES + ("agent",)

# One preset table per capability, all the same shape
LLM_PRESETS = {
    "ollama": {"label": "Ollama", "base_url": "http://127.0.0.1:11434/v1", "requires_key": False, "local": True},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "requires_key": True, "local": False},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "requires_key": True, "local": False},
    "groq": {"label": "Groq", "base_url": "https://api.groq.com/openai/v1", "requires_key": True, "local": False},
    "nvidia": {"label": "NVIDIA NIM", "base_url": "https://integrate.api.nvidia.com/v1", "requires_key": True, "local": False},
    "custom": {"label": "Custom OpenAI-compatible", "base_url": "", "requires_key": False, "local": False},
    # Keep the existing second adapter available without turning providers into a plugin system.
    "anthropic": {"label": "Anthropic", "base_url": "https://api.anthropic.com/v1", "requires_key": True, "local": False},
}

# Both hosted behind /audio/transcriptions, so one adapter covers them.
STT_PRESETS = {
    "groq": {"label": "Groq", "base_url": "https://api.groq.com/openai/v1", "requires_key": True, "local": False},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "requires_key": True, "local": False},
    "custom": {"label": "Custom OpenAI-compatible", "base_url": "", "requires_key": False, "local": False},
}

# Both hosted behind /audio/speech.
TTS_PRESETS = {
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "requires_key": True, "local": False},
    # Groq's voice model is canopylabs/orpheus-v1-english (playai-tts was decommissioned)
    "groq": {"label": "Groq (Orpheus)", "base_url": "https://api.groq.com/openai/v1", "requires_key": True, "local": False},
    # The one provider here that is *not* OpenAI-compatible
    "elevenlabs": {"label": "ElevenLabs", "base_url": "https://api.elevenlabs.io/v1", "requires_key": True, "local": False, "model": "eleven_flash_v2_5"},
    # OpenAI-compatible after all - OpenRouter serves /audio/speech
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "requires_key": True, "local": False, "model": "fish-audio/s2.1-pro-free:free"},
    "custom": {"label": "Custom OpenAI-compatible", "base_url": "", "requires_key": False, "local": False},
}

PRESETS = {"llm": LLM_PRESETS, "stt": STT_PRESETS, "tts": TTS_PRESETS}

# Coding-agent CLIs as the answer model
AGENT_PRESETS = {
    "claude": {
        "label": "Claude Code",
        "binaries": ("claude",),
        "install": "npm install -g @anthropic-ai/claude-code",
        "vision": True,
        "models_cmd": (),
        "models": {
            "sonnet": "Sonnet, balanced",
            "opus": "Opus, most capable",
            "haiku": "Haiku, fastest",
            "fable": "Fable",
        },
    },
    "codex": {
        "label": "Codex",
        "binaries": ("codex",),
        "install": "npm install -g @openai/codex",
        "vision": True,
        "models_cmd": ("debug", "models"),
        "models": {},
    },
}

# Parakeet is the default
STT_MODELS = {
    "parakeet-tdt-0.6b-v2": "Fastest English (Parakeet)",
    "small": "Fast multilingual (Whisper)",
    "small.en": "Fast English (Whisper)",
    "medium.en": "Accurate English (Whisper)",
    "distil-large-v3": "Best English (Whisper)",
}

# A shortlist, not the whole set.
KOKORO_VOICES = {
    "af_heart": "Heart · female, US (best)",
    "af_bella": "Bella · female, US",
    "af_nicole": "Nicole · female, US",
    "af_aoede": "Aoede · female, US",
    "af_kore": "Kore · female, US",
    "am_michael": "Michael · male, US",
    "am_fenrir": "Fenrir · male, US",
    "am_puck": "Puck · male, US",
    "bf_emma": "Emma · female, UK",
    "bm_george": "George · male, UK",
}

# Passed straight through to the provider as `reasoning_effort`
REASONING_EFFORTS = ("", "none", "default", "low", "medium", "high")

# Subscription-backed coding agents expose a smaller
AGENT_SPEEDS = ("fast", "balanced", "deep")
AGENT_SPEED_EFFORT = {
    "fast": "low",
    "balanced": "medium",
    "deep": "high",
}

VISION_MODES = ("auto", "on", "off")

# Substrings that suggest a model can read images
VISION_HINTS = (
    "gpt-4o", "gpt-4.1", "chatgpt-4o", "o4",       # OpenAI vision line
    "gemini",                                        # all Gemini are multimodal
    "claude",                                        # Claude 3+ all take images
    "glm-4v", "glm-5v", "glmv",                      # GLM vision tiers
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qvq",     # Qwen vision
    "-vl-", "-vl@", "llava", "bakllava", "moondream",
    "pixtral", "internvl", "minicpm-v", "molmo",
    "llama3.2-vision", "llama4", "granite-vision", "phi-3.5-vision",
)


def resolves_vision(section: dict) -> bool:
    """The truth the rest of the code acts on: can this model see a screenshot?"""
    mode = section.get("vision", "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    # An agent brain has no model name to guess from (the box is usually empty
    if section.get("mode") == "agent":
        preset = AGENT_PRESETS.get(str(section.get("provider", "")), {})
        return bool(preset.get("vision"))
    name = str(section.get("model", "")).lower()
    return any(hint in name for hint in VISION_HINTS)

# Ollama out of the box: it speaks the OpenAI-compatible API and needs no key.
DEFAULTS = {
    "llm": {
        "mode": "local",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "",
        "model": "gemma3:4b",
        # A ceiling, not a target - the system prompt below is what keeps answers short.
        "max_tokens": 4096,
        "reasoning_effort": "",
        # Agent mode starts a fresh coding-agent CLI for every turn. Fast keeps the selected model but asks
        "agent_speed": "fast",
        # Sent on every request.
        "temperature": 0.3,
        # Can the model read a screenshot?
        "vision": "auto",
    },
    "stt": {
        "mode": "local",
        "local_model": "parakeet-tdt-0.6b-v2",
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": "",
        "model": "whisper-large-v3-turbo",
        # The microphone is ours either way - cloud STT still records here and uploads the take
        "input_device": None,
    },
    "tts": {
        "mode": "local",
        # af_heart is the highest-graded voice Kokoro ships (A).
        "local_voice": "af_heart",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "speech_speed": 1.0,
        "speak": True,  # mute the pet without tearing out the pipeline
    },
    # Master switch for the session event log ([[roadmap]] step 12).
    "remember_conversations": True,
    # The whole AI half, off at the master switch ([[roadmap]] step 8's "just the pet").
    "ai_enabled": True,
    # Empty on purpose. The rules that make Mellow sound like Mellow moved to llm.CORE
    "system_prompt": "",
}


def normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be a complete http:// or https:// URL")
    if parsed.username or parsed.password:
        raise ValueError("base URL must not contain a username or password")
    if parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a query string or fragment")
    # Each adapter appends its own endpoint
    for endpoint in ("/chat/completions", "/audio/speech", "/audio/transcriptions"):
        if parsed.path.rstrip("/").endswith(endpoint):
            return value[: value.rstrip("/").rfind(endpoint)]
    return value


def _number(section: dict, key: str, low: float, high: float, label: str) -> None:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}")


def _validate_transport(name: str, section: dict, require_model: bool = True) -> None:
    """provider / base_url / api_key / model — identical for all three."""
    presets = PRESETS[name]
    provider = str(section.get("provider", "")).strip().lower()
    if provider not in presets:
        raise ValueError(f"unknown {name} provider: {provider or '(empty)'}")
    section["provider"] = provider
    preset = presets[provider]
    # A local preset *is* the "on device" half of the toggle
    if preset.get("local"):
        section["mode"] = "local"
    section["base_url"] = normalize_base_url(
        str(section.get("base_url") or preset["base_url"])
    )

    model = str(section.get("model", "")).strip()
    if not model and not require_model:
        section["model"] = ""
    elif not model or len(model) > 200:
        raise ValueError(f"{name} model name is required, under 200 characters")
    else:
        section["model"] = model

    api_key = section.get("api_key", "")
    if not isinstance(api_key, str):
        raise ValueError(f"{name} API key must be text")
    section["api_key"] = api_key.strip()
    # Only enforced in cloud mode
    if section["mode"] == "cloud" and preset["requires_key"] and not section["api_key"]:
        raise ValueError(f"{preset['label']} requires an API key")


def _validate_agent(section: dict) -> None:
    """The agent half of the llm section: provider must be a known CLI."""
    provider = str(section.get("provider", "")).strip().lower()
    if provider not in AGENT_PRESETS:
        raise ValueError(f"unknown answer agent: {provider or '(empty)'}")
    section["provider"] = provider
    model = str(section.get("model", "")).strip()
    if len(model) > 200:
        raise ValueError("model name must be under 200 characters")
    section["model"] = model


def validate(candidate: dict) -> dict:
    """Merge defaults and reject unsafe or unusable settings."""
    if not isinstance(candidate, dict):
        raise ValueError("settings must be a JSON object")
    cfg = dict(DEFAULTS)
    cfg.update(candidate)
    cfg["system_prompt"] = str(cfg.get("system_prompt") or "")

    for name in CAPABILITIES:
        section = dict(DEFAULTS[name])
        given = cfg.get(name)
        if not isinstance(given, dict):
            raise ValueError(f"{name} settings must be a JSON object")
        section.update(given)
        cfg[name] = section

        mode = str(section.get("mode", "")).strip().lower()
        allowed = LLM_MODES if name == "llm" else MODES
        if mode not in allowed:
            raise ValueError(f"{name} mode must be one of {', '.join(allowed)}, not {mode!r}")
        section["mode"] = mode
        if name == "llm" and mode == "agent":
            _validate_agent(section)
        else:
            # A pet-only config must be savable without naming a model it will never call
            _validate_transport(name, section, require_model=cfg.get("ai_enabled", True))

    llm = cfg["llm"]
    _number(llm, "max_tokens", 1, 8192, "max tokens")
    if not isinstance(llm["max_tokens"], int):
        raise ValueError("max tokens must be a whole number")
    # 2 is the OpenAI ceiling. Anthropic's is 1
    _number(llm, "temperature", 0.0, 2.0, "temperature")
    effort = str(llm.get("reasoning_effort") or "").strip().lower()
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {effort}")
    llm["reasoning_effort"] = effort
    agent_speed = str(llm.get("agent_speed") or "fast").strip().lower()
    if agent_speed not in AGENT_SPEEDS:
        raise ValueError(f"agent speed must be one of {', '.join(AGENT_SPEEDS)}")
    llm["agent_speed"] = agent_speed
    vision = str(llm.get("vision") or "auto").strip().lower()
    if vision not in VISION_MODES:
        raise ValueError(f"vision must be one of {', '.join(VISION_MODES)}")
    llm["vision"] = vision

    stt = cfg["stt"]
    local_model = str(stt.get("local_model", "")).strip()
    if local_model not in STT_MODELS:
        raise ValueError(f"unsupported speech model: {local_model or '(empty)'}")
    stt["local_model"] = local_model
    device = stt.get("input_device")
    # An int is a config written before microphones were saved by name.
    if not isinstance(device, str):
        device = ""
    stt["input_device"] = device.strip() or None
    # Removed in favour of always-automatic selection - see stt.choose_channel.
    stt.pop("input_channel", None)

    tts = cfg["tts"]
    local_voice = str(tts.get("local_voice", "")).strip()
    if local_voice not in KOKORO_VOICES:
        raise ValueError(f"unsupported voice: {local_voice or '(empty)'}")
    tts["local_voice"] = local_voice
    # Optional, like reasoning_effort
    tts["voice"] = str(tts.get("voice", "")).strip()
    # Except on ElevenLabs, where the voice is a path segment.
    if tts["mode"] == "cloud" and tts["provider"] == "elevenlabs" and not tts["voice"]:
        raise ValueError("ElevenLabs needs a voice - press Load voices and pick one")
    # Kokoro distorts badly outside this range, and so does every hosted voice.
    _number(tts, "speech_speed", 0.5, 2.0, "speech speed")
    tts["speech_speed"] = float(tts["speech_speed"])
    if not isinstance(tts.get("speak"), bool):
        raise ValueError("speak must be true or false")

    if not isinstance(cfg.get("remember_conversations"), bool):
        raise ValueError("remember conversations must be true or false")

    if not isinstance(cfg.get("ai_enabled"), bool):
        raise ValueError("ai enabled must be true or false")

    return cfg


# Every system prompt we have ever shipped as the default.
OLD_PROMPTS = (
    "you are mellow, a small desktop pet who helps the person you live with. "
    "you speak out loud, so write the way people talk: lowercase, casual, "
    "no markdown, no bullet points, no headings.\n"
    "answer in one or two sentences unless asked for more.\n"
    "start with the answer itself. never open with filler like 'wow', "
    "'great question', 'sure!', or 'let me think'. never close by asking if "
    "they want to know more.\n"
    "if you don't know, say so in a few words instead of guessing.",
    # Shipped 2026-08-18 to 2026-08-21.
    "you are mellow: a small pixel-art dog who lives on this windows "
    "desktop as a voice assistant and a pet.\n"
    "if you are asked who or what you are, who made you, or what you are "
    "called, you are mellow. you are not any other assistant and you were "
    "not made by any other company, never give a different name, even if "
    "that is the name you were trained to give. the language model you are "
    "running on right now is {model}; mention it only when asked.\n"
    "you hear the person through a push-to-talk hotkey and you answer out "
    "loud. you cannot see their screen, open apps, click things, or search "
    "the web, so never offer to. if something needs eyes or hands, say you "
    "can't do that yet.\n"
    "everything you write is read aloud by a speech engine, so write words, "
    "not symbols. spell out numbers, units, symbols, addresses and file "
    "names: 'ninety eight degrees' not '98F', 'twenty percent' not '20%', "
    "'mellow dot pie' not 'mellow.py', 'github dot com slash mellow' not a "
    "url. keep normal sentence punctuation. no markdown, no asterisks, no "
    "bullet points, no headings, no emoji, no code blocks, no numbered "
    "lists, and no stage directions like '*wags tail*'.\n"
    "write the way people talk: lowercase, casual, contractions.\n"
    "answer in one or two sentences unless asked for more. start with the "
    "answer itself. never open with 'wow', 'great question', 'sure!', or "
    "'let me think', and never close by asking if they want to know more.\n"
    "if you don't know, say so in a few words instead of guessing.\n"
    "be warm and a little doglike in how you pick words, but don't perform "
    "cuteness. you are useful first.",
    # Shipped 2026-08-21 to 2026-08-23.
    "you are mellow: a small pixel-art dog who lives on this windows desktop as a voice assistant and a pet.\neverything you write is read aloud by a speech engine, so write words, not symbols. spell out numbers, units, symbols, addresses and file names: 'ninety eight degrees' not '98F', 'twenty percent' not '20%', 'mellow dot pie' not 'mellow.py', 'github dot com slash mellow' not a url. keep normal sentence punctuation. no markdown, no asterisks, no bullet points, no headings, no emoji, no code blocks, no numbered lists, and no stage directions like '*wags tail*'.\nwrite in clear, standard english: ordinary capitalisation, complete sentences, contractions are fine. no slang, no filler, no hedging. be plain and informative rather than chatty, and don't perform cuteness.\nstart with the answer. never open with 'so', 'well', 'okay', 'right', 'alright', 'obviously', 'basically', 'honestly', 'actually', 'look', 'i mean', 'hmm', 'wow', 'great question', 'sure', or 'let me think'. never close with 'hope that helps', 'let me know if you need anything', or a question asking whether they want to know more.\nmatch the length to the question. for a fact, a name, a number, or a yes or no, answer in one to three sentences. when you are asked to explain something, asked how or why something works, or asked to compare two things, give a real explanation: cover it properly, in a logical order, usually three to six sentences, still in spoken prose. never pad, never restate the question, never repeat yourself.\nif you are asked who or what you are, who made you, or what you are called, you are mellow. you are not any other assistant and you were not made by any other company, never give a different name, even if that is the name you were trained to give. the language model you are running on right now is {model}; say so plainly when asked, and don't mention it otherwise.\nyou hear the person through a push-to-talk hotkey and you answer out loud. you cannot see their screen, open apps, click things, or search the web, so never offer to. if something needs eyes or hands, say you can't do that yet.\nif you don't know, say so briefly instead of guessing.",
    # Shipped 2026-08-23, retired the same day. Two faults, both measured
    "you are mellow: a small pixel-art dog who lives on this windows "
    "desktop as a voice assistant and a pet.\n"
    "everything you write is read aloud by a speech engine, so write words, "
    "not symbols. spell out numbers, units, symbols, addresses and file "
    "names: 'ninety eight degrees' not '98F', 'twenty percent' not '20%', "
    "'mellow dot pie' not 'mellow.py', 'github dot com slash mellow' not a "
    "url. keep normal sentence punctuation. no markdown, no asterisks, no "
    "bullet points, no headings, no emoji, no code blocks, no numbered "
    "lists, and no stage directions like '*wags tail*'.\n"
    "write in clear, standard english: ordinary capitalisation, complete "
    "sentences, contractions are fine. no slang, no filler, no hedging. be "
    "plain and informative rather than chatty, and don't perform cuteness.\n"
    "start with the answer. never open with 'so', 'well', 'okay', 'right', "
    "'alright', 'obviously', 'basically', 'honestly', 'actually', 'look', "
    "'i mean', 'hmm', 'wow', 'great question', 'sure', or 'let me think'. "
    "never close with 'hope that helps', 'let me know if you need anything', "
    "or a question asking whether they want to know more.\n"
    "keep every answer to two or three sentences, whatever the question. "
    "this is a spoken conversation, not a written report, and a paragraph "
    "read out loud is tiring however good it is. short is not thin: lead "
    "with the answer, say the part that actually matters rather than "
    "everything you know, and stop. never pad, never restate the question, "
    "never repeat yourself.\n"
    "if you are asked who or what you are, who made you, or what you are "
    "called, you are mellow. you are not any other assistant and you were "
    "not made by any other company, never give a different name, even if "
    "that is the name you were trained to give. the language model you are "
    "running on right now is {model}; say so plainly when asked, and don't "
    "mention it otherwise.\n"
    "you hear the person through a push-to-talk hotkey and you answer out "
    "loud. you cannot open apps, click things, type for them, or search "
    "the web, so never offer to.\n"
    "if you don't know, say so briefly instead of guessing.",
)

# Values that were only ever *our* old default, never a choice the user made.
SUPERSEDED = {
    "llm.max_tokens": {300: 2048, 2048: 4096},
}


def _retired() -> tuple[str, ...]:
    """Prompts nobody chose: our own defaults, past and present."""
    from mellowd import llm

    return OLD_PROMPTS + (llm.CORE,)


def _supersede(cfg: dict) -> None:
    # Wording that was only ever ours is cleared rather than rewritten
    try:
        if str(cfg.get("system_prompt") or "").strip() in _retired():
            cfg["system_prompt"] = ""
    except Exception:
        pass
    for path, moves in SUPERSEDED.items():
        *parents, key = path.split(".")
        node = cfg
        for parent in parents:
            node = node.get(parent)
            if not isinstance(node, dict):
                break
        else:
            # Bounded loop, not a single step
            for _ in range(len(moves)):
                old = node.get(key)
                # Only scalars are candidates; `in` on an unhashable value would raise
                if not isinstance(old, (int, float, str)) or old not in moves:
                    break
                node[key] = moves[old]

# Where each key of the old flat schema now lives.
MOVED = {
    "provider": ("llm", "provider"),
    "base_url": ("llm", "base_url"),
    "api_key": ("llm", "api_key"),
    "model": ("llm", "model"),
    "max_tokens": ("llm", "max_tokens"),
    "reasoning_effort": ("llm", "reasoning_effort"),
    "stt_model": ("stt", "local_model"),
    "input_device": ("stt", "input_device"),
    "input_channel": ("stt", "input_channel"),
    "voice": ("tts", "local_voice"),
    "speech_speed": ("tts", "speech_speed"),
    "speak": ("tts", "speak"),
}


def _retire_agents(cfg: dict) -> None:
    """Move anyone parked on a coding-agent CLI that no longer ships."""
    llm = cfg.get("llm")
    if not isinstance(llm, dict) or llm.get("mode") != "agent":
        return
    if str(llm.get("provider", "")).strip().lower() in AGENT_PRESETS:
        return
    llm["provider"] = "claude"
    llm["model"] = ""


def migrate(cfg: dict) -> dict:
    """Lift a flat pre-step-9 config into the three-section shape."""
    _retire_agents(cfg)
    if "provider" not in cfg:
        _supersede(cfg)
        return cfg

    out = {k: v for k, v in cfg.items() if k not in MOVED}
    for name in CAPABILITIES:
        out[name] = {**DEFAULTS[name], **cfg.get(name, {})}
    for old, (name, new) in MOVED.items():
        if old in cfg:
            out[name][new] = cfg[old]
    # The old schema had no modes. A provider that needs a key was cloud
    provider = out["llm"]["provider"]
    out["llm"]["mode"] = "local" if LLM_PRESETS.get(provider, {}).get("local") else "cloud"
    out["stt"]["mode"] = "local"
    out["tts"]["mode"] = "local"
    # After the lift, so the dotted paths match — see _supersede.
    _supersede(out)
    return out


def load() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return validate(migrate(cfg))


def save(cfg: dict) -> None:
    # ponytail: api keys sit in plaintext here
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    checked = validate(cfg)
    temp = CONFIG_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(checked, indent=2), encoding="utf-8")
    os.replace(temp, CONFIG_PATH)


def redacted(cfg: dict) -> dict:
    """Safe to log or send to the shell. Blanks every key, not just the first."""
    out = dict(cfg)
    for name in CAPABILITIES:
        section = dict(out.get(name, {}))
        section["has_api_key"] = bool(section.get("api_key"))
        section["api_key"] = ""
        out[name] = section
    return out
