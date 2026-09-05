"""Two adapters, not a plugin system."""

import json
import logging
import re
from typing import AsyncIterator

import httpx

from mellowd import config, errors

log = logging.getLogger("mellowd.llm")

TIMEOUT = httpx.Timeout(120.0, connect=10.0)

# A prompt is a request, not a guarantee
_ALWAYS = (
    "great question", "let me think", "i mean", "obviously", "basically",
    "honestly", "hmm", "wow", "sure", "um", "uh", "ah",
)
_NEEDS_COMMA = ("alright", "actually", "okay", "right", "well", "look", "so", "ok")


def _alternation(words: tuple[str, ...]) -> str:
    # Longest first, or "ok" shadows "okay" and leaves a stray "ay".
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# Either a bare filler word followed by punctuation or a space
_OPENER = re.compile(
    rf"(?:(?:{_alternation(_ALWAYS)})(?:\s*[,:;.!…-]+\s*|\s+)"
    rf"|(?:{_alternation(_NEEDS_COMMA)})\s*[,:;…-]+\s*)",
    re.IGNORECASE,
)

# Every opener word, for "might this still become filler?" while streaming.
_OPENER_WORDS = _ALWAYS + _NEEDS_COMMA

# How much of the answer may be held back while deciding.
_OPENER_HOLD = 24

# Reasoning models that dump chain-of-thought into `content` wrap it in one of these.
_THINK_TAGS = (
    ("<think>", "</think>"),
    ("<thought>", "</thought>"),
)


def _tag_hold(buf: str, tag: str) -> int:
    """How many trailing chars of buf could still grow into tag."""
    low, tag = buf.lower(), tag.lower()
    for n in range(min(len(tag) - 1, len(low)), 0, -1):
        if tag.startswith(low[-n:]):
            return n
    return 0


def _may_still_open(head: str) -> bool:
    """Could this partial text still turn out to be a filler opener?"""
    low = head.lower()
    for word in _OPENER_WORDS:
        if word.startswith(low):
            return True
        if low.startswith(word) and not low[len(word) :].strip(" ,:;.!…-"):
            return True
    return False


class _Stream:
    """What the stream did, so an empty answer can say *why* it was empty."""

    def __init__(self) -> None:
        self.chunks = 0
        self.finish: str | None = None
        self.reasoning = False
        # What the provider says it served
        self.model: str | None = None
        # Both streams so far, for is_thinking() below. Only ever grown
        self.thoughts = ""
        self.answer = ""
        self.echoing: bool | None = None
        # The opening of the answer, held only while it might be filler.
        self.opening = ""
        self.opened = False
        # State for strip_think(): a tag can split across chunks
        self._in_think = False
        self._think_close = ""
        self._think_buf = ""

    def strip_think(self, text: str) -> str:
        """Drop think/thought blocks that arrived inside content."""
        self._think_buf += text
        out: list[str] = []
        while self._think_buf:
            if self._in_think:
                end = self._think_buf.lower().find(self._think_close)
                if end < 0:
                    hold = _tag_hold(self._think_buf, self._think_close)
                    self._think_buf = self._think_buf[-hold:] if hold else ""
                    break
                self._think_buf = self._think_buf[end + len(self._think_close) :]
                self._in_think = False
                self._think_close = ""
                continue
            # Earliest open tag wins when a chunk somehow contains both.
            hit: tuple[int, str, str] | None = None
            low = self._think_buf.lower()
            for open_tag, close_tag in _THINK_TAGS:
                at = low.find(open_tag)
                if at >= 0 and (hit is None or at < hit[0]):
                    hit = (at, open_tag, close_tag)
            if hit is None:
                hold = max(_tag_hold(self._think_buf, open_t) for open_t, _ in _THINK_TAGS)
                if hold:
                    out.append(self._think_buf[:-hold])
                    self._think_buf = self._think_buf[-hold:]
                else:
                    out.append(self._think_buf)
                    self._think_buf = ""
                break
            at, open_tag, close_tag = hit
            out.append(self._think_buf[:at])
            self._think_buf = self._think_buf[at + len(open_tag) :]
            self._in_think = True
            self._think_close = close_tag
            self.reasoning = True
        return "".join(out)

    def flush_think(self) -> str:
        """Whatever is still outside a think block when the stream ends."""
        if self._in_think:
            # Truncated mid-thought: nothing after the open tag is an answer.
            self._think_buf = ""
            self._think_close = ""
            return ""
        leftover, self._think_buf = self._think_buf, ""
        return leftover

    def is_thinking(self, text: str) -> bool:
        """Is this `content` chunk actually the chain of thought again?"""
        self.answer += text
        if not self.thoughts:
            return False  # nothing to be a copy of
        if self.echoing is None:
            self.echoing = self.thoughts.startswith(self.answer)
        return self.echoing

    def emit(self, text: str) -> str:
        """One content chunk, with any filler opener taken off the front."""
        if self.opened:
            return text
        self.opening += text
        head = self.opening.lstrip()
        if len(head) < _OPENER_HOLD:
            # Decide on what is *left* after stripping, not on what arrived
            cleaned = self._without_opener(head)
            if not cleaned or _may_still_open(cleaned):
                return ""  # "so" could still become "so," or "so far"
        self.opened = True
        self.opening = ""
        return self._without_opener(head)

    def flush(self) -> str:
        """Whatever is still held when the stream ends."""
        if self.opened or not self.opening.strip():
            return ""
        self.opened = True
        head, self.opening = self.opening.lstrip(), ""
        # An answer that is *only* an opener is left alone
        return self._without_opener(head) or head

    @staticmethod
    def _without_opener(head: str) -> str:
        cut = 0
        for _ in range(2):  # twice, because "so, well, the reason is" happens
            match = _OPENER.match(head, cut)
            if not match:
                break
            cut = match.end()
        if not cut:
            # Nothing was filler, so leave the text exactly as written
            return head
        rest = head[cut:]
        return rest[:1].upper() + rest[1:]

    def done(self, cfg: dict) -> None:
        log.info(
            "stream done: %d chunks, finish=%s, reasoning=%s, served=%s",
            self.chunks,
            self.finish,
            self.reasoning,
            self.model or "(not reported)",
        )
        if self.chunks:
            return
        # "length" is OpenAI's word for truncation, "max_tokens" is Anthropic's.
        if self.reasoning or self.finish in ("length", "max_tokens"):
            raise RuntimeError(
                f"model spent all {cfg['max_tokens']} tokens reasoning before it "
                f"answered (finish_reason={self.finish}). Raise max tokens, or "
                f"set reasoning effort to low."
            )
        raise RuntimeError(
            f"provider returned an empty response (finish_reason={self.finish})"
        )


# Mellow answering as itself
ANCHOR = (
    (
        "who are you?",
        "I'm Mellow. I live on your desktop, and I answer out loud when you "
        "hold the hotkey.",
    ),
    # Deliberately *not* the phrasing a person uses ("what model are you running on?"). Word
    ("which language model is this?", "I'm running on {model}."),
    # Replaced the Tokyo clock exchange
    (
        "can you send a quick note to my landlord about the boiler?",
        "Here's the note: Hi, the boiler has stopped producing hot water and I'd "
        "like to arrange a repair this week. Copy that across and it's ready to "
        "go — I can't send it myself.",
    ),
    (
        "why does my laptop get slow when lots of tabs are open?",
        # Three sentences, and it still explains the whole mechanism.
        "Each tab keeps its own copy of the page in memory, so a few dozen can "
        "use more than you have. Windows then starts moving memory onto the "
        "disk, which is far slower, and everything stalls waiting on it. "
        "Closing the tabs you aren't reading usually fixes it within seconds.",
    ),
)

# Ten turns of history put the system prompt a long way from the question
_REMINDER_TONE = (
    # "Never something to describe" earns its place
    "Reminder: you are mellow. This note is for you and is never something to "
    "quote, explain or describe. Everything you say is read out loud and shown "
    "on screen, so no markdown or symbols. Answer in plain sentences, start "
    "with the answer, and no filler opener."
)
REMINDER = _REMINDER_TONE + (
    " Don't announce what you are about to do or repeat the question back."
)
# Exact marker used to request a screenshot.
LOOK = "[look]"
REMINDER_LOOK = _REMINDER_TONE + (
    " You may be shown the user's screen. If answering their question needs"
    " eyes on it, begin your reply with exactly [look] and write nothing"
    " else; you will then be shown a screenshot and expected to answer."
    " Never write [look] for any other reason."
)
# The honest refusal, in the model's own words — never a canned sentence
REMINDER_NOLOOK = _REMINDER_TONE + (
    " You cannot see the user's screen: the model you are running on takes no"
    " images. If asked what is on it, or to read something there, say plainly"
    " that you can't see the screen with the current model."
)
# The screengrab itself failed — locked workstation, a missing DLL
REMINDER_NOSHOT = _REMINDER_TONE + (
    " You tried to look at the user's screen and the capture failed, so there"
    " is no picture this turn. Say briefly that you couldn't get a look at the"
    " screen just now and they can ask again."
)
# Second pass of a screen turn: the screenshot rides on the latest user message.
REMINDER_SEEN = _REMINDER_TONE + (
    # "the user's screen", never "a screenshot" or "an image"
    " The user's screen is attached to their latest message. Use it to answer"
    " the question they actually asked, in two or three sentences — the part"
    " that answers them, and nothing else that happens to be on screen. If the"
    " screen doesn't show what they're asking about, just answer the question"
    " directly and say the screen doesn't show it."
)


# The marker, demonstrated.
ANCHOR_LOOK = (
    ("what does this error on my screen say?", LOOK),
    # NOTE: keep this list and ANCHOR_SEEN in step
    ("can you see my screen?", LOOK),
)

# The other half: what a screen answer looks like once the picture has arrived.
ANCHOR_SEEN = (
    (
        "what is this error telling me?",
        "The build is failing because port three thousand is already in use, so "
        "the dev server never starts. Something else is holding it — usually an "
        "older copy of the same server that didn't shut down.",
    ),
)


# Who Mellow is Always sent, and not editable from Settings.
CORE = (
    "you are mellow: a small pixel-art dog who lives on this windows "
    "desktop as a voice assistant and a pet.\n"
    # Speech engines already normalize numbers and URLs.
    "everything you write is both read aloud and shown in a speech bubble, "
    "so write plain prose: normal words, normal numbers, normal sentence "
    "punctuation. no markdown, no asterisks, no bullet points, no headings, "
    "no emoji, no code blocks, no numbered lists, and no stage directions "
    "like '*wags tail*'.\n"
    "write in clear, standard english: ordinary capitalisation, complete "
    "sentences, contractions are fine. no slang, no filler, no hedging. be "
    "plain and informative rather than chatty, and don't perform cuteness.\n"
    "start with the answer. never open with 'so', 'well', 'okay', 'right', "
    "'alright', 'obviously', 'basically', 'honestly', 'actually', 'look', "
    "'i mean', 'hmm', 'wow', 'great question', 'sure', or 'let me think'. "
    "never close with 'hope that helps', 'let me know if you need anything', "
    "or a question asking whether they want to know more.\n"
    # "whatever the question" used to end that first sentence.
    "keep every answer to two or three sentences, unless they ask you to go "
    "deeper. this is a spoken conversation, not a written report, and a "
    "paragraph read out loud is tiring however good it is. short is not "
    "thin: lead with the answer, say the part that actually matters rather "
    "than everything you know, and stop. never pad, never restate the "
    "question, never repeat yourself.\n"
    "if you are asked who or what you are, who made you, or what you are "
    "called, you are mellow. you are not any other assistant and you were "
    "not made by any other company, never give a different name, even if "
    "that is the name you were trained to give. the language model you are "
    "running on right now is {model}; say so plainly when asked, and don't "
    "mention it otherwise.\n"
    # Says nothing about the screen, deliberately.
    "you hear the person through a push-to-talk hotkey and you answer out "
    "loud. you write things out for them: asked to reply to something, "
    "draft the reply and say it; asked how to do something, give the steps. "
    # The honest counterpart to the line this replaces.
    "you can open apps, folders and websites on their computer, play "
    "something, and set one app's volume. asked to, do it and say what you "
    "did in one line. for anything else on their machine, say what to do and "
    "let them do it.\n"
    "if you don't know, say so briefly instead of guessing."
    )


def persona(cfg: dict, model: str | None = None) -> str:
    """CORE, plus whatever they added in Settings."""
    extra = str(cfg.get("system_prompt") or "").strip()
    whole = CORE + ("\n" + extra if extra else "")
    return whole.replace("{model}", cfg["llm"]["model"] if model is None else model)


# Step 14c: pointing The second marker
POINT = "[POINT:"

# What the model is asked to do, and the whole of step 14c is in the swap.
_PICK_RULE = (
    " Start your reply with [POINT:n], where n is the number of the one thing"
    " on that list they should go to first, and then say in two or three"
    " sentences what it is and what to do with it. Pick the thing itself,"
    " never a browser tab, a page heading or a breadcrumb that only happens to"
    " say the same words. One thing, never a list of them."
    # The list is measured off the screen; the picture is shrunk context.
    " Everything on that list was read off their screen a moment ago, so a row"
    " being on it means it is there: trust it over what you can make out in the"
    " picture, which is shrunk, and never say a thing is not on screen when it"
    " is on the list. If nothing there is"
    " worth pointing at, start with [POINT:none] and simply answer the"
    " question. Never mention the marker, the number or the list."
)
# Phase 2 of a pointing turn. REMINDER_SEEN's "two or three sentences" holds
REMINDER_PICK = REMINDER_SEEN + _PICK_RULE
REMINDER_TARGET = REMINDER_SEEN + (
    " The pointing engine has already located the exact control and Mellow is"
    " pointing at it now. It is named below. Trust that measured result over"
    " the reduced screenshot, explain what the control is and what to do with"
    " it in two or three sentences, and never mention the engine, measurement"
    " or target note.\n\nDETECTED TARGET: "
)
# A follow-up step: they did the thing, the screen moved
_GUIDE_INTRO = (
    _REMINDER_TONE
    + " You are walking the user through something one step at a time, and"
    " they have just done the step you pointed at. Their screen now is"
    " attached to their latest message. Say in one short sentence what to do"
    " next, or that they have finished."
)
REMINDER_GUIDE = _GUIDE_INTRO
REMINDER_GUIDE_PICK = _GUIDE_INTRO + _PICK_RULE

# Pointing, demonstrated - and both outcomes
ANCHOR_POINT = (
    (
        'where do i start a new chat?\n1 *"New" 4,22\n'
        '2  "New chat - Claude" 40,2\n3  "Projects" 5,26',
        "[POINT:1] That is the New button at the top of the sidebar, and it "
        "opens an empty conversation. The tab along the top of the window only "
        "says the same thing because it is the page you are already on.",
    ),
    (
        'how do i make this photo less blurry?\n1 *"Export" 90,4\n'
        '2  "Adjustments" 88,20',
        "[POINT:none] Nothing on screen will fix that. The blur is in the "
        "original shot, so sharpening it here would only make the edges noisy.",
    ),
)


# Step 15a: doing things The third marker
DO = "[DO:"

# Same swap as pointing, and the same reason it is safe.
_ACT_RULE = (
    " Start your reply with [DO:n] for the one thing on that list to do, or"
    " [DO:n|what to do] when the row asks for it, and then talk to them about"
    " it however suits the moment - you are doing them a favour, not filing a"
    " report. If none of them is what they meant - if they are asking about"
    " something on their screen rather than asking you to open something -"
    " start with [DO:none] instead and the question is handled the usual way."
    " When they say play that, it, this, that song, or that video, resolve the"
    " actual title from what they named earlier in this request or conversation"
    " and put that title after the |. Never pass a pronoun such as that or it"
    " as the search text."
    " Never mention the marker, the number or the list."
)
REMINDER_ACT = _REMINDER_TONE + (
    " They have asked you to do something on their computer, and everything"
    " you can do about it is listed below."
) + _ACT_RULE

# Both outcomes, because a model shown only the acting case will act on every turn rather than admit
ANCHOR_ACT = (
    (
        "put on back in black\n1 Play a song or video on YouTube (plays it"
        " straight away) - say what to do, the song or video to play\n"
        "2 Spotify (an installed app)",
        "[DO:1|back in black] Rock on. That one always sounds better loud.",
    ),
    (
        "can you open my downloads\n1 Downloads (a folder)\n2 Documents (a"
        " folder)",
        "[DO:1] Here you go.",
    ),
    (
        "do you know about a song called Back in Black? can you play that on"
        " youtube?\n1 Play a song or video on YouTube (plays it straight away)"
        " - say what to do, the song or video to play\n2 YouTube (a website)",
        "[DO:1|Back in Black] Absolutely. Putting it on now.",
    ),
    (
        "turn chrome down a bit, it's too loud\n1 Set an app's volume - say"
        " what to do, the app name and a percentage, like: spotify 50\n"
        "2 Google Chrome (an installed app)",
        "[DO:1|chrome 40] Got it, forty percent. Say the word if that is still"
        " too much.",
    ),
    (
        "where do i click to open the file menu\n1 File Explorer (an installed"
        " app)\n2 Files (a folder)",
        "[DO:none] That one is on your screen rather than on your computer, so"
        " let me have a look.",
    ),
)


def _reminder_for(cfg: dict) -> str:
    """Which screen rule this request carries."""
    # Appended rather than formatted in: these are OCR'd labels off a stranger's screen
    items = cfg.get("items") or ""
    # Acting outranks looking: an action turn never took a screenshot
    if cfg.get("doing"):
        return REMINDER_ACT + cfg["doing"]
    if cfg.get("target"):
        return REMINDER_TARGET + str(cfg["target"])
    if cfg.get("screen") == "guide":
        return REMINDER_GUIDE_PICK + items if items else REMINDER_GUIDE
    if cfg.get("screen") == "seen":
        # No list means nothing on screen could be read
        return REMINDER_PICK + items if items else REMINDER_SEEN
    if cfg.get("screen") == "failed":
        return REMINDER_NOSHOT
    if cfg.get("vision_ok", True):
        return REMINDER_LOOK
    return REMINDER_NOLOOK


def _anchored(cfg: dict, messages: list[dict]) -> list[dict]:
    if not cfg.get("anchor", True):
        return list(messages)
    # Exactly one of the two screen halves
    exchanges = ANCHOR
    screen = cfg.get("screen")
    if cfg.get("doing"):
        exchanges += ANCHOR_ACT
    elif cfg.get("items") and not cfg.get("target") and screen in ("seen", "guide"):
        exchanges += ANCHOR_POINT
    elif screen in ("seen", "guide"):
        exchanges += ANCHOR_SEEN
    elif cfg.get("vision_ok", True):
        exchanges += ANCHOR_LOOK
    out = []
    for question, answer in exchanges:
        out.append({"role": "user", "content": question})
        out.append({"role": "assistant", "content": answer.format(model=cfg["model"])})
    return out + list(messages)


def _with_image_openai(messages: list[dict], image_b64: str) -> list[dict]:
    """Attach a screenshot to the latest user turn, OpenAI content-parts style."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": str(out[i].get("content", ""))},
                ],
            }
            break
    return out


def _google_openai(cfg: dict) -> bool:
    """True when the OpenAI-compat base URL is Google AI Studio / Gemini API."""
    return "generativelanguage.googleapis.com" in (cfg.get("base_url") or "").lower()


# Gemma 4 on Google's OpenAI endpoint. reasoning_effort maps to a thinking *budget*
_GEMMA_THINK_LEVEL = {
    "": "MINIMAL",
    "none": "MINIMAL",
    "default": "LOW",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}


async def _openai(
    cfg: dict, messages: list[dict], image_b64: str | None = None
) -> AsyncIterator[str]:
    if image_b64:
        messages = _with_image_openai(messages, image_b64)
    raw = bool(cfg.get("raw"))
    payload = {
        "model": cfg["model"],
        # The reminder goes last
        "messages": (
            [{"role": "system", "content": cfg["system_prompt"]}, *messages]
            if raw
            else [
                {"role": "system", "content": cfg["system_prompt"]},
                *_anchored(cfg, messages),
                {"role": "system", "content": _reminder_for(cfg)},
            ]
        ),
        "max_tokens": cfg["max_tokens"],
        # Never sent before, so every provider used its own default
        "temperature": cfg["temperature"],
        "stream": True,
    }
    if cfg["provider"] == "ollama":
        payload["keep_alive"] = "30m"
    effort = cfg.get("reasoning_effort")
    if cfg["provider"] == "openrouter":
        # Don't send the thinking at all. The model still reasons
        payload["reasoning"] = {"exclude": True} | ({"effort": effort} if effort else {})
    elif cfg["provider"] == "groq":
        # Qwen 3.6 defaults to reasoning_format=raw
        if "gpt-oss" in cfg["model"].lower():
            payload["include_reasoning"] = False
        else:
            payload["reasoning_format"] = "hidden"
        if effort:
            payload["reasoning_effort"] = effort
    elif _google_openai(cfg) and "gemma" in cfg["model"].lower():
        # Always set a level: leaving it out makes Gemma emit <thought>… and bill those tokens.
        payload["extra_body"] = {
            "google": {
                "thinking_config": {
                    "thinking_level": _GEMMA_THINK_LEVEL.get(effort or "", "MINIMAL"),
                    "include_thoughts": False,
                }
            }
        }
    elif effort:
        # Only sent when the user set it. An unknown field is a 400 on some providers
        payload["reasoning_effort"] = effort
    headers = (
        {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
    )

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{cfg['base_url'].rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        ) as r:
            await _raise_with_body(r, cfg)
            seen = _Stream()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                if not body:
                    continue
                event = json.loads(body)
                if error := event.get("error"):
                    detail = (
                        error.get("message", str(error))
                        if isinstance(error, dict)
                        else str(error)
                    )
                    raise RuntimeError(f"provider stream failed: {detail[:300]}")
                seen.model = event.get("model") or seen.model
                choices = event.get("choices") or []
                if not choices:
                    continue
                seen.finish = choices[0].get("finish_reason") or seen.finish
                delta = choices[0].get("delta", {})
                # Some models stream reasoning separately before content.
                if thought := (
                    delta.get("reasoning") or delta.get("reasoning_content") or ""
                ):
                    seen.reasoning = True
                    seen.thoughts += thought
                if text := delta.get("content"):
                    text = seen.strip_think(text)
                    if not text or seen.is_thinking(text):
                        continue
                    if opening := seen.emit(text):
                        seen.chunks += 1
                        yield opening
            if leftover := seen.flush_think():
                if not seen.is_thinking(leftover):
                    if opening := seen.emit(leftover):
                        seen.chunks += 1
                        yield opening
            if rest := seen.flush():
                seen.chunks += 1
                yield rest
            seen.done(cfg)


def _with_image_anthropic(messages: list[dict], image_b64: str) -> list[dict]:
    """Attach a screenshot to the latest user turn, Anthropic block style."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": str(out[i].get("content", ""))},
                ],
            }
            break
    return out


async def _anthropic(
    cfg: dict, messages: list[dict], image_b64: str | None = None
) -> AsyncIterator[str]:
    if image_b64:
        messages = _with_image_anthropic(messages, image_b64)
    raw = bool(cfg.get("raw"))
    payload = {
        "model": cfg["model"],
        # Anthropic keeps `system` out of `messages` entirely
        "system": (
            cfg["system_prompt"]
            if raw
            else f"{cfg['system_prompt']}\n{_reminder_for(cfg)}"
        ),
        "messages": messages if raw else _anchored(cfg, messages),
        "max_tokens": cfg["max_tokens"],
        # Anthropic's ceiling is 1.0 where OpenAI's is 2.0.
        "temperature": min(cfg["temperature"], 1.0),
        "stream": True,
    }
    headers = {
        "x-api-key": cfg["api_key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    base = cfg.get("base_url") or "https://api.anthropic.com/v1"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream(
            "POST", f"{base.rstrip('/')}/messages", json=payload, headers=headers
        ) as r:
            await _raise_with_body(r, cfg)
            seen = _Stream()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("type") == "message_start":
                    seen.model = event.get("message", {}).get("model") or seen.model
                if event.get("type") == "message_delta":
                    seen.finish = (
                        event.get("delta", {}).get("stop_reason") or seen.finish
                    )
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "thinking_delta":
                        seen.reasoning = True
                        seen.thoughts += delta.get("thinking") or ""
                    if text := delta.get("text"):
                        # Anthropic keeps the two apart properly, so this never fires here.
                        text = seen.strip_think(text)
                        if not text or seen.is_thinking(text):
                            continue
                        if opening := seen.emit(text):
                            seen.chunks += 1
                            yield opening
            if leftover := seen.flush_think():
                if not seen.is_thinking(leftover):
                    if opening := seen.emit(leftover):
                        seen.chunks += 1
                        yield opening
            if rest := seen.flush():
                seen.chunks += 1
                yield rest
            seen.done(cfg)


async def _raise_with_body(r: httpx.Response, cfg: dict) -> None:
    """httpx hides the response body on streamed errors, which is exactly the part that says *why* the request failed."""
    if r.is_success:
        return
    await r.aread()
    raise errors.provider_error(r.status_code, r.text, cfg["provider"], cfg["model"])


# What Ollama told us about a local model, keyed by (base_url
_VISION_CACHE: dict[tuple[str, str], bool] = {}

# Short on purpose: this is localhost
_PROBE_TIMEOUT = 2.0


def _ollama_root(base_url: str) -> str:
    """The native API root behind Ollama's OpenAI-compatible base URL."""
    base = base_url.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def probe_vision(section: dict) -> None:
    """Ask Ollama whether this model takes images."""
    if section.get("provider") != "ollama" or section.get("vision", "auto") != "auto":
        return
    key = (str(section.get("base_url", "")), str(section.get("model", "")))
    if not all(key) or key in _VISION_CACHE:
        return
    try:
        response = httpx.post(
            f"{_ollama_root(key[0])}/api/show",
            json={"model": key[1]},
            timeout=_PROBE_TIMEOUT,
        )
        response.raise_for_status()
        capabilities = response.json().get("capabilities") or []
        _VISION_CACHE[key] = "vision" in capabilities
        log.info(
            "ollama says %s %s see images", key[1],
            "can" if _VISION_CACHE[key] else "cannot",
        )
    except Exception as e:
        # Not cached: Ollama may simply not be up yet.
        log.info("could not ask ollama about %s (%s)", key[1], e)


# Which models have already been reported as not fitting on the GPU.
_SPILL_SEEN: set[str] = set()

# A gigabyte, as ollama counts them.
_GIB = 1024 ** 3


def check_fit(section: dict) -> None:
    """Say so in the log when the local model does not fit on the GPU."""
    if section.get("provider") != "ollama":
        return
    model = str(section.get("model", ""))
    base = str(section.get("base_url", ""))
    if not model or not base or model in _SPILL_SEEN:
        return
    try:
        response = httpx.get(f"{_ollama_root(base)}/api/ps", timeout=_PROBE_TIMEOUT)
        response.raise_for_status()
        for loaded in response.json().get("models") or []:
            if loaded.get("name") != model and loaded.get("model") != model:
                continue
            total = loaded.get("size") or 0
            on_gpu = loaded.get("size_vram") or 0
            if not total:
                return
            _SPILL_SEEN.add(model)
            if on_gpu >= total:
                log.info("%s fits on the gpu (%.1fGB)", model, total / _GIB)
            else:
                log.warning(
                    "%s does not fit on the gpu: %.1fGB of %.1fGB on it, "
                    "%.1fGB left on the cpu. Expect slow turns - a smaller "
                    "model or a hosted one is the fix, not a faster gpu.",
                    model, on_gpu / _GIB, total / _GIB, (total - on_gpu) / _GIB,
                )
            return
    except Exception as e:
        # Not remembered: ollama may not be up, or the model not loaded yet.
        log.debug("could not ask ollama how %s is loaded (%s)", model, e)


def vision_ok(section: dict) -> bool:
    """Can this model read a screenshot?"""
    mode = section.get("vision", "auto")
    if mode in ("on", "off"):
        return mode == "on"
    # Agent mode first, ahead of the cache. The cache is keyed on base_url and model
    if section.get("mode") == "agent":
        return config.resolves_vision(section)
    known = _VISION_CACHE.get(
        (str(section.get("base_url", "")), str(section.get("model", "")))
    )
    return config.resolves_vision(section) if known is None else known


def _settings(cfg: dict) -> dict:
    """The flat view the adapters read: the `llm` section plus the shared prompt."""
    return {
        **cfg["llm"],
        "system_prompt": persona(cfg),
        # Which screen rule the adapters should inject.
        "vision_ok": vision_ok(cfg["llm"]),
    }


# A past turn where Mellow said it couldn't see.
_BLIND_CLAIM = re.compile(
    r"""
      \b(?:can(?:'|’)?t|cannot|can\s+not|unable\s+to|don(?:'|’)?t)\b
      [^.!?]{0,40}\b(?:see|read|view|look\s+at)\b
      [^.!?]{0,30}\b(?:screen|display|desktop)\b
    | \btakes?\s+no\s+images?\b
    | \bdoes(?:\s+not|n(?:'|’)?t)\s+take\s+images?\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _drop_stale_refusals(messages: list[dict]) -> list[dict]:
    """Stop replaying "I can't see your screen" once Mellow can."""
    keep = [
        m
        for m in messages
        if not (m.get("role") == "assistant" and _BLIND_CLAIM.search(str(m.get("content", ""))))
    ]
    if len(keep) != len(messages):
        log.info("dropped %d stale can't-see turn(s) from context", len(messages) - len(keep))
    # A user turn whose answer we just removed would leave two users in a row
    out: list[dict] = []
    for m in keep:
        if out and out[-1].get("role") == m.get("role") == "user":
            out[-1] = m
        else:
            out.append(m)
    return out


async def chat(
    messages: list[dict],
    cfg: dict | None = None,
    anchor: bool = True,
    image: bytes | None = None,
) -> AsyncIterator[str]:
    """Stream an answer."""
    section = {**_settings(cfg or config.load()), "anchor": anchor}
    if image:
        import base64

        # Preserve an existing walkthrough screen mode.
        section.setdefault("screen", "seen")
        image_b64 = base64.b64encode(image).decode("ascii")
    else:
        image_b64 = None
    # Only when it can see now: if the model still takes no images
    if section.get("vision_ok"):
        messages = _drop_stale_refusals(messages)
    adapter = _anthropic if section["provider"] == "anthropic" else _openai
    log.info(
        "chat via %s model=%s%s",
        section["provider"],
        section["model"],
        " +screenshot" if image_b64 else "",
    )
    async for chunk in adapter(section, messages, image_b64):
        yield chunk


async def complete_text(prompt: str, cfg: dict, system: str) -> str:
    """A text-only completion with no persona, anchoring or tool execution."""
    section = {**cfg["llm"], "raw": True, "anchor": False, "system_prompt": system,
               "max_tokens": 4096, "temperature": 0.2}
    adapter = _anthropic if section["provider"] == "anthropic" else _openai
    return "".join([part async for part in adapter(section, [{"role": "user", "content": prompt}])]).strip()


async def complete_vision(prompt: str, cfg: dict, image: bytes) -> str:
    """One private, strict-output vision call using the configured model."""
    import base64

    section = {
        **_settings(cfg),
        "raw": True,
        "anchor": False,
        "system_prompt": (
            "You are a precise GUI locator. Follow the requested output grammar "
            "exactly and output no explanation."
        ),
        # Reasoning-capable providers bill hidden thought against this ceiling.
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    image_b64 = base64.b64encode(image).decode("ascii")
    adapter = _anthropic if section["provider"] == "anthropic" else _openai
    chunks = []
    async for chunk in adapter(
        section, [{"role": "user", "content": prompt}], image_b64
    ):
        chunks.append(chunk)
        if len("".join(chunks)) > 240:
            break
    return "".join(chunks).strip()


async def test(cfg: dict) -> str:
    """Consume a tiny real completion so a saved key is never assumed valid."""
    # Was min(12, ...), which broke every reasoning model
    probe = {
        **cfg,
        "llm": {**cfg["llm"], "max_tokens": max(1024, cfg["llm"]["max_tokens"])},
    }
    chunks = []
    # anchor=False: the identity exchanges would talk the model out of replying with one literal word
    async for chunk in chat(
        [{"role": "user", "content": "Reply with only the word connected."}],
        probe,
        anchor=False,
    ):
        chunks.append(chunk)
        if len("".join(chunks)) >= 80:
            break
    answer = "".join(chunks).strip()
    if not answer:
        raise RuntimeError("provider returned an empty response")
    return answer[:80]
