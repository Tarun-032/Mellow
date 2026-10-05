"""One web-searching model call that comes back as a short report with sources.

Not an agent: the provider runs its own search tool inside a single request,
the way Bluey's WebResearch.swift does it with OpenAI.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from mellowd import errors, llm, perf, transport

TIMEOUT = 45.0
MAX_PARAGRAPHS = 4
MAX_SOURCES = 5
MAX_TOKENS = 1500
# Anthropic may pause a long search turn; it is resumed by sending it back.
PAUSES = 2

INSTRUCTIONS = (
    "Research the question on the web and write a short, friendly report for a "
    "busy person. First line: a plain title of under eight words. Then one to "
    "four short paragraphs, leading with the direct answer, then the most useful "
    "details. Plain text only: no markdown, no headings, no bullet lists, and no "
    "links or citations in the text. If the question depends on a place they did "
    "not name, say which place you assumed."
)

UNSUPPORTED = "Research needs a Gemini, OpenAI, Anthropic or OpenRouter key."


@dataclass
class Report:
    question: str
    title: str
    paragraphs: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # {"title", "url"}

    @property
    def plain(self) -> str:
        return "\n\n".join([self.title, *self.paragraphs])


def kind(section: dict) -> str | None:
    """Which search adapter this llm section can use, if any."""
    if section.get("mode") != "cloud":
        return None
    if llm._google_openai(section):
        return "gemini"
    provider = section.get("provider")
    return provider if provider in ("openai", "anthropic", "openrouter") else None


def supported(cfg: dict) -> bool:
    return kind(cfg["llm"]) is not None


def _now() -> str:
    now = datetime.now().astimezone()
    return f"{now:%A %d %B %Y, %H:%M}, time zone {now.tzname()}"


def _input(question: str) -> str:
    return f"{question}\n\n(Right now it is {_now()}.)"


# Inline citations like "([site.com](https://...))", then any other markdown link.
_CITATION = re.compile(r"\s*\(\[[^\]]*\]\([^)]*\)\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE = re.compile(r"\s*\[https?://[^\]]*\]")
_TITLE_PREFIX = re.compile(r"^(?:title\s*:\s*)", re.IGNORECASE)


def clean_url(url: str) -> str:
    """Drop the tracking parameter search tools add."""
    parts = urlparse(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "utm_source"]
    return urlunparse(parts._replace(query=urlencode(query)))


def report(question: str, text: str, sources: list[tuple[str, str]]) -> Report:
    """Shape whatever the provider wrote into a title, paragraphs and sources."""
    text = _BARE.sub("", _LINK.sub(r"\1", _CITATION.sub("", text)))
    text = text.replace("**", "").replace("#", "").replace("__", "")
    blocks = [line.strip().lstrip("-* ").strip() for line in text.splitlines()]
    blocks = [b for b in blocks if b]
    if not blocks:
        raise RuntimeError("the research came back empty.")
    title = _TITLE_PREFIX.sub("", blocks[0]).rstrip(".:")
    seen: set[str] = set()
    kept: list[dict] = []
    for title_text, url in sources:
        if not url or not url.startswith(("http://", "https://")):
            continue
        url = clean_url(url)
        if url in seen:
            continue
        seen.add(url)
        kept.append({"title": (title_text or urlparse(url).hostname or url).strip(), "url": url})
    return Report(question, title, blocks[1 : 1 + MAX_PARAGRAPHS], kept[:MAX_SOURCES])


# One parser per provider: (written text, [(title, url)]). Pure, so the check
# script can feed them canned responses.

def parse_gemini(body: dict) -> tuple[str, list[tuple[str, str]]]:
    candidate = (body.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    chunks = (candidate.get("groundingMetadata") or {}).get("groundingChunks") or []
    return text, [(c["web"].get("title", ""), c["web"].get("uri", "")) for c in chunks if c.get("web")]


def parse_openai(body: dict) -> tuple[str, list[tuple[str, str]]]:
    text, sources = "", []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            text += part.get("text", "")
            for note in part.get("annotations") or []:
                if note.get("type") == "url_citation":
                    sources.append((note.get("title", ""), note.get("url", "")))
    return text, sources


def parse_anthropic(content: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    # Text written before the last search is "I'll look that up" chatter.
    last = max((i for i, b in enumerate(content) if b.get("type") == "web_search_tool_result"), default=-1)
    text, cited, found = "", [], []
    for i, block in enumerate(content):
        if block.get("type") == "web_search_tool_result" and isinstance(block.get("content"), list):
            found += [(r.get("title", ""), r.get("url", "")) for r in block["content"]]
        elif block.get("type") == "text" and i > last:
            text += block.get("text", "")
            cited += [(c.get("title", ""), c.get("url", "")) for c in block.get("citations") or []]
    # What the answer actually cites first, then what the search turned up.
    return text, cited + found


def parse_openrouter(body: dict) -> tuple[str, list[tuple[str, str]]]:
    message = ((body.get("choices") or [{}])[0]).get("message") or {}
    sources = [
        (n["url_citation"].get("title", ""), n["url_citation"].get("url", ""))
        for n in message.get("annotations") or []
        if n.get("type") == "url_citation" and n.get("url_citation")
    ]
    return message.get("content") or "", sources


async def _post(section: dict, label: str, url: str, headers: dict, payload: dict) -> dict:
    async with transport.client() as client:
        r = await client.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    if r.status_code == 429 and label == "Gemini":
        # Measured: plain calls succeed on a free key while every grounded one is
        # refused, so "wait a minute" would be the wrong advice.
        errors.provider_error(r.status_code, r.text, label, section.get("model", ""))
        raise RuntimeError(
            "Gemini won't search the web on this key. Search grounding usually needs "
            "billing turned on for the key's Google project."
        )
    if not r.is_success:
        raise errors.provider_error(r.status_code, r.text, label, section.get("model", ""))
    return r.json()


async def _gemini(section: dict, question: str):
    base = (section.get("base_url") or "").rstrip("/")
    base = re.sub(r"/openai$", "", base) or "https://generativelanguage.googleapis.com/v1beta"
    model = section["model"].removeprefix("models/")
    body = await _post(section, "Gemini", f"{base}/models/{model}:generateContent",
                       {"x-goog-api-key": section.get("api_key", "")}, {
        "systemInstruction": {"parts": [{"text": INSTRUCTIONS}]},
        "contents": [{"role": "user", "parts": [{"text": _input(question)}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"maxOutputTokens": MAX_TOKENS},
    })
    return parse_gemini(body)


async def _openai(section: dict, question: str):
    base = (section.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    body = await _post(section, "OpenAI", f"{base}/responses",
                       {"Authorization": f"Bearer {section.get('api_key', '')}"}, {
        "model": section["model"],
        "instructions": INSTRUCTIONS,
        "input": _input(question),
        "tools": [{"type": "web_search", "search_context_size": "low"}],
        "max_output_tokens": MAX_TOKENS,
    })
    return parse_openai(body)


async def _anthropic(section: dict, question: str):
    base = (section.get("base_url") or "https://api.anthropic.com/v1").rstrip("/")
    headers = {"x-api-key": section.get("api_key", ""), "anthropic-version": "2023-06-01"}
    messages = [{"role": "user", "content": _input(question)}]
    content: list[dict] = []
    for _ in range(PAUSES + 1):
        body = await _post(section, "Anthropic", f"{base}/messages", headers, {
            "model": section["model"],
            "system": INSTRUCTIONS,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        })
        content += body.get("content") or []
        if body.get("stop_reason") != "pause_turn":
            break
        messages = [*messages[:1], {"role": "assistant", "content": content}]
    return parse_anthropic(content)


async def _openrouter(section: dict, question: str):
    base = (section.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")
    body = await _post(section, "OpenRouter", f"{base}/chat/completions",
                       {"Authorization": f"Bearer {section.get('api_key', '')}"}, {
        "model": section["model"],
        "messages": [{"role": "system", "content": INSTRUCTIONS},
                     {"role": "user", "content": _input(question)}],
        "tools": [{"type": "openrouter:web_search", "parameters": {"max_results": MAX_SOURCES}}],
        "max_tokens": MAX_TOKENS,
    })
    return parse_openrouter(body)


_ADAPTERS = {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic, "openrouter": _openrouter}


@perf.timed("research")
async def run(question: str, cfg: dict) -> Report:
    section = cfg["llm"]
    which = kind(section)
    if which is None:
        raise RuntimeError(UNSUPPORTED)
    with perf.purpose("research"):
        text, sources = await _ADAPTERS[which](section, question)
    return report(question, text, sources)
