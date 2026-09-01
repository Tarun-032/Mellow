"""What can Mellow open or change right now, and how one of them happens."""

import functools
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field

from mellowd import point

log = logging.getLogger(__name__)

# How many rows the model is shown. Same reasoning as point.MAX_ITEMS
MAX_ITEMS = 20

# Below this, nothing on the catalog is really what they said
THRESHOLD = 0.75

# Reading the Start Menu costs a PowerShell subprocess
APPS_TTL = 900.0

# Files are scanned live but not on every keystroke of a conversation.
FILES_TTL = 60.0

# How deep into Desktop/Downloads/Documents to look
FILE_DEPTH = 2
MAX_FILES = 400

# Windows' own names for the folders people ask for by name.
PLACES = {
    "Downloads": "shell:Downloads",
    "Desktop": "shell:Desktop",
    "Documents": "shell:Personal",
    "Pictures": "shell:My Pictures",
    "Music": "shell:My Music",
    "Videos": "shell:My Video",
    "This PC": "shell:MyComputerFolder",
    "Recycle Bin": "shell:RecycleBinFolder",
    "Downloads folder": "shell:Downloads",
}

# The handful of places people ask for by name rather than by URL.
SITES = {
    "Google Drive": "https://drive.google.com",
    "Gmail": "https://mail.google.com",
    "Google Calendar": "https://calendar.google.com",
    "Google Docs": "https://docs.google.com",
    "YouTube": "https://www.youtube.com",
    "Stripe dashboard": "https://dashboard.stripe.com",
    "GitHub": "https://github.com",
    "WhatsApp Web": "https://web.whatsapp.com",
    "ChatGPT": "https://chatgpt.com",
    "Claude": "https://claude.ai",
    "Google Maps": "https://maps.google.com",
    "Amazon": "https://www.amazon.in",
    "Netflix": "https://www.netflix.com",
    "LinkedIn": "https://www.linkedin.com",
}


@dataclass
class Thing:
    """One row of the catalog, and the only thing `run` will act on."""

    label: str
    kind: str
    target: str
    # What the free text after the marker means for this row
    wants: str = ""
    score: float = 0.0


# --- what is on this machine ------------------------------------------------


def _powershell(script: str, budget: float = 15.0) -> str:
    """Run one PowerShell line and give back stdout. Never raises."""
    try:
        done = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=budget,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return done.stdout or ""
    except Exception:
        log.exception("powershell failed")
        return ""


_apps: tuple[float, list[Thing]] = (0.0, [])


def apps() -> list[Thing]:
    """Everything on the Start Menu, the way Windows itself lists it."""
    global _apps
    when, cached = _apps
    if cached and time.monotonic() - when < APPS_TTL:
        return cached
    raw = _powershell(
        "Get-StartApps | ForEach-Object { $_.Name + '|' + $_.AppID }"
    )
    out = []
    for line in raw.splitlines():
        name, _, app_id = line.strip().partition("|")
        if name and app_id:
            out.append(Thing(label=name, kind="app", target=app_id))
    log.info("act: %d apps on the start menu", len(out))
    if out:
        _apps = (time.monotonic(), out)
    return out


def places() -> list[Thing]:
    """The folders people ask for by name, plus whatever drives exist."""
    out = [Thing(label=n, kind="place", target=t) for n, t in PLACES.items()]
    for letter in "CDEFGH":
        root = f"{letter}:\\"
        if os.path.isdir(root):
            out.append(Thing(label=f"{letter}: drive", kind="place", target=root))
    return out


def sites() -> list[Thing]:
    return [Thing(label=n, kind="site", target=u) for n, u in SITES.items()]


_files: tuple[float, list[Thing]] = (0.0, [])


def files() -> list[Thing]:
    """Their own documents, shallowly. This is "open my resume"."""
    global _files
    when, cached = _files
    if cached and time.monotonic() - when < FILES_TTL:
        return cached
    home = os.path.expanduser("~")
    out: list[Thing] = []
    for folder in ("Desktop", "Downloads", "Documents"):
        root = os.path.join(home, folder)
        if not os.path.isdir(root):
            continue
        for here, dirs, names in os.walk(root):
            depth = here[len(root) :].count(os.sep)
            if depth >= FILE_DEPTH:
                dirs[:] = []
            # Somebody's node_modules is not something they will ever ask for by name
            dirs[:] = [d for d in dirs if not d.startswith((".", "node_modules", "__"))]
            for name in names:
                if name.startswith("~$") or name.startswith("."):
                    continue
                out.append(
                    Thing(label=name, kind="file", target=os.path.join(here, name))
                )
                if len(out) >= MAX_FILES:
                    break
            if len(out) >= MAX_FILES:
                break
        if len(out) >= MAX_FILES:
            break
    _files = (time.monotonic(), out)
    return out


def verbs() -> list[Thing]:
    """The rows that do something *to* a thing rather than opening one."""
    return [
        Thing(
            label="Play a song or video on YouTube",
            kind="youtube",
            target="",
            wants="the song or video to play",
        ),
        Thing(
            label="Search for a song on Spotify",
            kind="spotify",
            target="",
            wants="the song to search for",
        ),
        Thing(
            label="Search Google",
            kind="google",
            target="",
            wants="what to search for",
        ),
        Thing(
            label="Set an app's volume",
            kind="volume",
            target="",
            wants="the app name and a percentage, like: spotify 50",
        ),
        Thing(
            label="Set a reminder",
            kind="remind",
            target="",
            wants="when, then what it is about, like: in 10 minutes take the "
            "pizza out. Pass on what they said - the when is worked out here, "
            "so you do not need to know the time",
        ),
        Thing(
            label="Start a focus timer (pomodoro)",
            kind="pomodoro",
            target="",
            wants="how many minutes to focus for, or nothing for the usual 25",
        ),
        Thing(
            label="Stop the focus timer",
            kind="pomodoro_stop",
            target="",
        ),
    ]


# --- the catalog ------------------------------------------------------------


def catalog(query: str) -> list[Thing]:
    """What could be done about this sentence, likeliest first."""
    try:
        rows = apps() + places() + sites() + files() + verbs()
    except Exception:
        log.exception("act: could not read the catalog")
        return []
    wanted = point.terms(query)
    if not wanted:
        return []
    for row in rows:
        if row.kind in VERBS:
            # A verb row is about the sentence's verb, never its own label
            row.score = _verb_score(query, row.kind)
        else:
            row.score = point.score(row.label, wanted)
            if row.kind == "file":
                # A file called file.jpe should not outrank the File Explorer for "open file explorer".
                row.score *= 0.9
    rows.sort(key=lambda r: (-r.score, len(r.label)))
    return [r for r in rows[:MAX_ITEMS] if r.score > 0]


_PLAY = re.compile(r"\b(play|put\s+on|listen\s+to)\b", re.IGNORECASE)
_SPOTIFY = re.compile(r"\bspotify\b", re.IGNORECASE)
# The bare word google is not a search verb
_SEARCH = re.compile('\\bsearch\\b|\\blook\\s+up\\b|\\bgoogle\\s+for\\b', re.IGNORECASE)
_VOLUME = re.compile(
    r"\bvolume\b|\b(?:louder|quieter|mute|unmute)\b"
    r"|\bturn\s+(?:\w+\s+){0,3}?(?:up|down)\b",
    re.IGNORECASE,
)

_REMIND = re.compile(
    r"\bremind\b|\breminder\b|\bwake\s+me\b|\bnudge\s+me\b"
    r"|\bset\s+a\s+timer\b|\bdon'?t\s+let\s+me\s+forget\b",
    re.IGNORECASE,
)
_POMO = re.compile(r"\bpomodoro\b|\bfocus\s+(?:timer|session|round)\b", re.IGNORECASE)
_STOP = re.compile(r"\b(?:stop|cancel|end|kill|pause)\b", re.IGNORECASE)

_OPEN_TARGET = re.compile(
    r"\b(?:open(?:\s+up)?|launch|start|run|fire\s+up|bring\s+up|go\s+to|"
    r"take\s+me\s+to|show\s+me)\b\s+(?P<target>.+)$",
    re.IGNORECASE,
)
_TRAILING_MANNERS = re.compile(
    r"(?:\s+(?:for\s+me|please|right\s+now|now))+$", re.IGNORECASE
)
_YOUTUBE_SUFFIX = re.compile(r"\s+(?:on|in)\s+youtube\s*$", re.IGNORECASE)
_MEDIA_REFERENCE = {
    "it", "that", "this", "that one", "this one", "the one", "same one",
    "that song", "this song", "the song", "that track", "this track",
    "that video", "this video",
}
_NAMED_MEDIA = re.compile(
    r"\b(?:song|track|video)\s+(?:(?:called|named|titled)\s+)?"
    r"(?P<title>[^?.,;]+?)"
    r"(?=\s*(?:[?.,;]|(?:and\s+)?(?:can|could|would|will)\s+you\b))",
    re.IGNORECASE,
)


def _words(text: str) -> str:
    """Lowercase words with stable single spaces for command-name matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def media_argument(query: str, argument: str) -> str | None:
    """Resolve "play that" to media explicitly named earlier in the request."""
    cleaned = argument.strip(" \t\r\n,.-?!\"'")
    if _words(cleaned) not in _MEDIA_REFERENCE:
        return cleaned or None
    named = _NAMED_MEDIA.search(query)
    if not named:
        return None
    title = named.group("title").strip(" \t\r\n,.-?!\"'")
    return title if title and _words(title) not in _MEDIA_REFERENCE else None

# The rows whose label is a sentence rather than a name.
VERBS = (
    "youtube", "spotify", "google", "volume", "remind", "pomodoro",
    "pomodoro_stop",
)

# Rows the sidecar cannot carry out on its own
ON_SCREEN = ("pomodoro", "pomodoro_stop")


def _verb_score(query: str, kind: str) -> float:
    """How much this sentence sounds like the verb rather than a noun."""
    if kind == "volume":
        return 0.95 if _VOLUME.search(query) else 0.0
    if kind == "spotify":
        return 0.99 if (_PLAY.search(query) and _SPOTIFY.search(query)) else 0.0
    if kind == "youtube":
        # Beaten by the spotify row when they named Spotify
        return 0.9 if (_PLAY.search(query) and not _SPOTIFY.search(query)) else 0.0
    if kind == "google":
        return 0.8 if _SEARCH.search(query) else 0.0
    if kind == "remind":
        return 0.97 if _REMIND.search(query) else 0.0
    if kind == "pomodoro":
        return 0.96 if (_POMO.search(query) and not _STOP.search(query)) else 0.0
    if kind == "pomodoro_stop":
        return 0.98 if (_POMO.search(query) and _STOP.search(query)) else 0.0
    return 0.0


def direct(query: str, things: list[Thing]) -> tuple[Thing, str] | None:
    """Resolve an unmistakable action without asking a model for a marker."""
    # "open youtube and play something" is a play request
    plays = list(_PLAY.finditer(query))
    if plays:
        found = plays[-1]
        argument = query[found.end() :].strip(" ,.-?!")
        argument = _YOUTUBE_SUFFIX.sub("", argument)
        argument = _TRAILING_MANNERS.sub("", argument).strip(" ,.-?!")
        if argument:
            kind = "spotify" if _SPOTIFY.search(query) else "youtube"
            argument = media_argument(query, argument)
            if argument is None:
                # The title may live in earlier conversation turns.
                return None
            if kind == "youtube" and _words(argument) in {
                "something", "anything", "some music", "a song", "music"
            }:
                argument = "music"
            row = next((thing for thing in things if thing.kind == kind), None)
            if row is not None:
                return row, argument

    command = _OPEN_TARGET.search(query)
    if not command:
        return None
    named = _TRAILING_MANNERS.sub("", command.group("target")).strip(" ,.-")
    normalized = _words(named)
    normalized = re.sub(r"^(?:my|the)\s+", "", normalized)
    if not normalized or " and " in normalized:
        return None

    matches: list[tuple[int, int, Thing]] = []
    for thing in things:
        if thing.kind not in ("app", "place", "site", "file"):
            continue
        label = _words(thing.label)
        aliases = {label}
        if thing.kind == "place":
            aliases.add(label.removesuffix(" folder"))
            aliases.add(label + " folder")
        if normalized in aliases:
            # Prefer the literal row name over an alias
            matches.append((2 if normalized == label else 1, -len(label), thing))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return matches[0][2], ""


def describe(things: list[Thing]) -> str:
    """The numbered list, as the model sees it."""
    rows = []
    for i, t in enumerate(things, 1):
        note = f" - say what to do, {t.wants}" if t.wants else ""
        rows.append(f"{i} {t.label} ({_WORD.get(t.kind, t.kind)}){note}")
    return (
        "\n\nON THIS COMPUTER, things you can do right now. One per line:"
        " number, what it is, and what kind of thing.\n" + "\n".join(rows)
    )


_WORD = {
    "app": "an installed app",
    "place": "a folder",
    "site": "a website",
    "file": "one of their files",
    "youtube": "plays it straight away",
    "spotify": "opens the search, cannot press play yet",
    "google": "a web search",
    "volume": "changes one app's volume",
    "remind": "sets a reminder on Mellow, which fires out loud",
    "pomodoro": "starts a focus round on Mellow",
    "pomodoro_stop": "stops the focus round",
}


def minutes(said: str) -> int | None:
    """A count of minutes out of "25", "twenty five" or "for half an hour"."""
    from mellowd import remind

    if not said:
        return None
    if remind._HALF.search("in " + said):
        return 30
    # Whitespace squashed so a spoken "twenty five" is one token to match against
    probe = re.sub(r"\s+", "", remind.join_numbers(said))
    found = re.search(remind._COUNT, probe, re.IGNORECASE)
    return remind._count(found.group(0)) if found else None


# doing


def youtube_top(query: str) -> str | None:
    """The first video for this search, as a watch URL, or None."""
    try:
        import urllib.parse

        import httpx

        r = httpx.get(
            "https://www.youtube.com/results?search_query="
            + urllib.parse.quote(query),
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en"},
            timeout=8.0,
            follow_redirects=True,
        )
        found = re.search(r'"videoId":"([A-Za-z0-9_-]{11})"', r.text)
        if not found:
            log.warning("youtube: no video id in the results for %r", query)
            return None
        log.info("youtube: %r -> %s", query, found.group(1))
        return "https://www.youtube.com/watch?v=" + found.group(1)
    except Exception:
        log.exception("youtube search failed")
        return None


def _launch(*argv: str) -> None:
    """Start something, detached, with no shell anywhere in the path."""
    subprocess.Popen(
        list(argv),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        close_fds=True,
    )


def run(thing: Thing, argument: str = "") -> str:
    """Do it."""
    kind, target = thing.kind, thing.target
    if kind == "app":
        _launch("explorer.exe", "shell:AppsFolder\\" + target)
        return f"opened {thing.label}"
    if kind == "place":
        _launch("explorer.exe", target)
        return f"opened {thing.label}"
    if kind in ("site", "file"):
        os.startfile(target)
        return f"opened {thing.label}"
    if kind == "youtube":
        import urllib.parse

        query = argument.strip() or "music"
        url = youtube_top(query)
        os.startfile(
            url
            or "https://www.youtube.com/results?search_query="
            + urllib.parse.quote_plus(query)
        )
        return f"playing {query}" if url else f"searched youtube for {query}"
    if kind == "spotify":
        # No autoplay parameter exists
        os.startfile("spotify:search:" + argument)
        return f"searched spotify for {argument}"
    if kind == "google":
        import urllib.parse

        os.startfile(
            "https://www.google.com/search?q=" + urllib.parse.quote(argument)
        )
        return f"searched for {argument}"
    if kind == "volume":
        from mellowd import volume

        app, level = _volume_args(argument)
        if app is None:
            return "could not tell which app or how loud"
        return volume.set_for(app, level)
    if kind == "remind":
        from mellowd import remind

        done = remind.add(argument)
        if done is None:
            # The one thing that cannot be guessed.
            return "could not tell when that should be"
        item, when = done
        return f"reminder {when}: {item['text']}"
    if kind == "pomodoro":
        return f"starting a {minutes(argument) or 25} minute focus round"
    if kind == "pomodoro_stop":
        return "stopping the focus timer"
    return f"nothing to do for {kind}"


def _volume_args(argument: str) -> tuple[str | None, float]:
    """("spotify", 0.5) out of "spotify 50" or "spotify to 50 percent"."""
    found = re.search(r"(\d{1,3})", argument)
    if not found:
        return None, 0.0
    level = max(0, min(100, int(found.group(1)))) / 100.0
    app = argument[: found.start()].strip(" to%").strip()
    return (app or None), level
