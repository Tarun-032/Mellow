"""One app's volume, without touching anything else's."""

import logging

log = logging.getLogger(__name__)


def _sessions():
    """Every process currently holding an audio session. [] if audio is off."""
    try:
        from pycaw.pycaw import AudioUtilities

        return AudioUtilities.GetAllSessions()
    except Exception:
        log.exception("could not read the audio sessions")
        return []


def _named(session) -> str:
    """What to call this session when matching it against what they said."""
    try:
        if session.Process:
            return session.Process.name() or ""
    except Exception:
        pass
    return getattr(session, "DisplayName", "") or ""


def playing() -> list[str]:
    """Which apps have sound open right now. For the log and for a near miss."""
    return sorted({_named(s).removesuffix(".exe") for s in _sessions() if _named(s)})


def set_for(app: str, level: float) -> str:
    """Set `app`'s volume to `level` (0.0-1.0)."""
    from mellowd import point

    wanted = point.squash(app)
    if not wanted:
        return "no app named"
    touched = []
    for session in _sessions():
        name = _named(session)
        if not name:
            continue
        bare = point.squash(name.removesuffix(".exe"))
        if wanted not in bare and bare not in wanted:
            continue
        try:
            session.SimpleAudioVolume.SetMasterVolume(level, None)
            touched.append(name)
        except Exception:
            log.exception("could not set the volume on %s", name)
    if not touched:
        open_now = playing()
        log.info("volume: nothing called %r is playing; open: %s", app, open_now)
        return (
            f"nothing called {app} is playing sound"
            + (f" (right now: {', '.join(open_now)})" if open_now else "")
        )
    log.info("volume: %s -> %d%%", ", ".join(touched), round(level * 100))
    return f"set {app} to {round(level * 100)}%"
