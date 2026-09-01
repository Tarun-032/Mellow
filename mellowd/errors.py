"""Provider failures, in words the person using Mellow can act on."""

import logging

import httpx

log = logging.getLogger("mellowd.errors")


def _wording(status: int, provider: str, model: str) -> str:
    if status in (401, 403):
        return f"{provider} rejected the api key. check it in settings."
    if status == 402:
        return f"{provider} says the account is out of credit."
    if status == 404:
        # The model name is the thing that is actually wrong nine times in ten
        return (
            f"{provider} has no model called {model}."
            if model
            else f"{provider} could not find that."
        )
    if status in (408, 504):
        return f"{provider} took too long to answer."
    if status == 429:
        return f"{provider} is rate limiting me. wait a minute and ask again."
    if 500 <= status < 600:
        return f"{provider} is having trouble right now."
    return f"{provider} refused the request ({status})."


def provider_error(
    status: int, body: str, provider: str, model: str = ""
) -> RuntimeError:
    """Build the exception to raise."""
    log.warning("%s returned %s: %s", provider, status, body[:300].strip())
    return RuntimeError(_wording(status, provider, model))


def _host(exc: httpx.RequestError) -> str:
    # httpx raises from .request when the error was built without one, which is not a thing worth
    try:
        url = exc.request.url
    except RuntimeError:
        return "the provider"
    # With the port, because the host that fails this way is nearly always a local server
    return f"{url.host}:{url.port}" if url.port else url.host


def message(exc: Exception) -> str:
    """Whatever went wrong, as something worth putting in the bubble."""
    if isinstance(exc, httpx.ConnectError):
        where = _host(exc)
        # The advice is opposite for the two cases and getting it backwards is worse than giving none
        local = where.startswith(("127.", "localhost", "0.0.0.0", "[::1]"))
        advice = "is it running?" if local else "check your internet connection."
        return f"can't reach {where}. {advice}"
    if isinstance(exc, httpx.TimeoutException):
        return f"{_host(exc)} took too long to answer."
    if isinstance(exc, httpx.RequestError):
        return f"couldn't talk to {_host(exc)}."
    if isinstance(exc, RuntimeError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"
