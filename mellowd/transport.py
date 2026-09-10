"""Application-owned model HTTP pool; standalone calls own a temporary client."""

import asyncio
from contextlib import asynccontextmanager

import httpx

TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_clients = {}


async def _without_cookies(request):
    # Pool sockets, not provider login state. The previous per-call clients
    # never carried response cookies into a later call or another saved key.
    request.headers.pop("cookie", None)


@asynccontextmanager
async def lifespan():
    loop = asyncio.get_running_loop()
    if loop in _clients:
        # Nested application lifespans on the same loop share ownership.
        _clients[loop][1] += 1
    else:
        _clients[loop] = [httpx.AsyncClient(
            timeout=TIMEOUT, limits=httpx.Limits(keepalive_expiry=60.0),
            event_hooks={"request": [_without_cookies]}
        ), 1]
    try:
        yield
    finally:
        entry = _clients[loop]
        entry[1] -= 1
        if not entry[1]:
            del _clients[loop]
            await entry[0].aclose()


@asynccontextmanager
async def client():
    entry = _clients.get(asyncio.get_running_loop())
    if entry is not None:
        yield entry[0]
    else:
        async with httpx.AsyncClient(timeout=TIMEOUT) as temporary:
            yield temporary
